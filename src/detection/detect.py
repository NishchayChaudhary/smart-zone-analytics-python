"""
detect.py

Basic YOLOv8 inference wrapper for the Smart Zone Analytics pipeline.

Phase 1 scope: person detection only, CPU / Apple MPS friendly.
This module is intentionally kept detection-only — tracking (ByteTrack)
is wired in separately via src/tracking/, using this class's output.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# COCO class index for "person"
PERSON_CLASS_ID = 0


class Detection:
    """Single detection result for one frame."""

    def __init__(self, xyxy: np.ndarray, confidence: float, class_id: int):
        self.xyxy = xyxy              # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self) -> str:
        return (
            f"Detection(bbox={self.xyxy.tolist()}, "
            f"conf={self.confidence:.2f}, class_id={self.class_id})"
        )


class Detector:
    """
    Thin wrapper around an Ultralytics YOLO model for person detection.

    Usage:
        detector = Detector(model_path="models/yolov8n.pt")
        detections = detector.infer(frame)
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ):
        """
        Args:
            model_path: path to a YOLOv8 .pt weights file.
            confidence_threshold: minimum confidence to keep a detection.
            device: "cpu", "mps", or "cuda". If None, ultralytics auto-selects.
            classes: list of class IDs to keep. Defaults to [person] only.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes if classes is not None else [PERSON_CLASS_ID]

        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                f"Place a YOLOv8 .pt file there (e.g. yolov8n.pt) or pass "
                f"a different --model path."
            )
        model = YOLO(str(self.model_path))
        return model

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (as read by cv2.VideoCapture).

        Returns:
            List of Detection objects filtered to self.classes and
            self.confidence_threshold.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append(Detection(xyxy, conf, cls_id))

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on a copy of the frame for quick visual checks."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
        return annotated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick YOLOv8 detection test")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to a video file, or '0' for webcam",
    )
    parser.add_argument(
        "--model", type=str, default="models/yolov8n.pt",
        help="Path to YOLOv8 weights",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device: cpu, mps, or cuda (auto if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Standalone smoke test: runs the detector on a video/webcam and shows
    live bounding boxes. Not the production entry point — this just
    confirms the detection module works in isolation before tracking,
    zones, etc. are wired in.
    """
    args = _parse_args()

    source = 0 if args.source == "0" else args.source
    detector = Detector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.infer(frame)
        annotated = detector.draw_detections(frame, detections)

        cv2.imshow("Smart Zone Analytics — Detection Test", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()