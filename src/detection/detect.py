"""
detect.py

YOLOv8-based person detection module for Smart Zone Analytics.
Phase 1 scope: CPU / Apple MPS friendly, no CUDA dependency.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


class Detection:
    """Single detection result for one frame."""

    def __init__(self, xyxy: np.ndarray, confidence: float, class_id: int):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self) -> str:
        return (
            f"Detection(bbox={self.xyxy.tolist()}, "
            f"conf={self.confidence:.2f}, class_id={self.class_id})"
        )


class YOLODetector:
    """
    YOLOv8 wrapper handling model loading, frame-level and video-level
    inference, and visualization for the detection stage of the pipeline.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes if classes is not None else [PERSON_CLASS_ID]
        self.model = self.load_model()

    def load_model(self) -> YOLO:
        """Load YOLOv8 weights. Auto-downloads standard weights (e.g. yolov8n.pt)
        if the file doesn't exist locally and the name matches a known model."""
        if not self.model_path.exists():
            print(
                f"[YOLODetector] '{self.model_path}' not found locally — "
                f"ultralytics will attempt to download it if it's a standard model name."
            )
        model = YOLO(str(self.model_path))
        return model

    def predict_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame and return filtered Detection objects."""
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
    def draw_boxes(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes + confidence labels on a copy of the frame."""
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

    def predict_video(
        self,
        source_path: str,
        output_path: str,
        show_live: bool = False,
    ) -> None:
        """
        Run detection over an entire video file and write an annotated
        copy to output_path.

        Args:
            source_path: path to input video (or "0" for webcam).
            output_path: path to save the annotated output video.
            show_live: if True, also displays the video in a window while processing.
        """
        source = 0 if source_path == "0" else source_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.predict_frame(frame)
                annotated = self.draw_boxes(frame, detections)
                writer.write(annotated)
                frame_count += 1

                if show_live:
                    cv2.imshow("YOLODetector — predict_video", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
                        break

                if frame_count % 30 == 0:
                    print(f"[YOLODetector] Processed {frame_count} frames...")

        finally:
            cap.release()
            writer.release()
            if show_live:
                cv2.destroyAllWindows()

        print(f"[YOLODetector] Done. {frame_count} frames written to '{output_path}'.")