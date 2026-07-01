"""
demo_detect.py

Standalone demo entry point: runs the YOLODetector on videos/input.mp4
and saves the annotated output to outputs/.

Usage:
    python demo_detect.py
"""

from src.detection.detect import YOLODetector

INPUT_VIDEO = "videos/input.mp4"
OUTPUT_VIDEO = "outputs/detection_output.mp4"
MODEL_PATH = "models/yolov8n.pt"


def main() -> None:
    detector = YOLODetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.4,
    )

    detector.predict_video(
        source_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        show_live=False,
    )


if __name__ == "__main__":
    main()