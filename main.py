#!/usr/bin/env python3
"""
main.py

End-to-end Smart Zone Analytics pipeline entry point.

Wires together the pipeline stages that already exist under src/ —
no new detection/tracking/analytics logic is introduced here, only
orchestration, config loading, and frame drawing:

    src.detection.YOLODetector    -> per-frame person detection
    src.tracking.Tracker          -> assigns persistent IDs to detections
    src.analytics.ZoneManager     -> polygon zone definitions + point tests
    src.analytics.PeopleCounter   -> per-zone entry/exit counting
    src.analytics.OccupancyTracker-> live + historical occupancy summary
    src.utils.video_utils         -> video I/O helpers
    src.utils.config_loader       -> config/config.yaml loading

Usage:
    python main.py
    python main.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.analytics import OccupancyTracker, PeopleCounter, ZoneManager
from src.analytics.occupancy import AnalyticsSummary
from src.detection import YOLODetector
from src.tracking import Tracker, TrackedObject
from src.utils import (
    create_video_writer,
    get_video_info,
    load_config,
    load_video,
    resolve_path,
    save_frame,
)

logger = logging.getLogger("smart_zone_analytics")


def setup_logging(log_config: dict) -> None:
    """Configures root logging based on the `logging` section of config.yaml."""
    level_name = str(log_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = log_config.get("log_file")
    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def draw_tracks(frame, tracked_objects: list[TrackedObject]):
    """Draws track boxes + persistent IDs on a copy of the frame.

    This is presentation logic for the orchestrated pipeline (main.py),
    kept out of src.tracking since Tracker itself has no drawing
    responsibility.
    """
    annotated = frame.copy()
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
        label = f"ID {obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA,
        )
    return annotated


def draw_analytics_overlay(frame, summary: AnalyticsSummary):
    """Draws a small per-zone occupancy / entries / exits HUD in the corner."""
    annotated = frame.copy()
    y = 24
    cv2.putText(
        annotated,
        f"Total occupancy: {summary.total_current_occupancy}",
        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    for zone_name, zs in summary.zones.items():
        y += 22
        text = (
            f"{zone_name}: occ={zs.current_occupancy} "
            f"in={zs.total_entries} out={zs.total_exits}"
        )
        cv2.putText(
            annotated, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def build_pipeline(config: dict):
    """Instantiates all pipeline components from a loaded config dict."""
    model_cfg = config.get("model", {})
    tracker_cfg = config.get("tracker", {})
    zones_cfg = config.get("zones", [])

    detector = YOLODetector(
        model_path=str(resolve_path(model_cfg.get("model_path", "models/yolov8n.pt"))),
        confidence_threshold=float(model_cfg.get("confidence", 0.4)),
        device=model_cfg.get("device"),
        classes=model_cfg.get("classes"),
    )

    tracker = Tracker(
        backend=tracker_cfg.get("backend", "centroid"),
        max_distance=float(tracker_cfg.get("max_distance", 80.0)),
        max_missed_frames=int(tracker_cfg.get("max_missed_frames", 30)),
    )

    zone_manager = ZoneManager(zones_cfg)
    people_counter = PeopleCounter(zone_manager)
    occupancy_tracker = OccupancyTracker(people_counter)

    return detector, tracker, zone_manager, people_counter, occupancy_tracker


def run(config_path: Path) -> int:
    """Runs the full detect -> track -> zone -> analytics pipeline over the
    configured input video and writes an annotated output video.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"[main] Config file not found: {config_path}", file=sys.stderr)
        return 1

    setup_logging(config.get("logging", {}))
    logger.info("Starting Smart Zone Analytics pipeline (config=%s)", config_path)

    video_cfg = config.get("video", {})
    input_path = resolve_path(video_cfg.get("video_path", "videos/sample.mp4"))
    output_path = resolve_path(video_cfg.get("output_path", "outputs/processed_video.mp4"))
    save_video_flag = bool(video_cfg.get("save_video", True))
    save_frames_flag = bool(video_cfg.get("save_frames", False))
    output_directory = resolve_path(video_cfg.get("output_directory", "outputs"))
    output_codec = video_cfg.get("output_codec", "mp4v")

    try:
        detector, tracker, zone_manager, people_counter, occupancy_tracker = build_pipeline(config)
    except Exception:
        logger.exception("Failed to initialize pipeline components.")
        return 1

    try:
        info = get_video_info(input_path)
    except (FileNotFoundError, RuntimeError):
        logger.exception("Could not read input video: %s", input_path)
        return 1

    cap = load_video(input_path)

    writer = None
    if save_video_flag:
        try:
            writer = create_video_writer(
                output_path, info.width, info.height, info.fps, codec=output_codec
            )
        except (ValueError, RuntimeError):
            logger.exception("Could not create output video writer for %s", output_path)
            cap.release()
            return 1

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            detections = detector.predict_frame(frame)
            tracked_objects = tracker.update(detections)
           

            people_counter.update(tracked_objects)
            occupancy_tracker.update(tracked_objects)

            annotated = zone_manager.draw_zones(frame)
            annotated = draw_tracks(annotated, tracked_objects)
            annotated = draw_analytics_overlay(annotated, occupancy_tracker.get_summary())

            if writer is not None:
                writer.write(annotated)

            if save_frames_flag:
                frame_path = output_directory / "frames" / f"frame_{frame_index:06d}.jpg"
                save_frame(annotated, frame_path)

            if frame_index % 30 == 0:
                logger.info("Processed %d frames...", frame_index)

    except Exception:
        logger.exception("Pipeline failed while processing frame %d", frame_index)
        return 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = occupancy_tracker.get_summary()
    logger.info("Pipeline finished. Processed %d frames.", frame_index)
    logger.info(
        "Final summary: total_occupancy=%d, total_entries=%d, total_exits=%d",
        summary.total_current_occupancy,
        summary.total_entries,
        summary.total_exits,
    )
    for zone_name, zs in summary.zones.items():
        logger.info(
            "  Zone '%s': occupancy=%d max=%d entries=%d exits=%d",
            zone_name, zs.current_occupancy, zs.max_occupancy,
            zs.total_entries, zs.total_exits,
        )

    return 0


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "config" / "config.yaml"
    parser = argparse.ArgumentParser(description="Smart Zone Analytics pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = run(Path(args.config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
