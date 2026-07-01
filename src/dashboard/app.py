"""
src/dashboard/app.py

Streamlit dashboard for Smart Zone Analytics.

Runs the same detect -> track -> zones -> analytics pipeline used by
main.py, reusing the exact same src.detection / src.tracking /
src.analytics components, but renders live video + occupancy metrics
in a Streamlit UI instead of writing an output video file.

Usage:
    streamlit run src/dashboard/app.py

Note: Streamlit reruns the whole script top-to-bottom on every widget
interaction. A true "Stop" mid-processing button needs threads or
st.fragment-based patterns, which is beyond wiring existing pipeline
components together, so this dashboard instead exposes a "Max frames
to process" limit to keep a single run bounded.
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from src.analytics import OccupancyTracker, PeopleCounter, ZoneManager
from src.dashboard.components import render_frame, render_kpi_row, render_zone_table
from src.dashboard.metrics import summary_to_kpis, summary_to_zone_rows
from src.detection import YOLODetector
from src.tracking import Tracker
from src.utils import load_config, load_video, resolve_path

logger = logging.getLogger("smart_zone_analytics.dashboard")


@st.cache_resource(show_spinner=False)
def _load_app_config() -> dict:
    """Loads config/config.yaml once per Streamlit server process."""
    return load_config()


def _build_pipeline(config: dict):
    """Instantiates detector/tracker/zone/analytics components from config.

    Mirrors main.py's build_pipeline() so both entry points stay
    behaviorally identical.
    """
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


def main() -> None:
    config = _load_app_config()
    dashboard_cfg = config.get("dashboard", {})
    video_cfg = config.get("video", {})

    st.set_page_config(
        page_title=dashboard_cfg.get("title", "Smart Zone Analytics"), layout="wide"
    )
    st.title(dashboard_cfg.get("title", "Smart Zone Analytics"))

    video_path_str = st.sidebar.text_input(
        "Input video path", value=video_cfg.get("video_path", "videos/sample.mp4")
    )
    max_frames = st.sidebar.number_input(
        "Max frames to process (0 = all)", min_value=0, value=300, step=50
    )
    refresh_interval = float(dashboard_cfg.get("refresh_interval_seconds", 1.0))
    run_clicked = st.sidebar.button("Run pipeline")

    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = _build_pipeline(config)
        except Exception as exc:  # model load / config errors
            st.error(f"Failed to initialize pipeline: {exc}")
            return

    detector, tracker, zone_manager, people_counter, occupancy_tracker = st.session_state.pipeline

    kpi_placeholder = st.empty()
    zone_table_placeholder = st.empty()
    frame_placeholder = st.empty()

    if not run_clicked:
        st.info("Configure the video path in the sidebar and click **Run pipeline** to start.")
        return

    try:
        input_path = resolve_path(video_path_str)
        cap = load_video(input_path)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return

    frame_index = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            detections = detector.predict_frame(frame)
            tracked_objects = tracker.update(detections)
            people_counter.update(tracked_objects)
            occupancy_tracker.update(tracked_objects)

            annotated = zone_manager.draw_zones(frame)
            annotated = detector.draw_boxes(annotated, detections)

            summary = occupancy_tracker.get_summary()
            with kpi_placeholder.container():
                render_kpi_row(summary_to_kpis(summary))
            with zone_table_placeholder.container():
                render_zone_table(summary_to_zone_rows(summary))
            render_frame(annotated, frame_placeholder)

            if max_frames and frame_index >= max_frames:
                st.warning(f"Stopped after reaching the {max_frames}-frame limit.")
                break

            time.sleep(max(refresh_interval, 0.0))
    finally:
        cap.release()

    st.success(f"Finished processing {frame_index} frames.")


if __name__ == "__main__":
    main()
