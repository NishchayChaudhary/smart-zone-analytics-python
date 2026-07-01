"""Utility helpers for Smart Zone Analytics.

Exposes video I/O helpers used across the detection, tracking, and
dashboard modules.
"""

from src.utils.config_loader import PROJECT_ROOT, load_config, resolve_path
from src.utils.video_utils import (
    VideoInfo,
    create_video_writer,
    get_video_info,
    load_video,
    read_frames,
    save_frame,
    save_video,
)

__all__ = [
    "PROJECT_ROOT",
    "VideoInfo",
    "create_video_writer",
    "get_video_info",
    "load_config",
    "load_video",
    "read_frames",
    "resolve_path",
    "save_frame",
    "save_video",
]