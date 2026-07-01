"""Video I/O utility functions.

Provides thin, well-documented wrappers around OpenCV's VideoCapture
and VideoWriter APIs, used throughout the detection/tracking/dashboard
pipeline. All functions accept and return `pathlib.Path` objects
rather than raw strings to avoid hardcoded path assumptions.

Note (macOS / Apple Silicon): some FourCC codecs (e.g. "avc1") are not
guaranteed to be available in every OpenCV build. `create_video_writer`
automatically falls back to "mp4v" if the requested codec fails to
open, so this remains robust across platforms.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK_CODEC = "mp4v"


@dataclass(frozen=True)
class VideoInfo:
    """Metadata describing a video file."""

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str


def load_video(video_path: Path) -> cv2.VideoCapture:
    """Opens a video file for reading.

    Args:
        video_path: Path to the input video file.

    Returns:
        An opened cv2.VideoCapture instance. Caller must release it
        (or use read_frames(), which does this automatically).

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video file cannot be opened by OpenCV.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        logger.error("OpenCV failed to open video: %s", video_path)
        raise RuntimeError(f"Could not open video file: {video_path}")

    logger.info("Opened video: %s", video_path)
    return capture


def get_video_info(video_path: Path) -> VideoInfo:
    """Reads metadata for a video file without decoding all frames.

    Args:
        video_path: Path to the input video file.

    Returns:
        A VideoInfo instance describing the video.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video file cannot be opened by OpenCV.
    """
    capture = load_video(video_path)
    try:
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int > 0:
            codec = (
                chr(fourcc_int & 0xFF)
                + chr((fourcc_int >> 8) & 0xFF)
                + chr((fourcc_int >> 16) & 0xFF)
                + chr((fourcc_int >> 24) & 0xFF)
            ).strip()
        else:
            codec = "unknown"

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            logger.warning(
                "FPS reported as %s for %s; defaulting to 30.0", fps, video_path
            )
            fps = 30.0

        info = VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            codec=codec,
        )
    finally:
        capture.release()

    logger.debug("Video info for %s: %s", video_path, info)
    return info


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """Creates a cv2.VideoWriter for saving processed video frames.

    If the requested codec fails to open (common on some macOS OpenCV
    builds for certain FourCCs), this automatically retries once with
    the "mp4v" fallback codec before raising.

    Args:
        output_path: Destination path for the output video file.
        width: Output frame width in pixels (must be > 0).
        height: Output frame height in pixels (must be > 0).
        fps: Output frames per second (must be > 0).
        codec: Four-character codec code (e.g. "mp4v", "avc1", "XVID").

    Returns:
        An opened cv2.VideoWriter ready to accept frames via .write().

    Raises:
        ValueError: If width, height, or fps are not positive.
        RuntimeError: If the writer fails to open, even after fallback.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size: {width}x{height}")
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _try_open(fourcc_code: str) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    writer = _try_open(codec)

    if not writer.isOpened() and codec != _FALLBACK_CODEC:
        logger.warning(
            "Codec '%s' failed to open writer for %s; retrying with '%s'.",
            codec,
            output_path,
            _FALLBACK_CODEC,
        )
        writer.release()
        writer = _try_open(_FALLBACK_CODEC)
        codec = _FALLBACK_CODEC

    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for %s", output_path)
        raise RuntimeError(f"Could not create video writer for: {output_path}")

    logger.info(
        "Created video writer: %s (%dx%d @ %.2f fps, codec=%s)",
        output_path,
        width,
        height,
        fps,
        codec,
    )
    return writer


def read_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Lazily yields frames from a video file, one at a time.

    Args:
        video_path: Path to the input video file.

    Yields:
        Successive BGR frames as numpy arrays, shape (H, W, 3).

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video file cannot be opened.
    """
    capture = load_video(video_path)
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_index += 1
            yield frame
    finally:
        capture.release()
        logger.info(
            "Finished reading %d frames from %s", frame_index, video_path
        )


def save_video(
    frames: Iterator[np.ndarray],
    output_path: Path,
    fps: float,
    codec: str = "mp4v",
) -> Path:
    """Writes an iterable/generator of frames to disk as a video file.

    Args:
        frames: An iterable of BGR frames with consistent width/height.
        output_path: Destination path for the output video.
        fps: Frames per second for the output video (must be > 0).
        codec: Four-character codec code for encoding.

    Returns:
        The output_path the video was written to.

    Raises:
        ValueError: If frames is empty (yields no frames) or fps <= 0.
        RuntimeError: If the video writer cannot be created.
    """
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    output_path = Path(output_path)
    writer: cv2.VideoWriter | None = None
    frame_count = 0

    try:
        for frame in frames:
            if frame is None:
                continue
            if writer is None:
                height, width = frame.shape[:2]
                writer = create_video_writer(
                    output_path, width, height, fps, codec
                )
            writer.write(frame)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        logger.error("No frames were provided to save_video for %s", output_path)
        raise ValueError("Cannot save video: no frames were provided.")

    logger.info("Saved %d frames to %s", frame_count, output_path)
    return output_path