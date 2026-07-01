"""Object tracking module.

This module defines a Tracker abstraction that assigns persistent IDs
to detections across frames. It currently ships with a lightweight
centroid-distance placeholder tracker so the rest of the pipeline
(zones, analytics, dashboard) can be built and tested end-to-end. The
placeholder is designed to be swapped out for a real ByteTrack
implementation with minimal changes to calling code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count

import numpy as np

from src.detection.detect import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A single tracked object with a persistent identity."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    missed_frames: int = 0


@dataclass
class _InternalTrack:
    """Internal bookkeeping for the placeholder tracker."""

    track_id: int
    centroid: np.ndarray
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    missed_frames: int = 0


def _centroid(xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Computes the (x, y) centroid of a bounding box."""
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


class Tracker:
    """Multi-object tracker with a pluggable backend."""

    def __init__(
        self,
        backend: str = "centroid",
        max_distance: float = 80.0,
        max_missed_frames: int = 30,
    ) -> None:
        """Initializes the tracker.

        Args:
            backend: "centroid" (default placeholder) or "bytetrack".
            max_distance: Max pixel distance to associate a detection
                with an existing track.
            max_missed_frames: Frames a track may go unmatched before
                being dropped.

        Raises:
            ValueError: If an unsupported backend is requested, or if
                max_distance / max_missed_frames are negative.
        """
        if backend not in {"centroid", "bytetrack"}:
            raise ValueError(f"Unsupported tracking backend: {backend}")
        if max_distance <= 0:
            raise ValueError("max_distance must be a positive number.")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative.")

        self.backend = backend
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self._tracks: dict[int, _InternalTrack] = {}
        self._id_counter = count(1)

        logger.info("Initialized Tracker with backend='%s'.", backend)

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Public entry point to update tracks with new detections."""
        return self.update(detections)

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Updates internal track state with a new frame's detections.

        Args:
            detections: Detections produced for the current frame.
                An empty list is valid and simply ages/drops tracks.

        Returns:
            List of currently active TrackedObject instances.
        """
        if detections is None:
            detections = []

        if self.backend == "bytetrack":
            return self._update_bytetrack(detections)
        return self._update_centroid(detections)

    def reset(self) -> None:
        """Clears all track state, starting fresh from an empty tracker."""
        logger.info(
            "Resetting tracker state (%d active tracks cleared).",
            len(self._tracks),
        )
        self._tracks.clear()
        self._id_counter = count(1)

    def _update_centroid(
        self, detections: list[Detection]
    ) -> list[TrackedObject]:
        """Placeholder nearest-centroid tracking implementation."""
        unmatched_detection_idxs = set(range(len(detections)))
        detection_centroids = [_centroid(det.xyxy) for det in detections]

        for trk in list(self._tracks.values()):
            best_idx: int | None = None
            best_dist = self.max_distance

            for idx in unmatched_detection_idxs:
                dist = float(
                    np.linalg.norm(trk.centroid - detection_centroids[idx])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                det = detections[best_idx]
                trk.xyxy = det.xyxy
                trk.confidence = det.confidence
                trk.class_id = det.class_id
                trk.class_name = det.class_name
                trk.centroid = detection_centroids[best_idx]
                trk.age += 1
                trk.missed_frames = 0
                unmatched_detection_idxs.discard(best_idx)
            else:
                trk.missed_frames += 1

        for track_id in list(self._tracks.keys()):
            if self._tracks[track_id].missed_frames > self.max_missed_frames:
                logger.debug("Dropping stale track_id=%d", track_id)
                del self._tracks[track_id]

        for idx in unmatched_detection_idxs:
            det = detections[idx]
            new_id = next(self._id_counter)
            self._tracks[new_id] = _InternalTrack(
                track_id=new_id,
                centroid=detection_centroids[idx],
                xyxy=det.xyxy,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                age=1,
                missed_frames=0,
            )
            logger.debug("Created new track_id=%d", new_id)

        return [
            TrackedObject(
                track_id=trk.track_id,
                xyxy=trk.xyxy,
                confidence=trk.confidence,
                class_id=trk.class_id,
                class_name=trk.class_name,
                age=trk.age,
                missed_frames=trk.missed_frames,
            )
            for trk in self._tracks.values()
        ]

    def _update_bytetrack(
        self, detections: list[Detection]
    ) -> list[TrackedObject]:
        """Integration point for a real ByteTrack backend.

        Raises:
            NotImplementedError: Always, until ByteTrack is wired in.
        """
        raise NotImplementedError(
            "ByteTrack backend is not yet implemented. Implement "
            "_update_bytetrack() to plug in a real ByteTrack tracker, "
            "or use backend='centroid' for now."
        )