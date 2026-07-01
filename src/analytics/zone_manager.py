"""Zone management for Smart Zone Analytics.

Defines polygon-based zones of interest within a video frame and
provides utilities to draw them and test whether a point (typically a
tracked person's centroid) falls inside any of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A single polygon zone of interest.

    Attributes:
        name: Unique, human-readable name for the zone.
        polygon: Ordered (N, 2) array of (x, y) vertices defining the
            polygon boundary, in pixel coordinates.
        color: BGR color used when drawing this zone.
    """

    name: str
    polygon: np.ndarray
    color: tuple[int, int, int] = (0, 255, 255)

    def __post_init__(self) -> None:
        """Validates the polygon shape after initialization.

        Raises:
            ValueError: If the polygon does not have shape (N, 2) with
                at least 3 vertices.
        """
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise ValueError(
                f"Zone '{self.name}' polygon must have shape (N, 2), "
                f"got {self.polygon.shape}."
            )
        if self.polygon.shape[0] < 3:
            raise ValueError(
                f"Zone '{self.name}' polygon must have at least 3 vertices."
            )


class ZoneManager:
    """Manages a collection of named polygon zones.

    Provides helpers to construct zones from raw coordinate lists
    (e.g. loaded from config.yaml), draw them onto frames, and check
    point-in-polygon membership using OpenCV.

    Attributes:
        zones: Dictionary mapping zone name to its Zone instance.
    """

    def __init__(self, zones_config: list[dict] | None = None) -> None:
        """Initializes the ZoneManager, optionally from a config list.

        Args:
            zones_config: Optional list of dicts, each containing
                "name" (str) and "polygon" (list of [x, y] pairs), as
                typically loaded from config.yaml's `zones` section.

        Raises:
            ValueError: If any zone definition is missing required
                keys or has a malformed polygon.
        """
        self.zones: dict[str, Zone] = {}

        if zones_config:
            for entry in zones_config:
                try:
                    name = entry["name"]
                    polygon = entry["polygon"]
                except KeyError as exc:
                    raise ValueError(
                        f"Zone config entry missing required key: {exc}"
                    ) from exc
                self.add_zone(name, polygon)

        logger.info(
            "Initialized ZoneManager with %d zone(s): %s",
            len(self.zones),
            list(self.zones.keys()),
        )

    def add_zone(
        self,
        name: str,
        polygon: list[list[float]] | np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        """Adds a new polygon zone.

        Args:
            name: Unique name identifying the zone.
            polygon: List of [x, y] vertex pairs, or an (N, 2) numpy
                array, defining the polygon boundary.
            color: BGR color used when drawing this zone.

        Raises:
            ValueError: If a zone with this name already exists, or if
                the polygon has fewer than 3 vertices.
        """
        if name in self.zones:
            raise ValueError(f"Zone '{name}' already exists.")

        polygon_arr = np.asarray(polygon, dtype=np.float32)
        zone = Zone(name=name, polygon=polygon_arr, color=color)
        self.zones[name] = zone
        logger.debug("Added zone '%s' with %d vertices.", name, len(polygon_arr))

    def remove_zone(self, name: str) -> None:
        """Removes a zone by name.

        Args:
            name: Name of the zone to remove.

        Raises:
            KeyError: If no zone with this name exists.
        """
        if name not in self.zones:
            raise KeyError(f"Zone '{name}' does not exist.")
        del self.zones[name]
        logger.debug("Removed zone '%s'.", name)

    def get_zone_names(self) -> list[str]:
        """Returns the names of all registered zones.

        Returns:
            List of zone names, in insertion order.
        """
        return list(self.zones.keys())

    def is_inside(self, point: tuple[float, float], zone_name: str) -> bool:
        """Checks whether a point lies inside a named zone.

        Uses cv2.pointPolygonTest for a precise geometric test that
        correctly handles convex and concave polygons.

        Args:
            point: (x, y) pixel coordinates to test.
            zone_name: Name of the zone to test against.

        Returns:
            True if the point lies inside or on the boundary of the
            zone's polygon, False otherwise.

        Raises:
            KeyError: If no zone with this name exists.
        """
        if zone_name not in self.zones:
            raise KeyError(f"Zone '{zone_name}' does not exist.")

        zone = self.zones[zone_name]
        result = cv2.pointPolygonTest(zone.polygon, point, measureDist=False)
        return result >= 0

    def get_zones_containing_point(
        self, point: tuple[float, float]
    ) -> list[str]:
        """Finds all zones that contain a given point.

        Args:
            point: (x, y) pixel coordinates to test.

        Returns:
            List of zone names whose polygon contains the point. Empty
            if the point falls outside every zone.
        """
        return [name for name in self.zones if self.is_inside(point, name)]

    def draw_zones(
        self,
        frame: np.ndarray,
        thickness: int = 2,
        alpha: float = 0.15,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draws all registered zones onto a copy of the given frame.

        Each zone is rendered as a semi-transparent filled polygon
        with a solid outline and an optional name label near its
        first vertex.

        Args:
            frame: Original BGR frame to draw onto (not modified).
            thickness: Line thickness for zone outlines.
            alpha: Opacity of the filled zone overlay, in [0, 1].
            show_labels: Whether to draw the zone name near its first
                vertex.

        Returns:
            A new numpy array (copy of frame) with zones drawn.
        """
        if not self.zones:
            return frame.copy()

        overlay = frame.copy()
        for zone in self.zones.values():
            points = zone.polygon.astype(np.int32)
            cv2.fillPoly(overlay, [points], zone.color)

        annotated = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for zone in self.zones.values():
            points = zone.polygon.astype(np.int32)
            cv2.polylines(
                annotated,
                [points],
                isClosed=True,
                color=zone.color,
                thickness=thickness,
            )
            if show_labels:
                label_pos = (int(points[0][0]), int(points[0][1]))
                cv2.putText(
                    annotated,
                    zone.name,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    zone.color,
                    2,
                    cv2.LINE_AA,
                )

        return annotated