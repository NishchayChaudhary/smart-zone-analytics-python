"""People counting logic for Smart Zone Analytics.

Tracks each person's zone membership over time and derives entry and
exit events per zone, based on tracked object centroids and a
ZoneManager. Designed to consume TrackedObject instances produced by
src.tracking.tracker.Tracker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.analytics.zone_manager import ZoneManager
from src.tracking.tracker import TrackedObject

logger = logging.getLogger(__name__)


def _centroid(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Computes the (x, y) centroid of a bounding box.

    Args:
        xyxy: Bounding box as (x1, y1, x2, y2).

    Returns:
        The (x, y) center point of the box.
    """
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class ZoneCounts:
    """Entry/exit counters for a single zone.

    Attributes:
        entries: Total number of entry events recorded for this zone.
        exits: Total number of exit events recorded for this zone.
    """

    entries: int = 0
    exits: int = 0


class PeopleCounter:
    """Counts unique people entering and exiting each configured zone.

    Maintains, per tracked object ID, the set of zones it was inside
    on the previous update. Comparing this against the current frame's
    membership yields entry events (newly inside a zone) and exit
    events (previously inside, now outside). Because membership is
    tracked per unique track_id, a person cannot be double-counted
    while continuously present inside a zone — only genuine
    enter/leave transitions increment the counters.

    Attributes:
        zone_manager: ZoneManager providing zone polygons and
            membership tests.
    """

    def __init__(self, zone_manager: ZoneManager) -> None:
        """Initializes the PeopleCounter.

        Args:
            zone_manager: ZoneManager instance defining the zones to
                count entries/exits for.
        """
        self.zone_manager = zone_manager
        self._zone_counts: dict[str, ZoneCounts] = {
            name: ZoneCounts() for name in zone_manager.get_zone_names()
        }
        self._track_zone_membership: dict[int, set[str]] = {}

        logger.info(
            "Initialized PeopleCounter for zones: %s",
            zone_manager.get_zone_names(),
        )

    def update(self, tracked_objects: list[TrackedObject]) -> None:
        """Updates entry/exit counts based on the current frame's tracks.

        For each tracked object, determines which zones its centroid
        currently falls inside, compares against its zone membership
        on the previous call, and records entry/exit events for any
        zones it has newly entered or left. Tracks that disappear
        entirely (not present in `tracked_objects`) are treated as
        having exited every zone they were previously inside.

        Args:
            tracked_objects: Currently active TrackedObject instances
                for this frame, as produced by Tracker.update().
        """
        current_ids = {obj.track_id for obj in tracked_objects}

        for obj in tracked_objects:
            centroid = _centroid(obj.xyxy)
            current_zones = set(
                self.zone_manager.get_zones_containing_point(centroid)
            )
            previous_zones = self._track_zone_membership.get(
                obj.track_id, set()
            )

            entered = current_zones - previous_zones
            exited = previous_zones - current_zones

            for zone_name in entered:
                self._zone_counts[zone_name].entries += 1
                logger.debug(
                    "Track %d entered zone '%s'.", obj.track_id, zone_name
                )

            for zone_name in exited:
                self._zone_counts[zone_name].exits += 1
                logger.debug(
                    "Track %d exited zone '%s'.", obj.track_id, zone_name
                )

            self._track_zone_membership[obj.track_id] = current_zones

        # Tracks that vanished entirely this frame (e.g. left the
        # camera view) are treated as exiting any zone they were
        # still inside, so counts stay consistent.
        vanished_ids = set(self._track_zone_membership.keys()) - current_ids
        for track_id in vanished_ids:
            previous_zones = self._track_zone_membership.pop(track_id, set())
            for zone_name in previous_zones:
                self._zone_counts[zone_name].exits += 1
                logger.debug(
                    "Track %d vanished while inside zone '%s'; "
                    "recorded exit.",
                    track_id,
                    zone_name,
                )

    def get_counts(self, zone_name: str) -> ZoneCounts:
        """Returns entry/exit counts for a specific zone.

        Args:
            zone_name: Name of the zone to query.

        Returns:
            A ZoneCounts instance with the zone's current entries and
            exits.

        Raises:
            KeyError: If no such zone is registered.
        """
        if zone_name not in self._zone_counts:
            raise KeyError(f"No counts available for zone '{zone_name}'.")
        return self._zone_counts[zone_name]

    def get_all_counts(self) -> dict[str, ZoneCounts]:
        """Returns entry/exit counts for every registered zone.

        Returns:
            Dictionary mapping zone name to its ZoneCounts.
        """
        return dict(self._zone_counts)

    def get_current_occupants(self, zone_name: str) -> set[int]:
        """Returns the set of track IDs currently inside a zone.

        Args:
            zone_name: Name of the zone to query.

        Returns:
            Set of track IDs currently located inside the zone.
        """
        return {
            track_id
            for track_id, zones in self._track_zone_membership.items()
            if zone_name in zones
        }

    def reset(self) -> None:
        """Resets all counts and zone membership state."""
        logger.info("Resetting PeopleCounter state.")
        self._zone_counts = {
            name: ZoneCounts() for name in self.zone_manager.get_zone_names()
        }
        self._track_zone_membership.clear()