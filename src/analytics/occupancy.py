"""Occupancy and analytics summary tracking for Smart Zone Analytics.

Consumes a PeopleCounter (and the live set of tracked objects) to
derive current occupancy, historical maximum occupancy, and overall
entry/exit totals, per zone and system-wide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.analytics.people_counter import PeopleCounter
from src.tracking.tracker import TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class ZoneSummary:
    """Analytics summary for a single zone.

    Attributes:
        name: Zone name.
        current_occupancy: Number of people currently inside the zone.
        max_occupancy: Highest occupancy recorded for the zone so far.
        total_entries: Total number of entry events recorded.
        total_exits: Total number of exit events recorded.
    """

    name: str
    current_occupancy: int
    max_occupancy: int
    total_entries: int
    total_exits: int


@dataclass
class AnalyticsSummary:
    """System-wide analytics summary across all zones.

    Attributes:
        total_current_occupancy: Sum of current occupancy across zones.
        total_max_occupancy: Sum of each zone's max occupancy.
        total_entries: Sum of entries across all zones.
        total_exits: Sum of exits across all zones.
        zones: Per-zone breakdown, keyed by zone name.
    """

    total_current_occupancy: int
    total_max_occupancy: int
    total_entries: int
    total_exits: int
    zones: dict[str, ZoneSummary] = field(default_factory=dict)


class OccupancyTracker:
    """Tracks live and historical occupancy per zone.

    Wraps a PeopleCounter to additionally track the maximum occupancy
    ever observed per zone, and exposes a consolidated analytics
    summary suitable for dashboards or reporting.

    Attributes:
        people_counter: PeopleCounter providing entry/exit counts and
            current zone membership.
    """

    def __init__(self, people_counter: PeopleCounter) -> None:
        """Initializes the OccupancyTracker.

        Args:
            people_counter: PeopleCounter instance already configured
                with a ZoneManager.
        """
        self.people_counter = people_counter
        self._max_occupancy: dict[str, int] = {
            name: 0 for name in people_counter.zone_manager.get_zone_names()
        }

        logger.info(
            "Initialized OccupancyTracker for zones: %s",
            list(self._max_occupancy.keys()),
        )

    def update(self, tracked_objects: list[TrackedObject]) -> None:
        """Updates occupancy statistics for the current frame.

        Must be called once per frame, after `PeopleCounter.update()`
        has already processed the same `tracked_objects` list, so that
        current zone membership reflects this frame's state.

        Args:
            tracked_objects: Currently active TrackedObject instances
                for this frame. Present for API symmetry with
                PeopleCounter.update(); occupancy is derived from the
                PeopleCounter's internal membership state.
        """
        del tracked_objects  # Membership already reflected in people_counter.

        for zone_name in self._max_occupancy:
            occupants = self.people_counter.get_current_occupants(zone_name)
            current = len(occupants)
            if current > self._max_occupancy[zone_name]:
                self._max_occupancy[zone_name] = current
                logger.debug(
                    "New max occupancy for zone '%s': %d", zone_name, current
                )

    def get_current_occupancy(self, zone_name: str) -> int:
        """Returns the current occupancy of a zone.

        Args:
            zone_name: Name of the zone to query.

        Returns:
            Number of people currently inside the zone.
        """
        return len(self.people_counter.get_current_occupants(zone_name))

    def get_max_occupancy(self, zone_name: str) -> int:
        """Returns the historical maximum occupancy of a zone.

        Args:
            zone_name: Name of the zone to query.

        Returns:
            Highest occupancy ever recorded for the zone.

        Raises:
            KeyError: If no such zone is tracked.
        """
        if zone_name not in self._max_occupancy:
            raise KeyError(f"No max occupancy tracked for zone '{zone_name}'.")
        return self._max_occupancy[zone_name]

    def get_summary(self) -> AnalyticsSummary:
        """Builds a full analytics summary across all zones.

        Returns:
            An AnalyticsSummary with system-wide and per-zone totals
            for current occupancy, max occupancy, entries, and exits.
        """
        zone_summaries: dict[str, ZoneSummary] = {}
        total_current = 0
        total_max = 0
        total_entries = 0
        total_exits = 0

        for zone_name, counts in self.people_counter.get_all_counts().items():
            current = self.get_current_occupancy(zone_name)
            max_occ = self._max_occupancy.get(zone_name, 0)

            zone_summaries[zone_name] = ZoneSummary(
                name=zone_name,
                current_occupancy=current,
                max_occupancy=max_occ,
                total_entries=counts.entries,
                total_exits=counts.exits,
            )

            total_current += current
            total_max += max_occ
            total_entries += counts.entries
            total_exits += counts.exits

        return AnalyticsSummary(
            total_current_occupancy=total_current,
            total_max_occupancy=total_max,
            total_entries=total_entries,
            total_exits=total_exits,
            zones=zone_summaries,
        )

    def reset(self) -> None:
        """Resets max occupancy tracking to zero for all zones."""
        logger.info("Resetting OccupancyTracker state.")
        self._max_occupancy = {
            name: 0
            for name in self.people_counter.zone_manager.get_zone_names()
        }