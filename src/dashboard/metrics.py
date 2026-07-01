"""Metric formatting helpers for the Smart Zone Analytics dashboard.

Pure functions with no Streamlit dependency: they convert the domain
objects produced by src.analytics (AnalyticsSummary, ZoneSummary)
into simple, display-ready structures. Kept separate from
components.py so formatting logic stays independently testable.
"""

from __future__ import annotations

from typing import Any

from src.analytics.occupancy import AnalyticsSummary


def summary_to_kpis(summary: AnalyticsSummary) -> dict[str, int]:
    """Builds the top-level KPI values shown at the top of the dashboard.

    Args:
        summary: System-wide analytics summary from OccupancyTracker.

    Returns:
        Ordered dict of KPI label -> value.
    """
    return {
        "Current Occupancy": summary.total_current_occupancy,
        "Peak Occupancy": summary.total_max_occupancy,
        "Total Entries": summary.total_entries,
        "Total Exits": summary.total_exits,
    }


def summary_to_zone_rows(summary: AnalyticsSummary) -> list[dict[str, Any]]:
    """Builds one row per zone for the per-zone breakdown table.

    Args:
        summary: System-wide analytics summary from OccupancyTracker.

    Returns:
        List of dict rows, one per zone, suitable for st.dataframe().
    """
    rows: list[dict[str, Any]] = []
    for zone_name, zs in summary.zones.items():
        rows.append(
            {
                "Zone": zone_name,
                "Current Occupancy": zs.current_occupancy,
                "Max Occupancy": zs.max_occupancy,
                "Entries": zs.total_entries,
                "Exits": zs.total_exits,
            }
        )
    return rows
