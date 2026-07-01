"""Reusable Streamlit UI components for the Smart Zone Analytics dashboard.

Each function renders one piece of UI and takes plain data structures
(as produced by src.dashboard.metrics) rather than domain objects
directly, keeping components decoupled from the analytics layer.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st


def render_kpi_row(kpis: dict[str, int]) -> None:
    """Renders a row of KPI metric cards.

    Args:
        kpis: Mapping of KPI label -> value, as produced by
            src.dashboard.metrics.summary_to_kpis.
    """
    columns = st.columns(len(kpis)) if kpis else []
    for col, (label, value) in zip(columns, kpis.items()):
        col.metric(label, value)


def render_zone_table(rows: list[dict[str, Any]]) -> None:
    """Renders the per-zone breakdown table.

    Args:
        rows: List of per-zone dict rows, as produced by
            src.dashboard.metrics.summary_to_zone_rows.
    """
    st.subheader("Zone Breakdown")
    if not rows:
        st.info("No zones configured in config.yaml.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_frame(frame: np.ndarray, placeholder: "st.delta_generator.DeltaGenerator") -> None:
    """Renders a single BGR video frame (converted to RGB) into a placeholder.

    Args:
        frame: Annotated BGR frame, as produced by the detection /
            tracking / zone drawing pipeline.
        placeholder: An `st.empty()` placeholder to render into, so
            the video updates in place instead of scrolling the page.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder.image(rgb, channels="RGB", use_container_width=True)
