"""Dashboard package for Smart Zone Analytics.

Exposes the reusable formatting/UI helpers used by the Streamlit
dashboard (src/dashboard/app.py). The Streamlit app itself is a
script (run via `streamlit run src/dashboard/app.py`), not a class
exported here.
"""

from src.dashboard.metrics import summary_to_kpis, summary_to_zone_rows

__all__ = ["summary_to_kpis", "summary_to_zone_rows"]
