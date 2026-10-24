"""The tables made by `hta build`, for the notebooks."""

import pandas as pd

from health_trackers_analysis.config import PROCESSED


def load(name: str) -> pd.DataFrame:
    """One table of data/processed/ (h10_sessions, h10_beats, h10_strap_hr, google_health, fitbit_hr)."""
    return pd.read_parquet(PROCESSED / f'{name}.parquet')
