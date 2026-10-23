"""The tables made by `htv build`, for the notebooks."""

import pandas as pd

from health_tracker_validation.config import PROCESSED


def load(name: str) -> pd.DataFrame:
    """One table of data/processed/ (h10_sessions, h10_beats, h10_strap_hr, google_health, fitbit_hr)."""
    return pd.read_parquet(PROCESSED / f'{name}.parquet')
