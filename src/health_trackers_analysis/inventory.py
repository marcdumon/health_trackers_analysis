"""What data each source has, and when."""

import pandas as pd


def spans(times: pd.Series, max_gap_s: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The stretches of time with data: samples less than `max_gap_s` apart form one stretch."""
    if times.empty:
        return []
    ordered = pd.Series(pd.DatetimeIndex(times).sort_values())
    starts = ordered.diff().dt.total_seconds().gt(max_gap_s).cumsum()
    return [(group.iloc[0], group.iloc[-1]) for _, group in ordered.groupby(starts)]
