"""The inventory: when each source has data."""

import pandas as pd

from health_tracker_validation import inventory


def test_samples_close_together_form_one_span() -> None:
    times = pd.Series(pd.to_datetime(['2026-10-23T12:00:00Z', '2026-10-23T12:00:03Z', '2026-10-23T12:00:06Z',
                                      '2026-10-23T12:10:00Z', '2026-10-23T12:10:03Z'], utc=True))
    spans = inventory.spans(times, max_gap_s=60)
    assert spans == [(pd.Timestamp('2026-10-23T12:00:00Z'), pd.Timestamp('2026-10-23T12:00:06Z')),
                     (pd.Timestamp('2026-10-23T12:10:00Z'), pd.Timestamp('2026-10-23T12:10:03Z'))]


def test_no_samples_give_no_spans() -> None:
    assert inventory.spans(pd.Series([], dtype='datetime64[ms, UTC]'), max_gap_s=60) == []
