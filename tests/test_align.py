"""The H10 reference HR from the beats, and the pairs of device samples with it."""

import numpy as np
import pandas as pd
import pytest

from health_tracker_validation import align


def ms_times(ms: np.ndarray) -> pd.Series:
    """UTC times from ms."""
    return pd.Series(pd.to_datetime(ms, unit='ms', utc=True))


def beats_frame(rr: np.ndarray, start_ms: int = 1_000_000, session: str = 's1', clean: np.ndarray | None = None) -> pd.DataFrame:
    """Beats with the given intervals from a start time."""
    return pd.DataFrame({'session': session, 'time': ms_times(start_ms + np.cumsum(rr)), 'rr_ms': rr,
                         'clean': np.ones(len(rr), dtype=bool) if clean is None else clean})


def test_steady_beats_give_their_rate_and_nothing_before_them() -> None:
    beats = beats_frame(np.full(100, 800.0))
    got = align.reference_hr(beats, ms_times(np.array([1_000_000, 1_020_000, 1_050_000])), window_s=5)
    assert np.isnan(got[0])
    assert got[1:] == pytest.approx([75, 75])


def test_the_window_averages_the_beats_as_a_count_over_time() -> None:
    rr = np.r_[np.full(10, 1000.0), np.full(10, 500.0)]
    beats = beats_frame(rr)
    end_ms = 1_000_000 + int(rr.sum())
    got = align.reference_hr(beats, ms_times(np.array([end_ms])), window_s=4)
    # the last 4 s hold 8 beats of 500 ms: 120 bpm; a mean of per-beat rates would be the same here
    assert got[0] == pytest.approx(120)


def test_a_window_with_too_many_unclean_beats_has_no_reference() -> None:
    clean = np.ones(40, dtype=bool)
    clean[30:33] = False
    beats = beats_frame(np.full(40, 800.0), clean=clean)
    at = 1_000_000 + 800 * 33
    assert np.isnan(align.reference_hr(beats, ms_times(np.array([at])), window_s=4)[0])
    assert align.reference_hr(beats, ms_times(np.array([at + 6000])), window_s=4)[0] == pytest.approx(75)


def test_a_lag_looks_back_in_time() -> None:
    rr = np.r_[np.full(50, 1000.0), np.full(50, 500.0)]
    beats = beats_frame(rr)
    switch_ms = 1_000_000 + 50_000
    assert align.reference_hr(beats, ms_times(np.array([switch_ms + 3000])), window_s=2)[0] == pytest.approx(120)
    assert align.reference_hr(beats, ms_times(np.array([switch_ms + 3000])), window_s=2, lag_s=4)[0] == pytest.approx(60)


def test_pairs_keep_only_samples_inside_a_session() -> None:
    beats = beats_frame(np.full(200, 800.0))
    sessions = pd.DataFrame({'key': ['s1'], 'start': ms_times(np.array([1_000_000])), 'end': ms_times(np.array([1_160_000])),
                             'activity': ['Rest']})
    device = pd.DataFrame({'time': ms_times(np.array([900_000, 1_030_000, 1_033_000, 1_500_000])), 'bpm': [70.0, 76, 74, 80]})
    got = align.pairs(device, beats, sessions, window_s=5)
    assert list(got['device']) == [76, 74]
    assert list(got['h10']) == pytest.approx([75, 75])
    assert list(got['diff']) == pytest.approx([1, -1])
    assert list(got['session']) == ['s1', 's1']
    assert list(got['activity']) == ['Rest', 'Rest']


def test_the_lag_curve_finds_a_device_that_runs_late() -> None:
    rng = np.random.default_rng(4)
    rr = np.clip(800 + np.cumsum(rng.normal(0, 15, 1500)), 400, 1400)
    beats = beats_frame(rr)
    sessions = pd.DataFrame({'key': ['s1'], 'start': [beats['time'].iloc[0]], 'end': [beats['time'].iloc[-1]], 'activity': ['Rest']})
    at = beats['time'].iloc[30] + pd.to_timedelta(np.arange(0, 900, 3), unit='s')
    late = align.reference_hr(beats, pd.Series(at), window_s=5, lag_s=6)
    device = pd.DataFrame({'time': at, 'bpm': late}).dropna()
    curve = align.lag_curve(device, beats, sessions, lags=range(0, 13), window_s=5)
    assert curve.loc[curve['mae'].idxmin(), 'lag_s'] == 6


def test_a_sample_inside_two_sessions_is_paired_once() -> None:
    beats = pd.concat([beats_frame(np.full(200, 800.0), session='phone'), beats_frame(np.full(200, 800.0), session='pc')])
    sessions = pd.DataFrame({'key': ['phone', 'pc'], 'start': ms_times(np.array([1_000_000] * 2)),
                             'end': ms_times(np.array([1_160_000] * 2)), 'activity': ['Rest', 'Rest']})
    device = pd.DataFrame({'time': ms_times(np.array([1_030_000, 1_033_000])), 'bpm': [76.0, 74]})
    got = align.pairs(device, beats, sessions, window_s=5)
    assert list(got['session']) == ['phone', 'phone']
