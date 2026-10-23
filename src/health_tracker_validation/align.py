"""The reference: the heart rate of the Polar H10 at any moment, from its beats, and the pairs of a device's samples with
that reference.

The H10 times every beat (RR intervals to the millisecond), so its HR over a window is the number of beats divided by
their summed intervals. A wrist device reports a smoothed value every few seconds; the window and a possible delay of the
device are chosen by a sensitivity analysis (lag_curve, window_curve), not assumed."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

# a window counts only when at least 80 % of its beats are clean and its beats cover at least half of it
MIN_CLEAN_SHARE = 0.8
MIN_COVER = 0.5


def _ms(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """UTC times as integer ms."""
    return pd.DatetimeIndex(times).as_unit('ms').asi8


def reference_hr(beats: pd.DataFrame, at: pd.Series | pd.DatetimeIndex, window_s: float = 5, lag_s: float = 0) -> np.ndarray:
    """The H10 HR (bpm) at each time of `at`: 60000 × the number of clean beats in (t − lag − window, t − lag] divided by
    their summed intervals. NaN when fewer than 80 % of the beats in the window are clean, or when its beats cover less than
    half of it (a gap in the recording)."""
    beat_ms = _ms(beats['time'])
    order = np.argsort(beat_ms, kind='stable')
    beat_ms = beat_ms[order]
    rr = beats['rr_ms'].to_numpy(dtype=float)[order]
    clean = beats['clean'].to_numpy(dtype=bool)[order]
    clean_rr = np.r_[0.0, np.cumsum(np.where(clean, rr, 0.0))]
    clean_count = np.r_[0, np.cumsum(clean)]
    all_rr = np.r_[0.0, np.cumsum(rr)]
    end = _ms(at) - round(lag_s * 1000)
    hi = np.searchsorted(beat_ms, end, side='right')
    lo = np.searchsorted(beat_ms, end - round(window_s * 1000), side='right')
    count, total = hi - lo, all_rr[hi] - all_rr[lo]
    good, good_rr = clean_count[hi] - clean_count[lo], clean_rr[hi] - clean_rr[lo]
    with np.errstate(divide='ignore', invalid='ignore'):
        hr = 60000 * good / good_rr
        usable = (good > 0) & (good >= MIN_CLEAN_SHARE * count) & (total >= MIN_COVER * window_s * 1000)
    return np.where(usable, hr, np.nan)


def pairs(device: pd.DataFrame, beats: pd.DataFrame, sessions: pd.DataFrame, window_s: float = 5, lag_s: float = 0) -> pd.DataFrame:
    """Each device sample (columns time, bpm) inside an H10 session, with the reference HR of that moment, the session,
    its activity, the difference (device − H10), the mean of the two and the minute it falls in. A sample inside two
    sessions is paired once, with the first."""
    found = []
    for session in sessions.itertuples():
        inside = device[(device['time'] >= session.start) & (device['time'] <= session.end)]
        if inside.empty:
            continue
        reference = reference_hr(beats[beats['session'] == session.key], inside['time'], window_s, lag_s)
        found.append(pd.DataFrame({'time': inside['time'].to_numpy(), 'session': session.key, 'activity': session.activity,
                                   'device': inside['bpm'].to_numpy(dtype=float), 'h10': reference}))
    columns = ['time', 'session', 'activity', 'device', 'h10']
    paired = pd.concat(found, ignore_index=True).dropna(subset=['h10']) if found else pd.DataFrame(columns=columns)
    # sessions that overlap (e.g. the phone and the PC recording at once) must not pair a sample twice
    paired = paired.drop_duplicates('time', keep='first')
    paired['diff'] = paired['device'] - paired['h10']
    paired['mean'] = (paired['device'] + paired['h10']) / 2
    paired['minute'] = pd.DatetimeIndex(paired['time']).floor('min')
    return paired.reset_index(drop=True)


def _errors(paired: pd.DataFrame) -> dict[str, float]:
    """The size of the errors of some pairs."""
    diff = paired['diff']
    return {'n': len(diff), 'bias': diff.mean(), 'mae': diff.abs().mean(), 'rmse': float(np.sqrt((diff**2).mean()))}


def lag_curve(device: pd.DataFrame, beats: pd.DataFrame, sessions: pd.DataFrame, lags: Iterable[float],
              window_s: float = 5) -> pd.DataFrame:
    """The errors when the device is assumed to run `lag` seconds late (positive) or early (negative)."""
    return pd.DataFrame([{'lag_s': lag} | _errors(pairs(device, beats, sessions, window_s, lag)) for lag in lags])


def window_curve(device: pd.DataFrame, beats: pd.DataFrame, sessions: pd.DataFrame, windows: Iterable[float],
                 lag_s: float = 0) -> pd.DataFrame:
    """The errors for each length of the reference window."""
    return pd.DataFrame([{'window_s': window} | _errors(pairs(device, beats, sessions, window, lag_s)) for window in windows])
