"""The Polar H10 chest strap: the sessions in the copied files of the phone app (Polar Sensor Logger) and of the PC logger,
read as beats (RR intervals with their times) and as the HR the strap itself reports.

The phone app writes local clock times without a zone; the PC logger writes UTC nanoseconds. Both write the time a
Bluetooth message arrived, the same for all beats in it, so the beat times are rebuilt from the intervals."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PHONE_FILE = re.compile(r'^Polar_H10_(?P<serial>[0-9A-Z]+)_(?P<start>\d{8}_\d{6})_(?P<kind>HR|RR|ECG|ACC)\.txt$')
STREAMS = ('hr', 'rr', 'ecg', 'acc')
# the files of one phone recording start a few seconds apart
SAME_SESSION_S = 60
# a phone time that goes back by more than this is the autumn clock change, not a late message
CLOCK_BACK_MS = 30 * 60 * 1000
# the column of the value in each file (column 0 is the time)
COLUMN = {('phone', 'hr'): 1, ('phone', 'rr'): 1, ('pc', 'hr'): 1, ('pc', 'rr'): 2}
# beat cleaning: a heart beat lasts 300 to 2000 ms and is at most 20 % from the median of the 5 beats before and after it
RR_MIN_MS, RR_MAX_MS, RR_AROUND, RR_OFF = 300, 2000, 5, 0.2
# the beat times use the smallest message delay of the next 20 messages (about 20 s)
DELAY_WINDOW = 20


@dataclass
class Found:
    """A session in the copied files: its key, source ('phone' or 'pc'), strap serial, the file of each stream, and the
    name the Automation desk knows it by (the phone HR file or the PC folder)."""

    key: str
    source: str
    serial: str
    name: str
    files: dict[str, Path] = field(default_factory=dict)


def phone_sessions(folder: Path) -> list[Found]:
    """The phone recordings: each HR file with the RR, ECG and acceleration files of the same strap that start within 60 s
    of it (the nearest, each file once)."""
    named = [(m['serial'], datetime.strptime(m['start'], '%Y%m%d_%H%M%S'), m['kind'].lower(), path)
             for path in folder.glob('Polar_H10_*.txt') if (m := PHONE_FILE.match(path.name))]
    used: set[Path] = set()
    found = []
    for serial, start, _, path in sorted(n for n in named if n[2] == 'hr'):
        files = {'hr': path}
        for kind in ('rr', 'ecg', 'acc'):
            near = sorted((abs((s - start).total_seconds()), str(p), p) for se, s, k, p in named
                          if se == serial and k == kind and p not in used and abs((s - start).total_seconds()) <= SAME_SESSION_S)
            if near:
                files[kind] = near[0][2]
                used.add(near[0][2])
        found.append(Found(f'phone_{start:%Y%m%d_%H%M%S}', 'phone', serial, path.name, files))
    return found


def pc_sessions(folder: Path) -> list[Found]:
    """The PC logger sessions that ended: a folder whose session.json has an end, with hr.csv."""
    found = []
    for info_file in sorted(folder.glob('*/session.json')):
        info = json.loads(info_file.read_text())
        files = {kind: info_file.parent / f'{kind}.csv' for kind in STREAMS if (info_file.parent / f'{kind}.csv').is_file()}
        if info.get('end') and 'hr' in files:
            name = info_file.parent.name
            found.append(Found(f'pc_{name}', 'pc', (info.get('device') or {}).get('serial', ''), name, files))
    return found


def phone_utc_ms(stamps: np.ndarray, tz: ZoneInfo) -> np.ndarray:
    """UTC ms of phone clock times (local, without a zone), in file order. In the hour that occurs twice in autumn, the
    times before the clock goes back are the first pass of that hour and the times after it the second pass."""
    local = pd.DatetimeIndex(pd.to_datetime(stamps)).as_unit('ms')
    back = np.flatnonzero(np.diff(local.asi8) < -CLOCK_BACK_MS)
    first_pass = np.ones(len(local), dtype=bool)
    if back.size:
        first_pass[back[0] + 1:] = False
    return local.tz_localize(tz, ambiguous=first_pass, nonexistent='shift_forward').as_unit('ms').asi8


def _rows(path: Path, column: int) -> list[tuple[str, str]]:
    """The time and the value text of each data row that has both; the header is left out."""
    with path.open(encoding='utf-8', errors='replace') as lines:
        sep = ';' if ';' in next(lines, '') else ','
        rows = []
        for line in lines:
            fields = line.rstrip('\r\n').split(sep)
            if len(fields) > column and fields[0] and fields[column]:
                rows.append((fields[0], fields[column]))
    return rows


def read_stream(found: Found, stream: str, tz: ZoneInfo) -> pd.DataFrame:
    """The readable rows of the HR or RR file of a session: the UTC ms of the message and the value. A row whose time or
    value does not parse (a cut-off last line, a damaged line) is left out."""
    kept_times, kept_values = [], []
    for stamp, text in _rows(found.files[stream], COLUMN[(found.source, stream)]):
        try:
            value = float(text.replace(',', '.'))
            at = np.datetime64(stamp, 'ms') if found.source == 'phone' else int(stamp) // 1_000_000
        except ValueError:
            continue
        kept_times.append(stamp if found.source == 'phone' else at)
        kept_values.append(value)
    if not kept_times:
        return pd.DataFrame({'t_ms': pd.Series(dtype='int64'), 'value': pd.Series(dtype=float)})
    times = phone_utc_ms(np.array(kept_times), tz) if found.source == 'phone' else np.array(kept_times, dtype=np.int64)
    return pd.DataFrame({'t_ms': times, 'value': kept_values})


def beat_times(sent_ms: np.ndarray, rr: np.ndarray, window: int = DELAY_WINDOW) -> np.ndarray:
    """The time (UTC ms) of the beat that ends each interval. The strap measures the intervals exactly, so the beats are the
    running sum of the intervals, put on the clock with the smallest delay of the next `window` messages: the last beat of
    a message happened at least that delay before the message arrived. A lost message makes the sum fall behind the clock,
    and the delay then steps up; the minimum over the next messages follows that step at once."""
    total = np.cumsum(rr.astype(float))
    sent = sent_ms.astype(float)
    new_message = np.r_[True, sent[1:] != sent[:-1]]
    last_of_message = np.r_[new_message[1:], True]
    message = np.cumsum(new_message) - 1
    delay = sent[last_of_message] - total[last_of_message]
    smallest_ahead = pd.Series(delay[::-1]).rolling(window, min_periods=1).min().to_numpy()[::-1]
    return total + smallest_ahead[message]


def clean_beats(rr: np.ndarray) -> np.ndarray:
    """Whether each RR interval is a real beat: 300 to 2000 ms, and at most 20 % from the median of the 5 beats before and
    the 5 after it (a missed or an extra beat is far from it)."""
    if len(rr) == 0:
        return np.zeros(0, dtype=bool)
    padded = np.concatenate([np.full(RR_AROUND, np.nan), rr.astype(float), np.full(RR_AROUND, np.nan)])
    around = np.delete(np.lib.stride_tricks.sliding_window_view(padded, 2 * RR_AROUND + 1), RR_AROUND, axis=1)
    median = np.nanmedian(around, axis=1) if len(rr) > 1 else np.full(1, np.nan)
    near = np.isnan(median) | (np.abs(rr - median) <= RR_OFF * median)
    return (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS) & near


def read_session(found: Found, tz: ZoneInfo) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """A session: its summary row, its beats (time, interval, clean) and the HR the strap reports (time, bpm)."""
    strap = read_stream(found, 'hr', tz).rename(columns={'value': 'bpm'}).sort_values('t_ms', kind='stable')
    beats = pd.DataFrame({'t_ms': pd.Series(dtype=float), 'rr_ms': pd.Series(dtype=float), 'clean': pd.Series(dtype=bool)})
    if 'rr' in found.files and len(rr := read_stream(found, 'rr', tz)):
        beats = pd.DataFrame({'t_ms': beat_times(rr['t_ms'].to_numpy(), rr['value'].to_numpy()), 'rr_ms': rr['value'],
                              'clean': clean_beats(rr['value'].to_numpy())})
    for frame in (strap, beats):
        frame.insert(0, 'session', found.key)
    session = {'key': found.key, 'source': found.source, 'serial': found.serial, 'name': found.name,
               'start_ms': int(strap['t_ms'].min()) if len(strap) else None,
               'end_ms': int(strap['t_ms'].max()) if len(strap) else None,
               'beats': len(beats), 'clean_beats': int(beats['clean'].sum()), 'streams': ','.join(sorted(found.files))}
    return session, beats.reset_index(drop=True), strap.reset_index(drop=True)
