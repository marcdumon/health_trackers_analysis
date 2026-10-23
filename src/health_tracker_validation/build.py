"""Turns the raw copies in data/raw/ into tables in data/processed/ (parquet, times in UTC):

- h10_sessions: one row per H10 session, with the activity the user gave it in the Automation desk
- h10_beats: every RR interval with the time of its beat and whether it is a clean beat
- h10_strap_hr: the HR the strap itself reports, about once a second
- google_health: every Fitbit value of every type, one row per value
- fitbit_hr: the Fitbit Air heart rate samples"""

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from health_tracker_validation import google_health, polar
from health_tracker_validation.config import PROCESSED, RAW, settings


def _utc(ms: pd.Series) -> pd.Series:
    """UTC ms as UTC times."""
    return pd.to_datetime(ms.round().astype('int64'), unit='ms', utc=True)


def polar_tables(folder: Path, tz: ZoneInfo, activities: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The sessions, beats and strap HR of the copied H10 files."""
    found = [*polar.phone_sessions(folder / 'phone')] if (folder / 'phone').is_dir() else []
    found += polar.pc_sessions(folder / 'pc') if (folder / 'pc').is_dir() else []
    sessions, beats, strap = [], [], []
    for session in found:
        row, session_beats, session_strap = polar.read_session(session, tz)
        if row['start_ms'] is None:
            continue
        sessions.append(row)
        beats.append(session_beats)
        strap.append(session_strap)
    sessions_frame = pd.DataFrame(sessions)
    if activities.is_file() and len(sessions_frame):
        known = pd.read_csv(activities)[['name', 'app_id', 'activity', 'test', 'removed', 'copy_of', 'note']]
        sessions_frame = sessions_frame.merge(known, on='name', how='left')
    if len(sessions_frame):
        sessions_frame['start'] = _utc(sessions_frame.pop('start_ms'))
        sessions_frame['end'] = _utc(sessions_frame.pop('end_ms'))
    beats_frame = pd.concat(beats, ignore_index=True) if beats else pd.DataFrame(columns=['session', 't_ms', 'rr_ms', 'clean'])
    strap_frame = pd.concat(strap, ignore_index=True) if strap else pd.DataFrame(columns=['session', 't_ms', 'bpm'])
    beats_frame.insert(1, 'time', _utc(beats_frame.pop('t_ms')))
    strap_frame.insert(1, 'time', _utc(strap_frame.pop('t_ms')))
    return sessions_frame, beats_frame, strap_frame


def fitbit_hr(points: pd.DataFrame) -> pd.DataFrame:
    """The heart rate samples of the Fitbit Air, in time order."""
    hr = points[(points['type'] == 'heart-rate') & (points['field'] == 'beatsPerMinute') & (points['device'] == 'Fitbit Air')]
    return (hr[['start', 'value']].rename(columns={'start': 'time', 'value': 'bpm'}).drop_duplicates('time')
            .sort_values('time', ignore_index=True))


def run(raw: Path = RAW, target: Path = PROCESSED) -> dict[str, int]:
    """Build every table; returns the rows of each."""
    tz = settings().tz
    target.mkdir(parents=True, exist_ok=True)
    sessions, beats, strap = polar_tables(raw / 'polar_h10', tz, raw / 'automation_desk' / 'polar_sessions.csv')
    points = google_health.table(raw / 'google_health')
    tables = {'h10_sessions': sessions, 'h10_beats': beats, 'h10_strap_hr': strap, 'google_health': points,
              'fitbit_hr': fitbit_hr(points)}
    for name, frame in tables.items():
        frame.to_parquet(target / f'{name}.parquet', index=False)
    return {name: len(frame) for name, frame in tables.items()}
