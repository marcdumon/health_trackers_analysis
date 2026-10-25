"""The collector: copies everything of the study period into data/raw/. It only reads the sources: the Polar H10 folders,
the health dashboard database (opened read-only), the Health Connect export and the Sleep as Android backup. The Fitbit
data comes from the Google Health API (the google_api library)."""

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from google_api import auth, health

from health_trackers_analysis.config import RAW, SECRETS, settings
from health_trackers_analysis.copying import copy_file, copy_version

PHONE_NAME = re.compile(r'^(?:Polar_H10_[0-9A-Z]+_|MARKER_)(?P<start>\d{8}_\d{6})(?:_(?:HR|RR|ECG|ACC))?\.txt$')
PC_NAME = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
GOOGLE_HEALTH_TOKEN = SECRETS / 'google_health_token.json'


def polar_files(phone_dir: Path, pc_dir: Path, start: datetime, tz: ZoneInfo, target: Path = RAW / 'polar_h10') -> int:
    """Copy the phone files and the finished PC sessions that start in the study period; returns the files copied. A PC
    session still recording (no end in its session.json) waits for the next run."""
    local_start = start.astimezone(tz).replace(tzinfo=None)
    copied = 0
    for path in sorted(phone_dir.glob('*.txt')):
        if (m := PHONE_NAME.match(path.name)) and datetime.strptime(m['start'], '%Y%m%d_%H%M%S') >= local_start:
            copied += copy_file(path, target / 'phone' / path.name)
    for folder in sorted(p for p in pc_dir.iterdir() if p.is_dir() and PC_NAME.match(p.name)):
        info = folder / 'session.json'
        if datetime.strptime(folder.name, '%Y-%m-%d_%H-%M-%S') < local_start or not info.is_file():
            continue
        if not json.loads(info.read_text()).get('end'):
            continue
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            copied += copy_file(path, target / 'pc' / folder.name / path.name)
    return copied


def dashboard_sessions(db: Path, start: datetime, target: Path = RAW / 'health_dashboard' / 'polar_sessions.csv') -> int:
    """Save the H10 sessions of the study period as the health dashboard knows them, with the activity the user gave each;
    returns the number of sessions. `name` is the phone HR file or the PC folder, as in polar.Found."""
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = pd.read_sql_query(
            "SELECT s.id AS app_id, s.source, s.serial, s.path, s.start_at, s.end_at, s.removed, s.copy_of, s.note, "
            "a.name AS activity, a.test FROM polar_sessions s LEFT JOIN polar_activities a ON a.id = s.activity_id "
            "WHERE s.end_at >= ? ORDER BY s.start_at", con, params=(start.astimezone(UTC).isoformat(),))
    finally:
        con.close()
    rows.insert(0, 'name', rows['path'].map(lambda p: Path(p).name))
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(target, index=False)
    return len(rows)


def export(source: Path, start: datetime, folder: Path, prefix: str) -> Path | None:
    """Keep a new version of a full export when it was written in the study period; None when there is none."""
    if not source.is_file() or source.stat().st_mtime < start.timestamp():
        return None
    return copy_version(source, folder, prefix)


def run() -> dict[str, object]:
    """Collect everything; returns what each source gave."""
    s = settings()
    now = datetime.now(s.tz)
    return {
        'polar_h10_files': polar_files(s.polar_phone_dir, s.polar_pc_dir, s.start, s.tz),
        'health_dashboard_sessions': dashboard_sessions(s.health_dashboard_db, s.start),
        'health_connect_export': export(s.health_connect_zip, s.start, RAW / 'health_connect', 'HealthConnect'),
        'sleep_as_android_backup': export(s.sleep_as_android_zip, s.start, RAW / 'sleep_as_android', 'SleepAsAndroid'),
        'google_health': health.collect(auth.creds(GOOGLE_HEALTH_TOKEN), RAW / 'google_health', s.start.astimezone(s.tz).date(),
                                        now.date(), s.tz),
    }
