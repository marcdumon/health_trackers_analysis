"""The collector: it copies only the study period, only finished PC sessions, never twice, and reads the activities of the
health dashboard without changing that database."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from health_trackers_analysis import collect

TZ = ZoneInfo('Europe/Brussels')
START = datetime(2026, 10, 23, 13, 26, tzinfo=TZ)


def write(path: Path, text: str = 'x\n') -> Path:
    """Write a small file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_phone_files_of_the_study_period_are_copied_once(tmp_path: Path) -> None:
    phone, pc, target = tmp_path / 'phone', tmp_path / 'pc', tmp_path / 'raw'
    write(phone / 'Polar_H10_E813952E_20261022_093636_HR.txt')
    write(phone / 'Polar_H10_E813952E_20261023_144412_HR.txt')
    write(phone / 'MARKER_20261023_150000.txt')
    write(phone / 'notes.txt')
    pc.mkdir()
    assert collect.polar_files(phone, pc, START, TZ, target) == 2
    copied = sorted(p.name for p in (target / 'phone').iterdir())
    assert copied == ['MARKER_20261023_150000.txt', 'Polar_H10_E813952E_20261023_144412_HR.txt']
    assert collect.polar_files(phone, pc, START, TZ, target) == 0


def test_a_changed_phone_file_is_copied_again(tmp_path: Path) -> None:
    phone, pc, target = tmp_path / 'phone', tmp_path / 'pc', tmp_path / 'raw'
    source = write(phone / 'Polar_H10_E813952E_20261023_144412_HR.txt')
    pc.mkdir()
    collect.polar_files(phone, pc, START, TZ, target)
    source.write_text('x\ny\n')
    assert collect.polar_files(phone, pc, START, TZ, target) == 1
    assert (target / 'phone' / source.name).read_text() == 'x\ny\n'


def test_only_finished_pc_sessions_of_the_study_period_are_copied(tmp_path: Path) -> None:
    phone, pc, target = tmp_path / 'phone', tmp_path / 'pc', tmp_path / 'raw'
    phone.mkdir()
    write(pc / '2026-10-21_20-12-49' / 'session.json', json.dumps({'end': '2026-10-21T21:27:02+02:00'}))
    write(pc / '2026-10-23_16-00-00' / 'session.json', json.dumps({'end': '2026-10-23T17:00:00+02:00'}))
    write(pc / '2026-10-23_16-00-00' / 'hr.csv')
    write(pc / '2026-10-23_17-25-49' / 'session.json', json.dumps({'start': '2026-10-23T17:25:49+02:00'}))
    assert collect.polar_files(phone, pc, START, TZ, target) == 2
    assert sorted(p.name for p in (target / 'pc').iterdir()) == ['2026-10-23_16-00-00']


def test_the_activities_are_read_from_the_health_dashboard(tmp_path: Path) -> None:
    db = tmp_path / 'health.db'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE polar_activities (id INTEGER PRIMARY KEY, name TEXT, position INTEGER, test TEXT)')
        con.execute('CREATE TABLE polar_sessions (id INTEGER PRIMARY KEY, source TEXT, serial TEXT, path TEXT, start_at TEXT, '
                    'end_at TEXT, removed INTEGER, copy_of INTEGER, note TEXT, activity_id INTEGER)')
        con.execute("INSERT INTO polar_activities VALUES (1, 'Rest', 1, NULL)")
        con.execute("INSERT INTO polar_sessions VALUES (130, 'phone', 'E813952E', '/x/Polar_H10_E813952E_20261023_144412_HR.txt', "
                    "'2026-10-23T12:44:13+00:00', '2026-10-23T13:32:18+00:00', 0, NULL, NULL, 1)")
        con.execute("INSERT INTO polar_sessions VALUES (129, 'phone', 'E813952E', '/x/old_HR.txt', '2026-10-22T09:36:36+00:00', "
                    "'2026-10-22T10:40:34+00:00', 0, NULL, NULL, NULL)")
    before = os.stat(db).st_mtime_ns
    target = tmp_path / 'raw' / 'sessions.csv'
    assert collect.dashboard_sessions(db, START, target) == 1
    rows = pd.read_csv(target)
    assert list(rows['name']) == ['Polar_H10_E813952E_20261023_144412_HR.txt']
    assert list(rows['activity']) == ['Rest']
    assert os.stat(db).st_mtime_ns == before


def test_an_export_is_kept_only_when_it_is_from_the_study_period(tmp_path: Path) -> None:
    old = write(tmp_path / 'old.zip')
    os.utime(old, (START.timestamp() - 3600, START.timestamp() - 3600))
    new = write(tmp_path / 'new.zip')
    assert collect.export(old, START, tmp_path / 'kept', 'Old') is None
    assert collect.export(new, START, tmp_path / 'kept', 'New') is not None
    assert collect.export(new, START, tmp_path / 'kept', 'New') is None
    assert collect.export(tmp_path / 'missing.zip', START, tmp_path / 'kept', 'Missing') is None
