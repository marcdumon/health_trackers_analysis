"""The tables of data/processed/: the Google Health values as one table."""

import json
from pathlib import Path

import pandas as pd

from health_trackers_analysis import build

HR_POINT = {'dataSource': {'recordingMethod': 'PASSIVELY_MEASURED', 'device': {'displayName': 'Fitbit Air'}, 'platform': 'FITBIT'},
            'heartRate': {'sampleTime': {'physicalTime': '2026-10-23T13:32:55Z', 'utcOffset': '7200s'}, 'beatsPerMinute': '71'}}
ROLLUP_POINT = {'startTime': '2026-10-23T12:35:00Z', 'endTime': '2026-10-23T12:36:00Z', 'totalCalories': {'kcalSum': 1.5}}


def test_the_saved_google_health_files_become_one_table_with_utc_times(tmp_path: Path) -> None:
    for day in ('2026-10-23', '2026-10-24'):
        (tmp_path / 'heart-rate').mkdir(exist_ok=True)
        (tmp_path / 'heart-rate' / f'{day}.json').write_text(json.dumps({'pages': [{'dataPoints': [HR_POINT]}]}))
    (tmp_path / 'total-calories').mkdir()
    (tmp_path / 'total-calories' / '2026-10-23.json').write_text(json.dumps({'pages': [{'rollupDataPoints': [ROLLUP_POINT]}]}))
    table = build.google_health_table(tmp_path)
    assert sorted(table['type']) == ['heart-rate', 'total-calories']
    assert table['start'].dtype == 'datetime64[ms, UTC]'
    assert table.loc[table['type'] == 'heart-rate', 'start'].iloc[0] == pd.Timestamp('2026-10-23T13:32:55Z')


def test_no_saved_google_health_files_give_an_empty_table_with_its_columns(tmp_path: Path) -> None:
    table = build.google_health_table(tmp_path)
    assert list(table.columns) == ['type', 'start', 'end', 'date', 'field', 'value', 'text', 'device', 'method']
    assert table.empty
