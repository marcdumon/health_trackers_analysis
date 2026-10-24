"""The Google Health collector and reader: day ranges, when a day is fetched again, and the points as rows."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from health_tracker_validation import google_health

TZ = ZoneInfo('Europe/Brussels')
HR_POINT = {'dataSource': {'recordingMethod': 'PASSIVELY_MEASURED', 'device': {'displayName': 'Fitbit Air'}, 'platform': 'FITBIT'},
            'heartRate': {'sampleTime': {'physicalTime': '2026-10-23T13:32:55Z', 'utcOffset': '7200s'}, 'beatsPerMinute': '71'}}
STEPS_POINT = {'dataSource': {'recordingMethod': 'PASSIVELY_MEASURED', 'device': {'displayName': 'MobileTrack'}},
               'steps': {'interval': {'startTime': '2026-10-23T13:32:00Z', 'endTime': '2026-10-23T13:33:00Z'}, 'count': '16'}}
ZONES_POINT = {'dataSource': {'recordingMethod': 'DERIVED', 'platform': 'FITBIT'},
               'dailyHeartRateZones': {'date': {'year': 2026, 'month': 10, 'day': 23},
                                       'heartRateZones': [{'heartRateZoneType': 'LIGHT', 'minBeatsPerMinute': '30'},
                                                          {'heartRateZoneType': 'PEAK', 'minBeatsPerMinute': '166'}]}}
LEVEL_POINT = {'dataSource': {'recordingMethod': 'DERIVED', 'device': {'displayName': 'Fitbit Air'}, 'platform': 'FITBIT'},
               'activityLevel': {'interval': {'startTime': '2026-10-23T13:32:00Z', 'endTime': '2026-10-23T13:33:00Z'},
                                 'activityLevelType': 'LIGHTLY_ACTIVE'}}
ROLLUP_POINT = {'startTime': '2026-10-23T12:35:00Z', 'endTime': '2026-10-23T12:36:00Z', 'totalCalories': {'kcalSum': 1.5}}


def test_a_day_runs_from_local_midnight_to_local_midnight() -> None:
    start, end = google_health.day_range(date(2026, 10, 23), TZ)
    assert (start, end) == (datetime(2026, 10, 22, 22, tzinfo=UTC), datetime(2026, 10, 23, 22, tzinfo=UTC))


def test_the_day_the_clock_goes_back_has_25_hours() -> None:
    start, end = google_health.day_range(date(2026, 10, 25), TZ)
    assert (end - start).total_seconds() == 25 * 3600


def test_a_day_is_fetched_until_two_days_after_it_ended(tmp_path: Path) -> None:
    path = tmp_path / '2026-10-23.json'
    day = date(2026, 10, 23)
    assert google_health.needs_fetch(path, day, TZ)
    path.write_text(json.dumps({'fetched_at': '2026-10-24T08:00:00+00:00'}))
    assert google_health.needs_fetch(path, day, TZ)
    path.write_text(json.dumps({'fetched_at': '2026-10-25T23:00:00+00:00'}))
    assert not google_health.needs_fetch(path, day, TZ)


def test_points_become_rows_with_their_time_value_and_device() -> None:
    rows = google_health.rows('heart-rate', [HR_POINT])
    assert rows == [{'type': 'heart-rate', 'start': '2026-10-23T13:32:55Z', 'end': '2026-10-23T13:32:55Z', 'date': None,
                     'field': 'beatsPerMinute', 'value': 71.0, 'text': None, 'device': 'Fitbit Air', 'method': 'PASSIVELY_MEASURED'}]


def test_interval_daily_and_rollup_points_keep_their_own_times() -> None:
    steps = google_health.rows('steps', [STEPS_POINT])[0]
    assert (steps['start'], steps['end'], steps['value'], steps['device']) == ('2026-10-23T13:32:00Z', '2026-10-23T13:33:00Z', 16.0,
                                                                              'MobileTrack')
    zones = google_health.rows('daily-heart-rate-zones', [ZONES_POINT])
    assert [(z['date'], z['field'], z['value']) for z in zones] == [('2026-10-23', 'heartRateZones[LIGHT].minBeatsPerMinute', 30.0),
                                                                   ('2026-10-23', 'heartRateZones[PEAK].minBeatsPerMinute', 166.0)]
    level = google_health.rows('activity-level', [LEVEL_POINT])[0]
    assert (level['field'], level['value'], level['text']) == ('activityLevelType', None, 'LIGHTLY_ACTIVE')
    total = google_health.rows('total-calories', [ROLLUP_POINT])[0]
    assert (total['start'], total['field'], total['value'], total['device']) == ('2026-10-23T12:35:00Z', 'kcalSum', 1.5, None)


def test_the_parts_of_a_point_keep_their_own_times() -> None:
    sleep = {'dataSource': {'recordingMethod': 'DERIVED', 'device': {'displayName': 'Fitbit Air'}},
             'sleep': {'interval': {'startTime': '2026-10-23T23:01:00Z', 'endTime': '2026-10-24T08:31:00Z'},
                       'stages': [{'startTime': '2026-10-23T23:01:00Z', 'endTime': '2026-10-23T23:05:00Z', 'type': 'AWAKE'}]}}
    rows = google_health.rows('sleep', [sleep])
    assert {(r['field'], r['text']) for r in rows} == {('stages[0].startTime', '2026-10-23T23:01:00Z'),
                                                      ('stages[0].endTime', '2026-10-23T23:05:00Z'), ('stages[0].type', 'AWAKE')}
    assert {(r['start'], r['end']) for r in rows} == {('2026-10-23T23:01:00Z', '2026-10-24T08:31:00Z')}


def test_the_raw_files_become_one_table_without_duplicates(tmp_path: Path) -> None:
    for day in ('2026-10-23', '2026-10-24'):
        (tmp_path / 'heart-rate').mkdir(exist_ok=True)
        (tmp_path / 'heart-rate' / f'{day}.json').write_text(json.dumps({'pages': [{'dataPoints': [HR_POINT]}]}))
    (tmp_path / 'total-calories').mkdir()
    (tmp_path / 'total-calories' / '2026-10-23.json').write_text(json.dumps({'pages': [{'rollupDataPoints': [ROLLUP_POINT]}]}))
    table = google_health.table(tmp_path)
    assert sorted(table['type']) == ['heart-rate', 'total-calories']
    assert table['start'].dtype == 'datetime64[ms, UTC]'
    assert table.loc[table['type'] == 'heart-rate', 'start'].iloc[0] == pd.Timestamp('2026-10-23T13:32:55Z')
