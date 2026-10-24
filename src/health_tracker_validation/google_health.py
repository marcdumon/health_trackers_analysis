"""The Fitbit Air through the Google Health API: every data type is saved as it comes, one file per type and local day
in data/raw/google_health/<type>/<YYYY-MM-DD>.json, and read back as one long table (a row per value).

A day is fetched again until two days after it ended, because the band syncs late and a night ends on the next day."""

import json
import time as clock
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

from health_tracker_validation.config import RAW, SECRETS

BASE = 'https://health.googleapis.com/v4/users/me'
FAMILY = 'users/me/dataSourceFamilies/google-wearables'
TOKEN_FILE = SECRETS / 'google_health_token.json'
FOLDER = RAW / 'google_health'
SETTLE = timedelta(days=2)
# the API often answers 500 or 503 for a moment: such a request is tried again after 2, 4, 8 and 16 s
RETRY_WAITS_S = (2, 4, 8, 16)
# how each type is asked for (checked on the API on 2026-10-23): by sample time, by interval start, by date, sleep by its
# local end time from all sources (the API refuses the wearables filter), exercise without a filter (no filter works), or
# as a roll-up of a window (types without a list)
TYPES = {
    'heart-rate': 'sample', 'heart-rate-variability': 'sample', 'oxygen-saturation': 'sample',
    'respiratory-rate-sleep-summary': 'sample', 'vo2-max': 'sample', 'run-vo2-max': 'sample', 'weight': 'sample',
    'body-fat': 'sample', 'height': 'sample', 'blood-glucose': 'sample', 'core-body-temperature': 'sample',
    'steps': 'interval', 'distance': 'interval', 'altitude': 'interval', 'active-minutes': 'interval',
    'active-zone-minutes': 'interval', 'activity-level': 'interval', 'sedentary-period': 'interval',
    'active-energy-burned': 'interval', 'time-in-heart-rate-zone': 'interval', 'swim-lengths-data': 'interval',
    'daily-heart-rate-variability': 'date', 'daily-resting-heart-rate': 'date', 'daily-oxygen-saturation': 'date',
    'daily-respiratory-rate': 'date', 'daily-sleep-temperature-derivations': 'date', 'daily-heart-rate-zones': 'date',
    'daily-vo2-max': 'date',
    'sleep': 'sleep',
    'exercise': 'all',
    'total-calories': 'rollup:60s', 'calories-in-heart-rate-zone': 'rollup:60s', 'floors': 'rollup:3600s',
}
TIME_KEYS = {'sampleTime', 'interval', 'date', 'startTime', 'endTime'}


def day_range(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """The UTC start and end of a local calendar day."""
    start = datetime.combine(day, time(), tz)
    end = datetime.combine(day + timedelta(days=1), time(), tz)
    return start.astimezone(UTC), end.astimezone(UTC)


def needs_fetch(path: Path, day: date, tz: ZoneInfo) -> bool:
    """Whether a day is fetched (again): its file is missing, or was fetched less than two days after the day ended."""
    if not path.exists():
        return True
    fetched = datetime.fromisoformat(json.loads(path.read_text())['fetched_at'])
    return fetched < day_range(day, tz)[1] + SETTLE


def http_session() -> AuthorizedSession:
    """The kept Google Health token, renewed when needed; never a browser login."""
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if not creds.valid:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json())
    return AuthorizedSession(creds)


def _zulu(at: datetime) -> str:
    """A UTC time as the API writes it."""
    return at.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _request(kind: str, day: date, tz: ZoneInfo) -> dict:
    """What to ask for one type on one day."""
    mode = TYPES[kind]
    name = kind.replace('-', '_')
    start, end = day_range(day, tz)
    if mode.startswith('rollup:'):
        return {'method': 'rollUp', 'body': {'range': {'startTime': _zulu(start), 'endTime': _zulu(end)},
                                             'windowSize': mode.split(':')[1], 'dataSourceFamily': FAMILY}}
    page_size = 25 if kind in ('sleep', 'exercise') else 10000
    if mode == 'all':
        return {'method': 'list', 'params': {'pageSize': page_size, 'dataSourceFamily': FAMILY}}
    if mode == 'sleep':
        text = (f'{name}.interval.civil_end_time >= "{day}T00:00:00" AND '
                f'{name}.interval.civil_end_time < "{day + timedelta(days=1)}T00:00:00"')
        return {'method': 'list', 'params': {'pageSize': page_size, 'filter': text}}
    field = {'sample': 'sample_time.physical_time', 'interval': 'interval.start_time'}.get(mode)
    text = (f'{name}.date >= "{day}" AND {name}.date < "{day + timedelta(days=1)}"' if mode == 'date' else
            f'{name}.{field} >= "{_zulu(start)}" AND {name}.{field} < "{_zulu(end)}"')
    return {'method': 'list', 'params': {'pageSize': page_size, 'filter': text, 'dataSourceFamily': FAMILY}}


def _send(call: Callable[[], requests.Response]) -> dict:
    """The answer of a request, tried again after a short wait while the API answers 429 or 5xx."""
    for wait in (*RETRY_WAITS_S, None):
        answer = call()
        if wait is None or (answer.status_code != 429 and answer.status_code < 500):
            answer.raise_for_status()
            return answer.json()
        clock.sleep(wait)
    raise AssertionError('unreachable')


def _pages(http: AuthorizedSession, kind: str, request: dict) -> list[dict]:
    """Every page of an answer."""
    url = f'{BASE}/dataTypes/{kind}/dataPoints'
    if request['method'] == 'rollUp':
        return [_send(lambda: http.post(f'{url}:rollUp', json=request['body']))]
    pages, token = [], None
    while True:
        params = request['params'] | ({'pageToken': token} if token else {})
        pages.append(_send(lambda params=params: http.get(url, params=params)))
        if not (token := pages[-1].get('nextPageToken')):
            return pages


def collect(first: date, last: date, tz: ZoneInfo, folder: Path = FOLDER) -> dict[str, int]:
    """Fetch every type for every day from `first` to `last` that needs it; exercise as one list on each run. Returns how
    many files were written and how many requests failed (a failed type is tried again on the next run)."""
    http = http_session()
    counts = {'written': 0, 'failed': 0}
    days = [first + timedelta(days=n) for n in range((last - first).days + 1)]
    for kind, mode in TYPES.items():
        for day in [None] if mode == 'all' else days:
            path = folder / kind / ('all.json' if day is None else f'{day}.json')
            if day is not None and not needs_fetch(path, day, tz):
                continue
            request = _request(kind, day or last, tz)
            try:
                pages = _pages(http, kind, request)
            except Exception as error:  # one failing type must not stop the others
                counts['failed'] += 1
                print(f'google health: {kind} {day or "all"}: {error}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fetched = {'type': kind, 'day': None if day is None else str(day),
                       'fetched_at': datetime.now(UTC).isoformat(timespec='seconds'), 'request': request, 'pages': pages}
            part = path.with_name(f'{path.name}.part')
            part.write_text(json.dumps(fetched))
            part.replace(path)
            counts['written'] += 1
    return counts


def _number(value: object) -> float | None:
    """A numeric field (the API sends 64-bit integers as text); None for anything else."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _leaves(value: object, path: str = '') -> Iterator[tuple[str, object]]:
    """The value fields of a point body, by path; a list item is named by its type field (e.g. LIGHT) or its index. Only the
    body's own times are left out (they are the row's start, end and date); a part such as a sleep stage keeps its times."""
    if isinstance(value, dict):
        for key, item in value.items():
            if path or key not in TIME_KEYS:
                yield from _leaves(item, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for n, item in enumerate(value):
            label = next((v for k, v in item.items() if isinstance(v, str) and (k.endswith('Type') or k.endswith('Zone'))), None) \
                if isinstance(item, dict) else None
            rest = {k: v for k, v in item.items() if v != label} if label is not None else item
            yield from _leaves(rest, f'{path}[{label if label is not None else n}]')
    else:
        yield path, value


def rows(kind: str, points: list[dict]) -> list[dict]:
    """The rows of some points of one type: one per value field, with the point's times, date, device and method."""
    found = []
    for point in points:
        source = point.get('dataSource', {})
        body_key = next((k for k in point if k not in ('dataSource', 'name', 'startTime', 'endTime')), None)
        body = point.get(body_key, {}) if body_key else {}
        start = end = None
        if 'sampleTime' in body:
            start = end = body['sampleTime'].get('physicalTime')
        elif 'interval' in body:
            start, end = body['interval'].get('startTime'), body['interval'].get('endTime')
        elif 'startTime' in point:
            start, end = point['startTime'], point.get('endTime')
        day = body.get('date')
        day_text = f"{day['year']}-{day['month']:02d}-{day['day']:02d}" if day else None
        base = {'type': kind, 'start': start, 'end': end, 'date': day_text}
        meta = {'device': source.get('device', {}).get('displayName'), 'method': source.get('recordingMethod')}
        leaves = list(_leaves(body)) or [('', None)]
        for field, value in leaves:
            number = _number(value)
            text = None if number is not None or value is None else str(value)
            found.append(base | {'field': field, 'value': number, 'text': text} | meta)
    return found


def table(folder: Path = FOLDER) -> pd.DataFrame:
    """All saved points as one long table, without the duplicates that overlapping files give."""
    found = []
    for path in sorted(folder.glob('*/*.json')):
        for page in json.loads(path.read_text())['pages']:
            found += rows(path.parent.name, page.get('dataPoints', []) + page.get('rollupDataPoints', []))
    columns = ['type', 'start', 'end', 'date', 'field', 'value', 'text', 'device', 'method']
    frame = pd.DataFrame(found, columns=columns).drop_duplicates(ignore_index=True)
    for column in ('start', 'end'):
        frame[column] = pd.to_datetime(frame[column], utc=True, format='ISO8601').dt.as_unit('ms')
    return frame
