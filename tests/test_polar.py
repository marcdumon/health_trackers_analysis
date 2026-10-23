"""The Polar H10 reader: phone clock times around the autumn clock change, beat times from Bluetooth messages, beat
cleaning, and the sessions in the copied folders."""

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from health_tracker_validation import polar

TZ = ZoneInfo('Europe/Brussels')


def utc_ms(text: str) -> int:
    """UTC ms of an ISO time with its offset."""
    return int(pd.Timestamp(text).value // 1_000_000)


def test_phone_times_use_the_summer_offset_before_the_change() -> None:
    got = polar.phone_utc_ms(np.array(['2026-10-24T22:00:00.000', '2026-10-25T01:30:00.250']), TZ)
    assert list(got) == [utc_ms('2026-10-24T20:00:00Z'), utc_ms('2026-10-24T23:30:00.250Z')]


def test_phone_times_after_the_clock_goes_back_use_the_winter_offset() -> None:
    stamps = np.array(['2026-10-25T02:59:59.000', '2026-10-25T02:00:00.500', '2026-10-25T02:30:00.000', '2026-10-25T03:10:00.000'])
    got = polar.phone_utc_ms(stamps, TZ)
    assert list(got) == [utc_ms('2026-10-25T00:59:59Z'), utc_ms('2026-10-25T01:00:00.500Z'), utc_ms('2026-10-25T01:30:00Z'),
                         utc_ms('2026-10-25T02:10:00Z')]
    assert np.all(np.diff(got) > 0)


def test_phone_times_on_a_normal_day_are_unchanged_in_order() -> None:
    got = polar.phone_utc_ms(np.array(['2026-10-23T15:33:10.957', '2026-10-23T15:33:11.732']), TZ)
    assert list(got) == [utc_ms('2026-10-23T13:33:10.957Z'), utc_ms('2026-10-23T13:33:11.732Z')]


def synthetic_messages(seed: int = 1, drop: int | None = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beats with known times, sent in messages about once a second with a delay of 50 to 200 ms; message `drop` is lost
    with its intervals. Returns the true beat times, and the message time and interval of each beat that arrived."""
    rng = np.random.default_rng(seed)
    rr = rng.normal(800, 40, 400).round()
    beats = 1_000_000 + np.cumsum(rr)
    message_of = np.floor((beats - beats[0]) / 1000).astype(int)
    delay = {m: 50 + rng.uniform(0, 150) for m in np.unique(message_of)}
    sent = np.array([beats[message_of == m].max() + delay[m] for m in message_of])
    keep = message_of != drop if drop is not None else np.ones(len(rr), dtype=bool)
    return beats[keep], sent[keep], rr[keep]


def test_beat_times_follow_the_intervals_between_messages() -> None:
    beats, sent, rr = synthetic_messages(drop=None)
    got = polar.beat_times(sent, rr)
    assert np.abs(got - beats).max() < 250


def test_beat_times_start_again_after_a_lost_message() -> None:
    beats, sent, rr = synthetic_messages(drop=40)
    got = polar.beat_times(sent, rr)
    assert np.abs(got - beats).max() < 250


def test_clean_beats_flags_a_missed_and_an_extra_beat() -> None:
    rr = np.full(30, 800.0)
    rr[10] = 1600  # a missed beat: two intervals as one
    rr[20] = 400  # an extra beat: one interval cut in two
    rr[25] = 250  # too short for a heart
    clean = polar.clean_beats(rr)
    assert list(np.flatnonzero(~clean)) == [10, 20, 25]


def write(path: Path, text: str) -> Path:
    """Write a small file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_phone_files_of_one_recording_form_one_session(tmp_path: Path) -> None:
    for name in ('Polar_H10_E813952E_20261023_144412_HR.txt', 'Polar_H10_E813952E_20261023_144412_RR.txt',
                 'Polar_H10_E813952E_20261023_144414_ACC.txt', 'Polar_H10_E813952E_20261023_153237_HR.txt',
                 'Polar_H10_E813952E_20261023_153239_ECG.txt', 'notes.txt'):
        write(tmp_path / name, 'x\n')
    found = polar.phone_sessions(tmp_path)
    assert [f.key for f in found] == ['phone_20261023_144412', 'phone_20261023_153237']
    assert sorted(found[0].files) == ['acc', 'hr', 'rr']
    assert sorted(found[1].files) == ['ecg', 'hr']


def test_a_pc_session_counts_only_when_it_ended(tmp_path: Path) -> None:
    write(tmp_path / '2026-10-23_17-25-49' / 'session.json', json.dumps({'start': '2026-10-23T17:25:49+02:00'}))
    write(tmp_path / '2026-10-23_17-25-49' / 'hr.csv', 'host_ns,hr_bpm,contact\n')
    write(tmp_path / '2026-10-21_20-12-49' / 'session.json', json.dumps({'end': '2026-10-21T21:27:02+02:00'}))
    write(tmp_path / '2026-10-21_20-12-49' / 'hr.csv', 'host_ns,hr_bpm,contact\n')
    assert [f.key for f in polar.pc_sessions(tmp_path)] == ['pc_2026-10-21_20-12-49']


def test_a_phone_session_reads_beats_and_strap_hr(tmp_path: Path) -> None:
    write(tmp_path / 'Polar_H10_E813952E_20261023_153310_HR.txt',
          'Phone timestamp;HR [bpm];HRV [ms];Breathing interval [rpm];\n'
          '2026-10-23T15:33:10.957;76;19,8\n2026-10-23T15:33:11.973;75\n2026-10-23T15:33:12.9\n')
    write(tmp_path / 'Polar_H10_E813952E_20261023_153310_RR.txt',
          'Phone timestamp;RR-interval [ms]\n2026-10-23T15:33:10.957;775\n2026-10-23T15:33:11.973;796\n'
          '2026-10-23T15:33:11.973;790\n2026-10-23T15:33:12.9;broken\n')
    session, beats, strap = polar.read_session(polar.phone_sessions(tmp_path)[0], TZ)
    assert session['key'] == 'phone_20261023_153310'
    assert session['source'] == 'phone'
    assert list(strap['bpm']) == [76, 75]
    assert list(strap['t_ms']) == [utc_ms('2026-10-23T13:33:10.957Z'), utc_ms('2026-10-23T13:33:11.973Z')]
    assert list(beats['rr_ms']) == [775, 796, 790]
    assert beats['t_ms'].is_monotonic_increasing


def test_a_pc_session_reads_rr_in_ms(tmp_path: Path) -> None:
    folder = tmp_path / '2026-10-21_20-12-49'
    write(folder / 'session.json', json.dumps({'end': '2026-10-21T21:27:02+02:00', 'device': {'serial': 'E813952E'}}))
    write(folder / 'hr.csv', 'host_ns,hr_bpm,contact\n1792606371000000000,80,\n1792606372000000000,81,\n')
    write(folder / 'rr.csv', 'host_ns,rr_raw,rr_ms\n1792606371000000000,768,750.0\n1792606372000000000,778,759.765625\n')
    session, beats, strap = polar.read_session(polar.pc_sessions(tmp_path)[0], TZ)
    assert session['key'] == 'pc_2026-10-21_20-12-49'
    assert list(beats['rr_ms']) == [750.0, 759.765625]
    assert list(strap['t_ms']) == [1792606371000, 1792606372000]
