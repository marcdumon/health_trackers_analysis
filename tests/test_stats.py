"""The agreement statistics, each checked against a known answer or against an established library."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from health_tracker_validation import stats


def test_a_constant_offset_is_all_bias_and_no_spread() -> None:
    reference = np.array([60.0, 70, 80, 90, 100])
    got = stats.agreement(reference + 2, reference)
    assert got['n'] == 5
    assert got['bias'] == pytest.approx(2)
    assert got['sd'] == pytest.approx(0)
    assert got['mae'] == pytest.approx(2)
    assert got['rmse'] == pytest.approx(2)
    assert got['within_5'] == pytest.approx(100)
    assert got['r'] == pytest.approx(1)
    assert got['ccc'] < 1


def test_mape_and_the_share_within_10_percent() -> None:
    got = stats.agreement(np.array([110.0, 50]), np.array([100.0, 100]))
    assert got['mape'] == pytest.approx(30)
    assert got['within_10pct'] == pytest.approx(50)
    assert got['max_abs'] == pytest.approx(50)


def test_lins_ccc_equals_pearson_r_times_the_bias_correction() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(80, 10, 500)
    y = 0.9 * x + 10 + rng.normal(0, 4, 500)
    r = np.corrcoef(x, y)[0, 1]
    v = x.std() / y.std()
    u = (x.mean() - y.mean()) / np.sqrt(x.std() * y.std())
    assert stats.lin_ccc(x, y) == pytest.approx(r * 2 / (v + 1 / v + u**2))


def test_repeated_measures_limits_match_a_mixed_model() -> None:
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(12), 40)
    diff = 1.0 + rng.normal(0, 2.0, 12)[clusters] + rng.normal(0, 3.0, len(clusters))
    got = stats.repeated_measures_limits(diff, clusters)
    fit = smf.mixedlm('d ~ 1', pd.DataFrame({'d': diff, 'c': clusters}), groups='c').fit(reml=True)
    sd_total = np.sqrt(float(fit.cov_re.iloc[0, 0]) + fit.scale)
    assert got['clusters'] == 12
    assert got['bias'] == pytest.approx(diff.mean())
    assert got['sd_total'] == pytest.approx(sd_total, rel=0.02)
    assert got['loa_high'] - got['loa_low'] == pytest.approx(2 * stats.Z95 * got['sd_total'])


def test_repeated_measures_limits_are_wider_than_naive_ones_when_sessions_differ() -> None:
    rng = np.random.default_rng(11)
    clusters = np.repeat(np.arange(8), 100)
    diff = rng.normal(0, 3, 8)[clusters] + rng.normal(0, 1, len(clusters))
    assert stats.repeated_measures_limits(diff, clusters)['sd_total'] > diff.std(ddof=1)


def test_the_block_bootstrap_is_repeatable_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({'diff': rng.normal(1, 2, 600), 'block': np.repeat(np.arange(30), 20)})
    first = stats.block_bootstrap(frame, lambda f: f['diff'].mean(), 'block', n=300, seed=1)
    again = stats.block_bootstrap(frame, lambda f: f['diff'].mean(), 'block', n=300, seed=1)
    assert first == again
    assert first['low'] < first['estimate'] < first['high']
    assert first['estimate'] == pytest.approx(frame['diff'].mean())


def test_the_block_bootstrap_is_wider_for_correlated_samples() -> None:
    rng = np.random.default_rng(9)
    blocks = np.repeat(np.arange(40), 25)
    correlated = pd.DataFrame({'diff': rng.normal(0, 2, 40)[blocks] + rng.normal(0, 0.5, 1000), 'block': blocks})
    by_block = stats.block_bootstrap(correlated, lambda f: f['diff'].mean(), 'block', n=400, seed=2)
    by_sample = stats.block_bootstrap(correlated.assign(block=np.arange(1000)), lambda f: f['diff'].mean(), 'block', n=400, seed=2)
    assert by_block['high'] - by_block['low'] > 3 * (by_sample['high'] - by_sample['low'])


def test_proportional_bias_finds_a_slope() -> None:
    rng = np.random.default_rng(13)
    mean = rng.uniform(50, 150, 800)
    frame = pd.DataFrame({'h10': mean, 'diff': 0.1 * (mean - 100) + rng.normal(0, 1, 800), 'block': np.repeat(np.arange(40), 20)})
    got = stats.proportional_bias(frame, 'h10', 'block', n=300)
    assert got['slope'] == pytest.approx(0.1, abs=0.01)
    assert 0 < got['slope_low'] < got['slope'] < got['slope_high']


def test_icc_is_one_for_identical_values_and_lower_with_noise() -> None:
    rng = np.random.default_rng(17)
    reference = rng.normal(80, 10, 200)
    assert stats.icc(reference, reference)['icc_agreement'] == pytest.approx(1)
    assert stats.icc(reference + rng.normal(0, 10, 200), reference)['icc_agreement'] < 0.8


def test_the_bootstrap_table_gives_an_interval_for_each_statistic() -> None:
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({'diff': rng.normal(1, 2, 400), 'block': np.repeat(np.arange(20), 20)})
    table = stats.bootstrap_table(frame, lambda f: {'bias': f['diff'].mean(), 'sd': f['diff'].std()}, 'block', n=200, seed=3)
    assert list(table.index) == ['bias', 'sd']
    assert (table['low'] < table['estimate']).all() and (table['estimate'] < table['high']).all()


def test_bins_give_bias_interval_mae_and_count() -> None:
    frame = pd.DataFrame({'h10': [60.0, 62, 64, 80, 82, 84], 'diff': [1.0, 1, 1, -2, -2, -2], 'block': [1, 1, 2, 3, 3, 4]})
    table = stats.binned(frame, 'h10', [0, 70, 200], 'block', n=50)
    assert list(table['bin']) == ['< 70', '70–200']
    assert list(table['n']) == [3, 3]
    assert list(table['bias']) == pytest.approx([1, -2])
    assert list(table['mae']) == pytest.approx([1, 2])
    assert table['bias_low'].isna().all()


def test_the_decorrelation_time_of_an_ar1_series() -> None:
    rng = np.random.default_rng(29)
    values = np.zeros(20000)
    for i in range(1, len(values)):
        values[i] = 0.8 * values[i - 1] + rng.normal()
    times = pd.Series(pd.date_range('2026-10-23', periods=len(values), freq='3s', tz='UTC'))
    # 0.8 ** k falls below 0.1 at k = 11 samples: 33 s
    assert stats.decorrelation_time(pd.Series(values), pd.Series(['a'] * len(values)), times) == pytest.approx(33, abs=6)
