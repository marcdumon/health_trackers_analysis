"""Agreement between a device and a reference (method comparison).

- Accuracy is the bias: the mean of device minus reference. Precision is the spread of those differences: their SD and
  the 95 % limits of agreement (Bland & Altman 1986).
- The samples of one session are not independent. So the limits use the method for repeated measures where the true
  value varies (Bland & Altman 2007), and the confidence intervals come from a block bootstrap: whole blocks of time
  (e.g. minutes or sessions) are drawn, not single samples.
- Lin's concordance correlation (1989) and the intraclass correlation (pingouin) measure agreement on one scale; the
  slope of the differences on the reference shows a proportional bias."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pingouin as pg

Z95 = 1.959963984540054
MIN_BLOCKS = 5


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient: 1 only when every pair lies on the line of identity."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * covariance / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def agreement(device: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Accuracy, precision and error measures of a device against the reference, sample by sample.

    Returns n, bias, sd (of the differences), loa_low and loa_high (bias ± 1.96 sd), mae, mape (%), rmse, within_5
    (% of samples within ±5 bpm), within_10pct (% within ±10 % of the reference), max_abs, r (Pearson) and ccc (Lin)."""
    device, reference = np.asarray(device, dtype=float), np.asarray(reference, dtype=float)
    diff = device - reference
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float('nan')
    bias = float(diff.mean())
    relative = np.abs(diff / reference)
    return {'n': len(diff), 'bias': bias, 'sd': sd, 'loa_low': bias - Z95 * sd, 'loa_high': bias + Z95 * sd,
            'mae': float(np.abs(diff).mean()), 'mape': float(100 * relative.mean()), 'rmse': float(np.sqrt(np.mean(diff**2))),
            'within_5': float(100 * np.mean(np.abs(diff) <= 5)), 'within_10pct': float(100 * np.mean(relative <= 0.1)),
            'max_abs': float(np.abs(diff).max()), 'r': float(np.corrcoef(device, reference)[0, 1]),
            'ccc': lin_ccc(device, reference)}


def repeated_measures_limits(diff: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    """Limits of agreement with several samples per cluster (session) whose true value varies (Bland & Altman 2007,
    section 5): the SD of a single difference holds the spread within sessions and between them, from a one-way ANOVA."""
    frame = pd.DataFrame({'d': np.asarray(diff, dtype=float), 'c': np.asarray(clusters)})
    groups = frame.groupby('c')['d']
    sizes, means = groups.size().to_numpy(), groups.mean().to_numpy()
    k, n = len(sizes), len(frame)
    overall = frame['d'].mean()
    within = float(((frame['d'] - groups.transform('mean')) ** 2).sum() / (n - k)) if n > k else float('nan')
    if k < 2:
        sd_total = np.sqrt(within)
        between = float('nan')
    else:
        ms_between = float(np.sum(sizes * (means - overall) ** 2) / (k - 1))
        divisor = (n**2 - np.sum(sizes**2)) / ((k - 1) * n)
        between = max(0.0, (ms_between - within) / divisor)
        sd_total = np.sqrt(between + within)
    return {'clusters': k, 'n': n, 'bias': float(overall), 'sd_within': float(np.sqrt(within)), 'sd_between': float(np.sqrt(between)),
            'sd_total': float(sd_total), 'loa_low': float(overall - Z95 * sd_total), 'loa_high': float(overall + Z95 * sd_total)}


def block_bootstrap(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], block: str, n: int = 2000, seed: int = 0,
                    level: float = 0.95) -> dict[str, float]:
    """A percentile confidence interval of a statistic, drawing whole blocks (the values of column `block`) with
    replacement, so the correlation inside a block is kept."""
    rows = [np.asarray(idx) for idx in frame.groupby(block, sort=True).indices.values()]
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        picked = rng.integers(0, len(rows), len(rows))
        draws[i] = statistic(frame.iloc[np.concatenate([rows[p] for p in picked])])
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(draws, [tail, 100 - tail])
    return {'estimate': float(statistic(frame)), 'low': float(low), 'high': float(high), 'blocks': len(rows)}


def proportional_bias(frame: pd.DataFrame, x: str, block: str, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """The slope of the differences (column 'diff') on column `x` (0: the bias is the same at every level), with a
    block-bootstrap 95 % interval."""
    def slope(f: pd.DataFrame) -> float:
        """The least-squares slope."""
        return float(np.polyfit(f[x], f['diff'], 1)[0])

    interval = block_bootstrap(frame, slope, block, n=n, seed=seed)
    fit = np.polyfit(frame[x], frame['diff'], 1)
    return {'intercept': float(fit[1]), 'slope': float(fit[0]), 'slope_low': interval['low'], 'slope_high': interval['high']}


def icc(device: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Intraclass correlations with pingouin: absolute agreement ICC(A,1) and consistency ICC(C,1), single measures."""
    n = len(device)
    long = pd.DataFrame({'sample': np.tile(np.arange(n), 2), 'method': np.repeat(['device', 'reference'], n),
                         'value': np.concatenate([np.asarray(device, dtype=float), np.asarray(reference, dtype=float)])})
    with warnings.catch_warnings():
        # identical values leave no error variance, and pingouin then divides by zero for the interval
        warnings.simplefilter('ignore', RuntimeWarning)
        table = pg.intraclass_corr(data=long, targets='sample', raters='method', ratings='value').set_index('Type')
    low, high = table.loc['ICC(A,1)', 'CI95']
    return {'icc_agreement': float(table.loc['ICC(A,1)', 'ICC']), 'icc_agreement_low': float(low), 'icc_agreement_high': float(high),
            'icc_consistency': float(table.loc['ICC(C,1)', 'ICC'])}


def bootstrap_table(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], dict[str, float]], block: str, n: int = 2000,
                    seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Percentile intervals for several statistics at once (one row each: estimate, low, high), drawing whole blocks."""
    rows = [np.asarray(idx) for idx in frame.groupby(block, sort=True).indices.values()]
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([statistic(frame.iloc[np.concatenate([rows[p] for p in rng.integers(0, len(rows), len(rows))])])
                          for _ in range(n)])
    tail = (1 - level) / 2
    return pd.DataFrame({'estimate': pd.Series(statistic(frame)), 'low': draws.quantile(tail), 'high': draws.quantile(1 - tail)})


def binned(frame: pd.DataFrame, by: str, edges: list[float], block: str, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per bin of column `by` (edges as in pd.cut, left-closed): the samples, the blocks, the bias with its
    block-bootstrap 95 % interval (none with fewer than 5 blocks), and the mean absolute error of column 'diff'."""
    labels = [f'< {edges[1]:g}' if i == 0 else f'{edges[i]:g}–{edges[i + 1]:g}' for i in range(len(edges) - 1)]
    # with fewer blocks than this, a bootstrap interval means nothing
    cut = pd.cut(frame[by], edges, right=False, labels=labels)
    found = []
    for label in labels:
        part = frame[cut == label]
        if part.empty:
            continue
        interval = block_bootstrap(part, lambda f: f['diff'].mean(), block, n=n, seed=seed)
        enough = interval['blocks'] >= MIN_BLOCKS
        found.append({'bin': label, 'n': len(part), 'blocks': interval['blocks'], 'bias': interval['estimate'],
                      'bias_low': interval['low'] if enough else np.nan, 'bias_high': interval['high'] if enough else np.nan,
                      'mae': float(part['diff'].abs().mean())})
    return pd.DataFrame(found)


def decorrelation_time(values: pd.Series, groups: pd.Series, times: pd.Series, below: float = 0.1, max_lag: int = 200) -> float:
    """How long (s) the samples stay correlated: the first lag at which the autocorrelation, pooled over the groups
    (sessions), falls below `below`, times the median sample spacing. It sets the length of the bootstrap blocks."""
    frame = pd.DataFrame({'v': values.to_numpy(dtype=float), 'g': groups.to_numpy(), 't': pd.DatetimeIndex(times)})
    spacing = frame.groupby('g')['t'].diff().dt.total_seconds().median()
    sums, counts = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
    for _, part in frame.groupby('g'):
        x = part['v'].to_numpy() - part['v'].mean()
        for lag in range(min(max_lag, len(x) - 1) + 1):
            sums[lag] += np.sum(x[: len(x) - lag] * x[lag:])
            counts[lag] += len(x) - lag
    with np.errstate(invalid='ignore', divide='ignore'):
        acf = (sums / counts) / (sums[0] / counts[0])
    lag = next((k for k in range(1, max_lag + 1) if counts[k] and acf[k] < below), max_lag)
    return float(lag * spacing)
