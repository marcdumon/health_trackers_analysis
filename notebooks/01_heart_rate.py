# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Heart rate: Fitbit Air against the Polar H10
#
# **Question.** How close is the heart rate of the Fitbit Air (an optical sensor on the wrist) to the true heart rate?
#
# **Reference.** The Polar H10 chest strap measures the electrical signal of the heart and times every beat to the
# millisecond. Its HR at a moment is the number of clean beats in a short window divided by their summed intervals.
#
# **Terms.**
# - *Accuracy* is the **bias**: the mean of Fitbit − H10. Zero means no systematic error.
# - *Precision* is the **spread** of those differences: their SD, and the 95 % limits of agreement (bias ± 1.96 SD).
#   About 95 % of single readings fall between the limits.
# - Error sizes: mean absolute error (MAE), mean absolute percentage error (MAPE), root mean square error (RMSE).
# - Agreement on one scale: Lin's concordance correlation (CCC) and the intraclass correlation ICC(A,1).
#
# **Samples are not independent.** The Fitbit reports about every 3 s, and neighbouring errors are alike. So:
# - the limits of agreement use the method for repeated measures (Bland & Altman 2007), with the session as the unit;
# - every confidence interval comes from a block bootstrap: whole blocks of minutes are drawn, not single samples.

# %%
import math

import pandas as pd
from IPython.display import Markdown, display
from matplotlib import pyplot as plt

from health_tracker_validation import align, charts, stats
from health_tracker_validation.config import settings
from health_tracker_validation.data import load

charts.use_style()
tz = settings().tz
# the primary analysis: the H10 HR over the last 10 s, chosen on physiological grounds (2 to 3 breaths, so the normal rise
# and fall of the HR with each breath is averaged out), not by the size of the error; no delay assumed for the Fitbit.
# The grid below shows how the result depends on this choice.
WINDOW_S, LAG_S = 10, 0
BOOTSTRAP = 2000
# a reading is steady when the H10 HR changed less than this over the last 10 s (bpm)
STEADY_BPM = 2

sessions, beats, fitbit = load('h10_sessions'), load('h10_beats'), load('fitbit_hr')
sessions = sessions[(sessions['end'] - sessions['start']) >= pd.Timedelta(minutes=2)].copy()
sessions = sessions[(sessions['removed'].fillna(0) == 0) & sessions['copy_of'].isna()].copy()
sessions['activity'] = sessions['activity'].fillna('not set')
paired = align.pairs(fitbit, beats, sessions, WINDOW_S, LAG_S)
# how fast the HR changes: the H10 HR now minus 10 s ago
paired['change'] = pd.concat([pd.Series(part['h10'].to_numpy()
                                        - align.reference_hr(beats[beats['session'] == key], part['time'], WINDOW_S, lag_s=10),
                                        index=part.index) for key, part in paired.groupby('session')])
hours = paired.groupby('session')['time'].agg(lambda t: (t.max() - t.min()).total_seconds()).sum() / 3600
print(f'{len(paired)} pairs in {paired["session"].nunique()} sessions, {hours:.1f} h; '
      f'H10 {paired["h10"].min():.0f}–{paired["h10"].max():.0f} bpm.')

# %% [markdown]
# ## How the Fitbit reports its heart rate
#
# A wrist device smooths its HR and may report it with a delay. The grid gives the mean absolute error for each
# combination of the reference window (how many seconds of H10 beats are averaged) and an assumed delay of the Fitbit
# (negative: the window ends after the reading, e.g. for a small clock difference between phone and band).
# The ringed cell is the smallest error; the dashed ring is the primary analysis.

# %%
grid = pd.concat([align.lag_curve(fitbit, beats, sessions, range(-4, 21, 2), window).assign(window_s=window)
                  for window in (2, 5, 10, 15, 20, 30, 45, 60)], ignore_index=True)
best = grid.loc[grid['mae'].idxmin()]
fig, ax = plt.subplots(figsize=(8.5, 5))
charts.heatmap(ax, grid, 'mae', 'window_s', 'lag_s', 'Mean absolute error (bpm)', mark=(WINDOW_S, LAG_S))
ax.set_xlabel('Reference: H10 heart rate over the last … s')
ax.set_ylabel('Assumed delay of the Fitbit (s)')
shift = 'at' if best['lag_s'] == 0 else f'{abs(best["lag_s"]):g} s {"before" if best["lag_s"] > 0 else "after"}'
charts.heading(fig, f'The Fitbit reads like an average of about {best["window_s"]:g} seconds',
               f'Smallest error ({best["mae"]:.2f} bpm): the H10 averaged over {best["window_s"]:g} s, ending {shift} each reading. '
               f'Primary analysis: {WINDOW_S} s, no delay.')
charts.save(fig, 'hr_lag_window_grid')
plt.show()

# %% [markdown]
# ## Each session

# %%
for session in sessions.itertuples():
    part = paired[paired['session'] == session.key]
    if len(part) < 20:
        continue
    seconds = pd.Series(pd.date_range(session.start, session.end, freq='1s'))
    reference = pd.DataFrame({'time': seconds, 'h10': align.reference_hr(beats[beats['session'] == session.key], seconds, WINDOW_S)})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True, gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})
    charts.session_hr(top, bottom, reference, part)
    start, end = session.start.tz_convert(tz), session.end.tz_convert(tz)
    part_stats = stats.agreement(part['device'], part['h10'])
    name = '' if session.activity == 'not set' else f'{session.activity}, '
    charts.heading(fig, f'{name}{start:%a %d %b %H:%M}–{end:%H:%M}',
                   f'Bias {charts.signed(part_stats["bias"])} bpm, MAE {part_stats["mae"]:.1f} bpm, {len(part)} pairs. '
                   f'H10 over the last {WINDOW_S} s; Fitbit as it reports, about every 3 s.')
    charts.save(fig, f'hr_session_{session.key}')
    plt.show()

# %% [markdown]
# ## Agreement, with 95 % confidence intervals
#
# The bootstrap blocks last twice as long as the differences stay correlated (rounded up to whole minutes), to be safe.

# %%
block_s = stats.decorrelation_time(paired['diff'], paired['session'], paired['time'])
BLOCK_MIN = max(1, math.ceil(2 * block_s / 60))
paired['block'] = paired['session'] + ' ' + pd.DatetimeIndex(paired['time']).floor(f'{BLOCK_MIN}min').astype(str)
print(f'The differences stay correlated for about {block_s:.0f} s: blocks of {BLOCK_MIN} min, '
      f'{paired["block"].nunique()} blocks.')


def summary(frame: pd.DataFrame) -> dict[str, float]:
    """The headline statistics of some pairs."""
    sample = stats.agreement(frame['device'], frame['h10'])
    limits = stats.repeated_measures_limits(frame['diff'], frame['session'])
    return {'bias (bpm)': sample['bias'], 'SD of differences (bpm)': limits['sd_total'],
            'lower limit of agreement (bpm)': limits['loa_low'], 'upper limit of agreement (bpm)': limits['loa_high'],
            'MAE (bpm)': sample['mae'], 'MAPE (%)': sample['mape'], 'RMSE (bpm)': sample['rmse'],
            'within ±5 bpm (%)': sample['within_5'], 'within ±10 % (%)': sample['within_10pct'], 'Lin CCC': sample['ccc']}


table = stats.bootstrap_table(paired, summary, 'block', n=BOOTSTRAP)
display(table.round(2))
icc = stats.icc(paired['device'], paired['h10'])
slope = stats.proportional_bias(paired, 'h10', 'block', n=BOOTSTRAP)
steady = paired[paired['change'].abs() < STEADY_BPM]
steady_slope = stats.proportional_bias(steady, 'h10', 'block', n=BOOTSTRAP)
# pingouin's interval for the ICC assumes independent samples, so it is too narrow here; the CCC above has a block interval
print(f'ICC(A,1) {icc["icc_agreement"]:.3f}; ICC(C,1) {icc["icc_consistency"]:.3f}.')
print(f'Proportional bias: {slope["slope"]:+.3f} bpm per bpm of H10 (95 % CI {slope["slope_low"]:+.3f} to '
      f'{slope["slope_high"]:+.3f}); steady readings only ({len(steady)}): {steady_slope["slope"]:+.3f} '
      f'({steady_slope["slope_low"]:+.3f} to {steady_slope["slope_high"]:+.3f}).')
bias90 = stats.block_bootstrap(paired, lambda f: f['diff'].mean(), 'block', n=BOOTSTRAP, level=0.90)
print(f'90 % CI of the bias, for an equivalence test (TOST) with any margin: {bias90["low"]:+.2f} to {bias90["high"]:+.2f} bpm.')

# %% [markdown]
# ## Bland-Altman plot
#
# Each dot is one Fitbit reading. Lines: the bias and the 95 % limits of agreement; shaded: their 95 % confidence
# intervals. The x axis is the H10, not the mean of the two: with a reference method, the mean mixes in the device's own
# error (Krouwer 2008). The thin line is the regression of the differences on the H10. Its negative slope is partly an
# effect of the smoothing: at rest, a high H10 value is mostly a short peak, which the Fitbit averages away. On steady
# readings the slope is smaller (see above). Data with exercise must show whether the bias truly changes with the heart
# rate. The diagonal stripes come from the Fitbit's whole-number readings.

# %%
limits = stats.repeated_measures_limits(paired['diff'], paired['session'])
ci = {'bias': table.loc['bias (bpm)', ['low', 'high']].tolist(),
      'loa_low': table.loc['lower limit of agreement (bpm)', ['low', 'high']].tolist(),
      'loa_high': table.loc['upper limit of agreement (bpm)', ['low', 'high']].tolist()}
fig, ax = plt.subplots(figsize=(10, 5.6))
charts.bland_altman(ax, paired, limits, ci, slope)
charts.heading(fig, 'Fitbit Air minus Polar H10',
               f'{len(paired)} readings in {paired["session"].nunique()} sessions. 95 % of single readings lie between '
               f'{charts.signed(limits["loa_low"])} and {charts.signed(limits["loa_high"])} bpm of the H10.')
charts.save(fig, 'hr_bland_altman')
plt.show()

# %% [markdown]
# ## The readings against the reference, and the size of the errors

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.28})
charts.identity(left, paired)
left.set_title('Fitbit against H10')
charts.ecdf(right, paired['diff'].abs().to_numpy())
right.set_title('Share of readings within an error')
charts.heading(fig, 'How often the Fitbit is close', f'Lin CCC {table.loc["Lin CCC", "estimate"]:.2f}; '
               f'MAE {table.loc["MAE (bpm)", "estimate"]:.1f} bpm.')
charts.save(fig, 'hr_identity_ecdf')
plt.show()

# %% [markdown]
# ## When the Fitbit errs
#
# By how fast the heart rate changes (the H10 HR now minus 10 s ago), and by its level on steady readings only (a change
# under 2 bpm in 10 s), so that the level is not mixed up with short peaks. A wrist sensor that smooths its value lags
# behind a rise, and reads low while the HR goes up. Bins with fewer than 5 blocks show no interval.

# %%
low, high = 10 * math.floor(steady['h10'].min() / 10), 10 * math.ceil(steady['h10'].max() / 10)
by_level = stats.binned(steady, 'h10', list(range(low, high + 1, 5 if high - low <= 60 else 10)), 'block')
by_change = stats.binned(paired.dropna(subset=['change']), 'change', [-100, -6, -3, -1, 1, 3, 6, 100], 'block')
names = {'< -6': '< −6', '-6–-3': '−6 to −3', '-3–-1': '−3 to −1', '-1–1': '−1 to 1', '1–3': '1 to 3', '3–6': '3 to 6', '6–100': '> 6'}
by_change['bin'] = by_change['bin'].map(names)
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'wspace': 0.22})
charts.binned(left, by_level, 'H10 heart rate (bpm)')
left.set_title('By heart rate, steady readings')
charts.binned(right, by_change, 'H10 change over the last 10 s (bpm; negative: falling)')
right.set_title('By change of heart rate')
charts.heading(fig, 'The error depends on how fast the heart rate changes',
               'Bias with its 95 % interval, and the mean absolute error.')
charts.save(fig, 'hr_error_by_level_and_change')
plt.show()
display(by_change.round(2))

# %% [markdown]
# ## Per session and per activity

# %%
def group_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """The main statistics per group."""
    rows = []
    for name, part in frame.groupby(by):
        sample = stats.agreement(part['device'], part['h10'])
        rows.append({by: name, 'pairs': sample['n'], 'H10 mean': part['h10'].mean(), 'bias': sample['bias'], 'SD': sample['sd'],
                     'MAE': sample['mae'], 'MAPE (%)': sample['mape'], 'within ±5 bpm (%)': sample['within_5']})
    return pd.DataFrame(rows).set_index(by).round(2)


display(group_table(paired, 'session'))
display(group_table(paired, 'activity'))

# %% [markdown]
# ## Summary

# %%
row = table['estimate']
display(Markdown(
    f'- **Data:** {len(paired)} Fitbit readings against the H10 in {paired["session"].nunique()} sessions ({hours:.1f} h), '
    f'H10 {paired["h10"].min():.0f}–{paired["h10"].max():.0f} bpm.\n'
    f'- **Accuracy:** bias {row["bias (bpm)"]:+.2f} bpm (95 % CI {table.loc["bias (bpm)", "low"]:+.2f} to '
    f'{table.loc["bias (bpm)", "high"]:+.2f}).\n'
    f'- **Precision:** SD of the differences {row["SD of differences (bpm)"]:.2f} bpm; 95 % limits of agreement '
    f'{row["lower limit of agreement (bpm)"]:+.1f} to {row["upper limit of agreement (bpm)"]:+.1f} bpm.\n'
    f'- **Error size:** MAE {row["MAE (bpm)"]:.2f} bpm, MAPE {row["MAPE (%)"]:.1f} %, '
    f'{row["within ±5 bpm (%)"]:.0f} % of readings within ±5 bpm.\n'
    f'- **Smoothing:** the error is smallest against a {best["window_s"]:g}-s H10 average with a {best["lag_s"]:+g}-s delay '
    f'({best["mae"]:.2f} bpm).\n'
    f'- **Limit:** one person; only the conditions recorded so far.'))
