"""The look of every chart, and the chart types of the analysis.

One color per device, the same in every chart: the Polar H10 (the reference) blue, the Fitbit Air orange, the Hume Band
2.0 aqua, Sleep as Android yellow. Text is never in a device color; the marks carry it. The palette is the validated
default categorical palette in its fixed order (the first three slots pass the colour-blindness checks for every pair)."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from health_tracker_validation.config import FIGURES, settings

DEVICE = {'Polar H10': '#2a78d6', 'Fitbit Air': '#eb6834', 'Hume Band 2.0': '#1baf7a', 'Sleep as Android': '#eda100'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BAND = '#f0efec'
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def signed(value: float, digits: int = 1) -> str:
    """A number with its sign and a true minus sign; never −0.0."""
    text = f'{value:+.{digits}f}'
    if float(text) == 0:
        text = f'{0:+.{digits}f}'
    return text.replace('-', '−')


def use_style() -> None:
    """The chart style: one sans font, quiet axes and grid, the surface color."""
    mpl.rcParams.update({
        'font.family': ['Adwaita Sans', 'DejaVu Sans'], 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'semibold',
        'axes.titlelocation': 'left', 'axes.titlepad': 10, 'axes.labelsize': 10, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8, 'axes.facecolor': SURFACE, 'axes.grid': True, 'axes.axisbelow': True, 'axes.spines.top': False,
        'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.7, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0, 'text.color': INK,
        'figure.facecolor': SURFACE, 'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.facecolor': SURFACE,
        'savefig.bbox': 'tight', 'legend.frameon': False, 'legend.fontsize': 9, 'lines.linewidth': 1.6,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round', 'svg.fonttype': 'none',
    })


def save(fig: Figure, name: str, folder: Path = FIGURES) -> Path:
    """Save a figure as PNG (for the post) and SVG (for editing) in figures/."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.svg')
    fig.savefig(folder / f'{name}.png')
    return folder / f'{name}.png'


def heading(fig: Figure, title: str, subtitle: str = '') -> None:
    """A left-aligned title with a muted subtitle above the plots."""
    fig.suptitle(title, x=0.01, y=1.0, ha='left', va='bottom', fontsize=13, fontweight='semibold', color=INK)
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha='left', va='top', fontsize=9.5, color=INK2)


def local_time_axis(ax: Axes) -> None:
    """Hours and minutes of the local clock on the x axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=settings().tz))


def _local(times: pd.Series) -> pd.Series:
    """UTC times on the local clock, for plotting."""
    return pd.Series(pd.DatetimeIndex(times).tz_convert(settings().tz))


def session_hr(ax_hr: Axes, ax_diff: Axes, reference: pd.DataFrame, paired: pd.DataFrame, device: str = 'Fitbit Air') -> None:
    """One session: the H10 reference HR as a line and the device samples over it; below, the difference of each sample."""
    ax_hr.plot(_local(reference['time']), reference['h10'], color=DEVICE['Polar H10'], lw=1.6, label='Polar H10 (reference)')
    ax_hr.plot(_local(paired['time']), paired['device'], color=DEVICE[device], lw=1.1, alpha=0.95, label=device)
    ax_hr.set_ylabel('Heart rate (bpm)')
    ax_hr.legend(loc='upper left', ncols=2, handlelength=1.4)
    ax_diff.axhline(0, color=AXIS, lw=0.9)
    ax_diff.fill_between(_local(paired['time']), paired['diff'], 0, color=DEVICE[device], alpha=0.18, lw=0)
    ax_diff.plot(_local(paired['time']), paired['diff'], color=DEVICE[device], lw=0.9)
    ax_diff.set_ylabel(f'{device.split()[0]} − H10')
    local_time_axis(ax_diff)


def bland_altman(ax: Axes, paired: pd.DataFrame, limits: dict, ci: dict | None = None, slope: dict | None = None,
                 device: str = 'Fitbit Air') -> Axes:
    """Differences against the reference (Krouwer 2008: with a reference method, plot against it, not the mean), with the
    bias and the 95 % limits of agreement (shaded: their 95 % confidence intervals), the regression line of a proportional
    bias, and the spread of the differences at the right. Returns the side axis."""
    ax.scatter(paired['h10'], paired['diff'], s=11, color=DEVICE[device], alpha=0.28, linewidths=0.3, edgecolors=SURFACE)
    x0, x1 = paired['h10'].min() - 2, paired['h10'].max() + 2
    dashed = (0, (4, 3))
    for key, label, style in (('bias', 'bias', '-'), ('loa_low', 'lower limit', dashed), ('loa_high', 'upper limit', dashed)):
        value = limits[key]
        if ci and key in ci:
            ax.axhspan(ci[key][0], ci[key][1], color=INK2, alpha=0.09, lw=0)
        ax.axhline(value, color=INK2, lw=1.1, ls=style)
        ax.annotate(f'{label} {signed(value)}', (1, value), xycoords=('axes fraction', 'data'), xytext=(-4, 4),
                    textcoords='offset points', ha='right', va='bottom', fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.9)
    if slope:
        xs = np.array([x0, x1])
        ax.plot(xs, slope['intercept'] + slope['slope'] * xs, color=INK, lw=0.9, alpha=0.6)
    ax.set_xlim(x0, x1)
    ax.set_xlabel('Polar H10 (bpm)')
    ax.set_ylabel(f'{device} − H10 (bpm)')
    side = ax.inset_axes((1.015, 0, 0.13, 1), sharey=ax)
    side.hist(paired['diff'], bins=np.arange(np.floor(paired['diff'].min()) - 0.5, paired['diff'].max() + 1, 1),
              orientation='horizontal', color=DEVICE[device], alpha=0.55, lw=0)
    side.axhline(limits['bias'], color=INK2, lw=1.1)
    side.grid(False)
    side.tick_params(labelleft=False, labelbottom=False)
    for spine in side.spines.values():
        spine.set_visible(False)
    return side


def identity(ax: Axes, paired: pd.DataFrame, device: str = 'Fitbit Air') -> None:
    """The device against the H10, with the line where they are equal."""
    low = np.floor(min(paired['h10'].min(), paired['device'].min())) - 2
    high = np.ceil(max(paired['h10'].max(), paired['device'].max())) + 2
    ax.plot([low, high], [low, high], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.scatter(paired['h10'], paired['device'], s=11, color=DEVICE[device], alpha=0.28, linewidths=0.3, edgecolors=SURFACE)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal')
    ax.set_xlabel('Polar H10 (bpm)')
    ax.set_ylabel(f'{device} (bpm)')
    ax.annotate('equal', (high, high), xytext=(-4, -12), textcoords='offset points', ha='right', fontsize=8.5, color=INK2)


def heatmap(ax: Axes, grid: pd.DataFrame, value: str, x: str, y: str, label: str, mark: tuple[float, float] | None = None) -> None:
    """A grid of a value (darker = larger) over two settings, the smallest y at the top. The smallest cell is ringed and
    `mark` (x, y), the cell the analysis uses, gets a dashed ring; both show their value."""
    table = grid.pivot(index=y, columns=x, values=value).sort_index()
    values = table.to_numpy()
    cmap = mpl.colors.LinearSegmentedColormap.from_list('blues', BLUES)
    image = ax.imshow(values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(table.columns)), [f'{c:g}'.replace('-', '−') for c in table.columns])
    ax.set_yticks(range(len(table.index)), [f'{i:g}'.replace('-', '−') for i in table.index])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    low, high = np.nanmin(values), np.nanmax(values)
    cells = [(np.unravel_index(np.nanargmin(values), values.shape), (0, ()))]
    if mark is not None:
        cells.append(((list(table.index).index(mark[1]), list(table.columns).index(mark[0])), (0, (2, 2))))
    for (row, col), style in cells:
        ax.scatter([col], [row], s=560, facecolors='none', edgecolors=INK, linewidths=1.3, linestyles=[style])
        dark = (values[row, col] - low) / (high - low or 1) > 0.55
        ax.annotate(f'{values[row, col]:.2f}', (col, row), ha='center', va='center', fontsize=8.5, color='white' if dark else INK)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5, colors=INK2)
    bar.set_label(label, color=INK2)


def binned(ax: Axes, table: pd.DataFrame, x_label: str, device: str = 'Fitbit Air') -> None:
    """The bias (dot, with its 95 % interval) and the mean absolute error (ring) per bin, with the samples of each bin."""
    x = np.arange(len(table))
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.vlines(x, table['bias_low'], table['bias_high'], color=DEVICE[device], lw=2, alpha=0.55)
    ax.scatter(x, table['bias'], s=46, color=DEVICE[device], edgecolors=SURFACE, linewidths=1.5, zorder=3, label='bias (95 % CI)')
    ax.scatter(x, table['mae'], s=46, facecolors=SURFACE, edgecolors=INK2, linewidths=1.3, zorder=3, label='mean absolute error')
    ax.set_xticks(x, [f'{label}\nn {n}' for label, n in zip(table['bin'], table['n'], strict=True)])
    ax.set_xlabel(x_label)
    ax.set_ylabel('bpm')
    low = min(table['bias_low'].min(), table['bias'].min(), 0)
    high = max(table['mae'].max(), table['bias_high'].max(), table['bias'].max())
    ax.set_ylim(low - 0.08 * (high - low), high + 0.3 * (high - low))
    ax.legend(loc='upper left', ncols=2)


def ecdf(ax: Axes, abs_error: np.ndarray, marks: tuple[float, ...] = (2, 5, 10), device: str = 'Fitbit Air') -> None:
    """The share of samples with an error up to each size, with the shares at a few sizes."""
    values = np.sort(np.asarray(abs_error))
    share = np.arange(1, len(values) + 1) / len(values) * 100
    ax.step(values, share, where='post', color=DEVICE[device], lw=1.8)
    for mark in marks:
        at = float(np.mean(values <= mark) * 100)
        ax.plot([mark, mark], [0, at], color=AXIS, lw=0.9)
        ax.scatter([mark], [at], s=40, color=DEVICE[device], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(f'{at:.0f} % within {mark:g} bpm', (mark, at), xytext=(6, -12), textcoords='offset points', fontsize=8.5,
                    color=INK2)
    ax.set_xlim(0, max(values.max(), max(marks)) + 0.5)
    ax.set_ylim(0, 101)
    ax.set_xlabel('Absolute error (bpm)')
    ax.set_ylabel('Readings (%)')


def lanes(ax: Axes, spans: dict[str, tuple[str, list[tuple[pd.Timestamp, pd.Timestamp]]]]) -> None:
    """When each source has data: one lane per source (label -> device, spans), on the local clock."""
    for row, (device, intervals) in enumerate(spans.values()):
        bars = [(mdates.date2num(a.tz_convert(settings().tz)), mdates.date2num(b.tz_convert(settings().tz))
                 - mdates.date2num(a.tz_convert(settings().tz))) for a, b in intervals]
        ax.broken_barh(bars, (row - 0.32, 0.64), color=DEVICE.get(device, MUTED), lw=0)
        if not intervals:
            ax.annotate('no data yet', (0.005, row), xycoords=('axes fraction', 'data'), va='center', fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(spans)), list(spans))
    ax.set_ylim(len(spans) - 0.5, -0.5)
    ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=settings().tz))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator(), tz=settings().tz))
