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
# # Data inventory
#
# What each device has recorded since the start of the study, and when. Update the data first:
#
# ```bash
# uv run htv update
# ```
#
# `collect` copies the new files and API data into `data/raw/`; `build` turns them into the tables in `data/processed/`.

# %%
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

from health_tracker_validation import charts, inventory
from health_tracker_validation.config import RAW, settings
from health_tracker_validation.data import load

charts.use_style()
tz = settings().tz
sessions, beats, fitbit_hr, points = load('h10_sessions'), load('h10_beats'), load('fitbit_hr'), load('google_health')
print(f'Study start: {settings().start:%d %b %Y %H:%M}. Now: {pd.Timestamp.now(tz):%d %b %Y %H:%M}.')

# %% [markdown]
# ## Sources

# %%
def local(t: pd.Timestamp) -> str:
    """A time on the local clock, short."""
    return '' if pd.isna(t) else f'{t.tz_convert(tz):%d %b %H:%M}'


hume_exports = sorted((RAW / 'health_connect').glob('*.zip'))
sleep_backups = sorted((RAW / 'sleep_as_android').glob('*.zip'))
sources = pd.DataFrame([
    {'device': 'Polar H10', 'data': 'sessions (RR intervals, strap HR)', 'rows': len(sessions),
     'first': local(sessions['start'].min()), 'last': local(sessions['end'].max())},
    {'device': 'Polar H10', 'data': 'beats', 'rows': len(beats), 'first': local(beats['time'].min()),
     'last': local(beats['time'].max())},
    {'device': 'Fitbit Air', 'data': 'heart rate samples', 'rows': len(fitbit_hr), 'first': local(fitbit_hr['time'].min()),
     'last': local(fitbit_hr['time'].max())},
    {'device': 'Fitbit Air', 'data': 'all Google Health values', 'rows': len(points), 'first': local(points['start'].min()),
     'last': local(points['start'].max())},
    {'device': 'Hume Band 2.0', 'data': 'Health Connect exports', 'rows': len(hume_exports), 'first': '', 'last': ''},
    {'device': 'Sleep as Android', 'data': 'backups', 'rows': len(sleep_backups), 'first': '', 'last': ''},
])
display(sources)

# %% [markdown]
# ## When each source has data

# %%
def interval_spans(kind: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The intervals of a Google Health type (one per point)."""
    part = points[(points['type'] == kind) & points['end'].notna()].drop_duplicates(['start', 'end'])
    return list(zip(part['start'], part['end'], strict=True))


steps = points[(points['type'] == 'steps') & (points['device'] == 'Fitbit Air')]
lanes = {
    'Fitbit · heart rate': ('Fitbit Air', inventory.spans(fitbit_hr['time'], 60)),
    'Fitbit · minutes with steps': ('Fitbit Air', inventory.spans(steps['start'], 120)),
    'Fitbit · sleep': ('Fitbit Air', interval_spans('sleep')),
    'Polar H10 · sessions': ('Polar H10', list(zip(sessions['start'], sessions['end'], strict=True))),
    'Hume Band 2.0': ('Hume Band 2.0', []),
    'Sleep as Android': ('Sleep as Android', []),
}
fig, ax = plt.subplots(figsize=(11, 3.4))
charts.lanes(ax, lanes)
charts.heading(fig, 'Data per source over time', 'Local time. A gap in a bar is a time without data.')
charts.save(fig, 'inventory_lanes')
plt.show()

# %% [markdown]
# ## Polar H10 sessions

# %%
table = sessions.assign(minutes=((sessions['end'] - sessions['start']).dt.total_seconds() / 60).round(1),
                        clean_pct=(100 * sessions['clean_beats'] / sessions['beats']).round(1), start=sessions['start'].map(local))
display(table[['key', 'activity', 'start', 'minutes', 'beats', 'clean_pct', 'streams']])

# %% [markdown]
# ## Fitbit Air: every Google Health data type

# %%
by_type = (points.groupby('type')
           .agg(rows=('field', 'size'), fields=('field', 'nunique'), first=('start', 'min'), last=('start', 'max'),
                devices=('device', lambda d: ', '.join(sorted(d.dropna().unique()))))
           .assign(first=lambda f: f['first'].map(local), last=lambda f: f['last'].map(local))
           .sort_values('rows', ascending=False))
display(by_type)
