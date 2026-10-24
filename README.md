# health_tracker_validation

How accurate and how precise are consumer health trackers? A single-person study of the **Fitbit Air** and the
**Hume Band 2.0**, with the **Polar H10** chest strap (ECG-based) as the reference for heart rate.

The data is collected and analysed here. The blog post comes later.

## What can be validated against what

| Value | Devices | Reference | Kind of result |
|---|---|---|---|
| Heart rate | Fitbit Air, Hume Band 2.0 | Polar H10 (every beat, ms timing) | accuracy and precision |
| HRV (RMSSD) | Fitbit Air, Hume Band 2.0 | Polar H10 RR intervals, only when worn at the same time | accuracy and precision |
| SpO2 | Fitbit Air | none (the Pulox PO-250 finger oximeter is broken) | description only |
| Sleep, steps, energy | Fitbit Air, Hume Band 2.0, Sleep as Android | none | agreement between devices only |

Accuracy is the **bias** (mean of device − reference). Precision is the **spread** of the differences (their SD and the
95 % limits of agreement). Both are reported separately.

## Method (heart rate)

- **Reference.** The H10 HR at a moment = the number of clean beats in the last 10 s ÷ their summed RR intervals
  (2 to 3 breaths, so the rise and fall of the HR with breathing is averaged out). Beats are cleaned: 300–2000 ms and at
  most 20 % from the median of the 5 beats before and after. Beat times are rebuilt from the intervals, because the files
  only have the arrival time of each Bluetooth message.
- **Primary analysis:** 10-s reference window, no delay assumed for the device. The window is chosen on physiological
  grounds (2 to 3 breaths), not by the size of the error. A grid of windows (2–60 s) and delays (−4 to 20 s) is the
  sensitivity analysis; it also shows how much the device smooths.
- **Statistics.** Bias, SD, limits of agreement for repeated measures (Bland & Altman 2007, session as the unit), MAE,
  MAPE, RMSE, share within ±5 bpm and ±10 %, Lin's CCC, ICC(A,1) (pingouin), proportional bias (slope of the
  differences on the reference, also on steady readings only), and the 90 % CI of the bias for an equivalence test.
  The Bland-Altman plot has the reference on the x axis (Krouwer 2008).
- **Dependence.** Readings a few seconds apart are not independent. Every confidence interval comes from a block
  bootstrap: blocks of whole minutes, twice as long as the differences stay correlated. A Fitbit reading inside two
  sessions (phone and PC at once) is paired once; a session the Automation desk marks as a copy is left out
  (a session it hides as removed stays in: the recording is fine).
- **Check.** The first look (23 Oct 2026, strap HR per second, as in the Automation desk) gave n 1991, bias +0.02,
  MAE 1.52 bpm. The tables of this project give exactly the same numbers with that method.

## Data

Everything is copied into `data/` (not in git). The sources are only read:

| Source | Where it comes from | Copy |
|---|---|---|
| Polar H10 | phone app files and PC logger sessions in `/path/to/polar_h10_logger/` | `data/raw/polar_h10/` |
| H10 activities | the activity given to each session in the Automation desk (database opened read-only) | `data/raw/automation_desk/` |
| Fitbit Air | Google Health API, every data type, one file per type and day | `data/raw/google_health/` |
| Hume Band 2.0 | the phone's Health Connect export in Dropbox | `data/raw/health_connect/` |
| Sleep as Android | the backup in Dropbox | `data/raw/sleep_as_android/` |

Only the study period is copied (from `start` in `config.toml`). A PC session that is still recording waits for the next
run. A Fitbit day is fetched again until two days after it ended (late syncs, nights that end the next day). Each new
version of an export is kept.

## Run it

```bash
uv sync
uv run htv update      # collect (copy the new data), then build the tables in data/processed/
uv run pytest
```

Then open the notebooks in `notebooks/` (kernel: `.venv/bin/python`). The notebooks are paired with `.py` files
(jupytext); the charts go to `figures/` as PNG and SVG.

| Notebook | Content |
|---|---|
| `00_data_inventory` | what each device recorded, and when |
| `01_heart_rate` | Fitbit Air heart rate against the Polar H10 |

## Setup

- `secrets/client_secret.json` (Google OAuth client, Desktop) and `secrets/google_health_token.json` (Google Health
  login with the scopes `googlehealth.activity_and_fitness.readonly`, `googlehealth.health_metrics_and_measurements.readonly`
  and `googlehealth.sleep.readonly`). The folder is not in git.
- The Google Health code is in the library `google_api` (`../../google_api`), linked as an editable path dependency.
- The paths of the sources are in `config.toml`.

## Layout

```
config.toml                      study start, time zone, source paths
src/health_tracker_validation/
  collect.py                     copy the sources into data/raw/
  polar.py                       Polar H10 files: sessions, beats, strap HR (clock change safe)
  build.py                       data/raw/ -> data/processed/*.parquet
  align.py                       H10 reference HR, pairs, delay and window curves
  stats.py                       agreement statistics, block bootstrap
  charts.py                      chart style and chart types
notebooks/                       the analysis
tests/                           pytest, no network
```

## Data still to come

- **Exercise.** The H10 is worn at every Weights and Gym cardio session. The Fitbit Air is worn from 23 Oct 2026, so
  each new session adds exercise pairs. The activity comes from the Automation desk.
- **Hume Band 2.0.** Not arrived yet (23 Oct 2026).
- **Night.** H10 nights start when the Hume Band 2.0 arrives, to check the night HR and the HRV of both bands.

## Open questions

1. **Hume Band 2.0.** The Health Connect export in Dropbox is from January. Does the Hume app write to Health Connect,
   and can the export run again (daily)?
2. **Steps.** Count steps by hand on a few walks, as the reference for steps.
3. **Automatic update.** Run `htv update` every day by a timer, or only by hand?
4. **Publication.** Will this repository and its data be public with the blog post?
