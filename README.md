# VampireBCI

A Python pipeline for extracting trial-level feedback timing data from bigP3BCI-format EDF files.

## What it does

VampireBCI reads raw EEG files from the bigP3BCI dataset (PhysioNet) and extracts, for each trial:

- **`t0_s`** — the onset of the first stimulus in the trial
- **`tfb_s`** — the moment feedback was delivered
- **`L_s`** — the Write-Back Gap, `tfb − t0`, in seconds
- **`Y`** — whether the selected cell matched the target (1) or not (0)
- **`selected_cell`, `selected_row`, `selected_col`** — the decoded selection
- **`true_target`** — the true target cell for that trial
- **`n_flashes`** — number of stimulus flashes in the trial

The output is a pooled CSV with one row per trial, plus provenance columns (`file`, `subject`, `session`, `test`).

## Why it exists

Public BCI datasets often record the raw event channels but don't publish trial-level selection or feedback timing in an accessible format. In bigP3BCI Study M, the information is present in the EDF event channels — `StimulusBegin`, `DisplayResults`, `CurrentTarget`, `SelectedTarget` — but it has to be decoded.

VampireBCI does that decoding. It was written to make trial-level `(L, Y)` data extractable from the raw files, for analyses that depend on when feedback was delivered, not just whether it was correct.

## How it works

1. **Find EDFs.** Recursively searches a root directory, case-insensitive, unzipping any `.zip` archives it finds.
2. **Read event channels.** Uses `pyedflib` to read `StimulusBegin`, `DisplayResults`, `CurrentTarget`, `SelectedTarget`, and related channels.
3. **Find feedback onsets.** `DisplayResults` rising edges mark the moment feedback was delivered.
4. **Find the trial's first stimulus.** `StimulusBegin` rising edges mark stimulus onsets; the first one after the previous feedback is taken as `t0`.
5. **Decode the selection.** `SelectedTarget` values encode the chosen cell as `(row − 1) × 8 + col`, 1-indexed. Verified on 4192/4192 samples with zero exceptions.
6. **Determine correctness.** `Y = 1` if the selected cell equals the last nonzero `CurrentTarget` before feedback, else `0`.
7. **Pool and save.** Writes `pooled_trials_full.csv` with all trials, sorted by subject, session, test, and trial.

## Requirements
pyedflib
numpy
pandas


Install with:

```
pip install -r requirements.txt
```

## Usage

```
from VampireBCI import main

pooled = main(root="/path/to/edfs", out_csv="pooled_trials_full.csv")
```

Or from the command line:

```
python VampireBCI.py
```

By default it searches
```
/content
```
 and writes
 ```
pooled_trials_full.csv
````
 to the working directory.

## Input

EDF files from bigP3BCI Study M, named like:
```
M_XX_SEXXX_ADdiff_TestXX.edf
```

Example: 

```
M_05_SE001_ADdiff_Test01.edf.
```
Subject, session, and test are parsed from the filename.

---

Developed against bigP3BCI Study M and verified trial-by-trial against raw channels. May not generalize to other studies without modification.
