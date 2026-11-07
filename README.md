# VampireBCI

A conservative Python pipeline for extracting trial-level feedback timing data from bigP3BCI-format EDF files.

## What it extracts

For each accepted trial, VampireBCI records:

- `t0_s`: first `StimulusBegin` rising edge in the trial interval
- `tfb_s`: `DisplayResults` rising edge
- `L_s`: write-back gap, `tfb_s - t0_s`
- `selected_cell`, `selected_row`, `selected_col`: decoded selection
- `true_target`: final nonzero `CurrentTarget` label within the trial interval
- `Y`: 1 when selected cell equals the target, otherwise 0
- `n_flashes`: number of stimulus onsets in the interval

The CSV also includes file, subject, session, and test provenance.

## Important validation rule

**The extractor fails closed.** It rejects a file or trial when the event streams cannot be paired unambiguously, channel lengths or sampling rates disagree, a selection is missing or duplicated, or no valid nonzero `CurrentTarget` label exists in that trial interval.

It does not convert missing target labels into incorrect trials, truncate mismatched event counts with `min()`, or infer target identity from another channel. If `CurrentTarget` is absent or all zero in a recording, correctness cannot be established by this implementation. Resolve the event semantics against the dataset documentation and raw EDF before changing that rule.

The extractor currently assumes the event channels share a common sampling rate and that one selection event occurs between successive feedback events. These assumptions must be checked against each dataset variant.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python vampirebci.py /path/to/edfs --out pooled_trials_full.csv
```

ZIP extraction is disabled by default. To extract archives, use the path-checked option:

```bash
python vampirebci.py /path/to/edfs --extract-archives
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic event arrays. They verify software behavior only; they do not validate the scientific interpretation of the bigP3BCI event channels or reproduce any historical Study M result.

## Historical notebook

`VampireBCI.py` is the original Colab notebook export, including historical outputs. It is retained for provenance and should not be treated as the maintained extraction module. The maintained implementation is `vampirebci.py`.
