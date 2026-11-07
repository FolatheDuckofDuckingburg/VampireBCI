"""Fail-closed trial extraction for bigP3BCI EDF event channels.

This module deliberately refuses ambiguous event/target mappings. It does not
infer missing target labels or silently truncate mismatched event sequences.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pyedflib

EVENT_CHANNELS = (
    "StimulusBegin", "DisplayResults", "CurrentTarget", "SelectedTarget",
)
GRID_COLS = 8
N_CELLS = 64


class ExtractionError(ValueError):
    """Raised when an EDF cannot be mapped to unambiguous trials."""


def _rising_edges(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ExtractionError("event channel must be one-dimensional")
    if signal.size == 0:
        return np.array([], dtype=int)
    active = signal != 0
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def decode_cell(cell_index: int) -> tuple[int, int]:
    """Decode an integer 1-indexed 1..64 cell into 1-indexed (row, column)."""
    if isinstance(cell_index, (bool, np.bool_)) or not isinstance(
        cell_index, (int, np.integer)
    ):
        raise ExtractionError(
            f"cell index must be an integer in 1..{N_CELLS}; got {cell_index!r}"
        )
    if not 1 <= cell_index <= N_CELLS:
        raise ExtractionError(
            f"cell index must be in 1..{N_CELLS}; got {cell_index}"
        )
    return (cell_index - 1) // GRID_COLS + 1, (cell_index - 1) % GRID_COLS + 1


def parse_filename(path: str | os.PathLike[str]) -> tuple[str, str, str]:
    """Parse names such as M_05_SE001_ADdiff_Test01.edf."""
    name = Path(path).name
    match = re.fullmatch(
        r"(M_\d+)_(SE\d+)_([A-Za-z0-9-]+)_(Test\d+)\.edf",
        name,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ExtractionError(f"unexpected EDF filename format: {name}")
    subject, session, paradigm, test = match.groups()
    return subject.upper(), session.upper(), test


def _safe_extract_zip(archive: str | os.PathLike[str], destination: str | os.PathLike[str]) -> list[str]:
    """Extract regular files only, rejecting paths that escape destination."""
    destination = Path(destination).resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            # Reject symlinks and other non-regular entries.
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and (mode & 0o170000) == 0o120000:
                raise ExtractionError(f"ZIP symlink is not allowed: {info.filename}")
            target = (destination / info.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ExtractionError(f"unsafe ZIP member path: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted.append(str(target))
    return extracted


def find_edfs(root: str | os.PathLike[str], extract_archives: bool = False) -> list[str]:
    """Find EDFs recursively. ZIP extraction is opt-in and path-checked."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input root is not a directory: {root}")
    if extract_archives:
        for archive in root.rglob("*"):
            if archive.is_file() and archive.suffix.lower() == ".zip":
                _safe_extract_zip(archive, archive.parent)
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".edf")


def read_edf(path: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], float]:
    """Read required event channels and verify their sampling rates agree."""
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        missing = [name for name in EVENT_CHANNELS if name not in labels]
        if missing:
            raise ExtractionError(f"missing required event channels: {', '.join(missing)}")
        data: dict[str, np.ndarray] = {}
        rates: dict[str, float] = {}
        for name in EVENT_CHANNELS:
            idx = labels.index(name)
            data[name] = np.asarray(reader.readSignal(idx))
            rates[name] = float(reader.getSampleFrequency(idx))
    finally:
        reader.close()

    unique_rates = set(rates.values())
    if len(unique_rates) != 1:
        raise ExtractionError(f"event channels have different sampling rates: {rates}")
    fs = unique_rates.pop()
    if not np.isfinite(fs) or fs <= 0:
        raise ExtractionError(f"invalid sampling frequency: {fs}")
    lengths = {name: len(values) for name, values in data.items()}
    if len(set(lengths.values())) != 1:
        raise ExtractionError(f"event channels have different sample counts: {lengths}")
    return data, fs


def extract_arrays(
    data: dict[str, np.ndarray],
    fs: float,
    *,
    filename: str = "synthetic.edf",
) -> pd.DataFrame:
    """Extract trials using event intervals; reject ambiguous mappings."""
    missing = [name for name in EVENT_CHANNELS if name not in data]
    if missing:
        raise ExtractionError(f"missing required channels: {', '.join(missing)}")
    arrays = {name: np.asarray(data[name]) for name in EVENT_CHANNELS}
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ExtractionError(f"event channels have different sample counts: {lengths}")
    if not np.isfinite(fs) or fs <= 0:
        raise ExtractionError(f"invalid sampling frequency: {fs}")

    feedback = _rising_edges(arrays["DisplayResults"])
    flashes = _rising_edges(arrays["StimulusBegin"])
    selections = _rising_edges(arrays["SelectedTarget"])
    if len(feedback) == 0:
        raise ExtractionError("no DisplayResults rising edges found")

    rows = []
    previous_feedback = -1
    for trial_index, feedback_i in enumerate(feedback):
        trial_flashes = flashes[(flashes > previous_feedback) & (flashes < feedback_i)]
        trial_selections = selections[
            (selections > previous_feedback) & (selections < feedback_i)
        ]
        if len(trial_flashes) == 0:
            raise ExtractionError(f"trial {trial_index}: no stimulus onset before feedback")
        if len(trial_selections) != 1:
            raise ExtractionError(
                f"trial {trial_index}: expected exactly one selection event "
                f"between feedback events; found {len(trial_selections)}"
            )

        selection_i = int(trial_selections[0])
        # SelectedTarget may remain nonzero after its onset; use the value at
        # the event, and validate it rather than silently accepting invalid IDs.
        selection_end = selection_i + 1
        selected_signal = arrays["SelectedTarget"]
        while selection_end < len(selected_signal) and selected_signal[selection_end] != 0:
            selection_end += 1
        selected_cell = int(np.max(selected_signal[selection_i:selection_end]))
        if not 1 <= selected_cell <= N_CELLS:
            raise ExtractionError(
                f"trial {trial_index}: SelectedTarget={selected_cell} is outside 1..{N_CELLS}"
            )

        target_window = arrays["CurrentTarget"][previous_feedback + 1 : feedback_i + 1]
        nonzero_targets = target_window[target_window != 0]
        if len(nonzero_targets) == 0:
            raise ExtractionError(
                f"trial {trial_index}: CurrentTarget has no nonzero target label "
                "in this trial interval; correctness cannot be determined"
            )
        true_target = int(nonzero_targets[-1])
        if not 1 <= true_target <= N_CELLS:
            raise ExtractionError(
                f"trial {trial_index}: CurrentTarget={true_target} is outside 1..{N_CELLS}"
            )

        t0_i = int(trial_flashes[0])
        row, col = decode_cell(selected_cell)
        rows.append({
            "file": Path(filename).name,
            "trial": trial_index,
            "t0_s": t0_i / fs,
            "tfb_s": int(feedback_i) / fs,
            "L_s": (int(feedback_i) - t0_i) / fs,
            "selected_cell": selected_cell,
            "selected_row": row,
            "selected_col": col,
            "true_target": true_target,
            "Y": int(selected_cell == true_target),
            "n_flashes": int(len(trial_flashes)),
        })
        previous_feedback = int(feedback_i)

    subject, session, test = parse_filename(filename)
    frame = pd.DataFrame(rows)
    frame["subject"] = subject
    frame["session"] = session
    frame["test"] = test
    return frame


def extract_file(path: str | os.PathLike[str], verbose: bool = True) -> pd.DataFrame:
    data, fs = read_edf(path)
    frame = extract_arrays(data, fs, filename=str(path))
    if verbose:
        print(
            f"{Path(path).name}: {len(frame)} trials; "
            f"accuracy={frame['Y'].mean():.3f}; "
            f"mean L={frame['L_s'].mean():.3f}s"
        )
    return frame


def main(
    root: str | os.PathLike[str],
    out_csv: str | os.PathLike[str] = "pooled_trials_full.csv",
    *,
    extract_archives: bool = False,
) -> pd.DataFrame:
    paths = find_edfs(root, extract_archives=extract_archives)
    if not paths:
        raise FileNotFoundError(f"no EDF files found under {root}")

    frames = []
    failures = []
    for path in paths:
        try:
            frames.append(extract_file(path))
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"[REJECTED] {Path(path).name}: {exc}")

    if not frames:
        details = "; ".join(f"{Path(p).name}: {reason}" for p, reason in failures)
        raise ExtractionError(f"no files yielded valid trials. {details}")

    pooled = pd.concat(frames, ignore_index=True)
    pooled["_subject_number"] = pooled["subject"].str.extract(r"M_(\d+)", expand=False).astype(int)
    pooled = pooled.sort_values(
        ["_subject_number", "session", "test", "trial"]
    ).drop(columns="_subject_number").reset_index(drop=True)
    pooled.to_csv(out_csv, index=False)
    print(
        f"Saved {len(pooled)} trials from {pooled['file'].nunique()} files "
        f"and {pooled['subject'].nunique()} subjects to {out_csv}"
    )
    if failures:
        print(f"Rejected {len(failures)} file(s); see messages above.")
    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory containing bigP3BCI EDF files")
    parser.add_argument("--out", default="pooled_trials_full.csv", help="Output CSV path")
    parser.add_argument(
        "--extract-archives", action="store_true",
        help="Extract ZIP archives after validating member paths",
    )
    args = parser.parse_args()
    main(args.root, args.out, extract_archives=args.extract_archives)
