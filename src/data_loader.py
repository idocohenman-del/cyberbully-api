"""
data_loader.py
──────────────
Loads every dataset listed in config/datasets.yaml, applies the per-dataset
label mapping, and returns a single unified DataFrame with two columns:
  - text  : the raw message string
  - label : 0 (safe) or 1 (bullying)

Large files are streamed in fixed-size chunks so a >32 MB CSV never has to
sit in memory all at once.
"""

import os
import yaml
import pandas as pd

# Tune down if you hit RAM pressure; tune up for faster loading on big RAM machines
CHUNK_SIZE = 50_000


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_labels(series: pd.Series, label_map: dict, dataset_name: str) -> pd.Series:
    """
    Map original labels to 0/1 using label_map.
    Raises a clear error if any value in the series is missing from the map.
    """
    sample_key = next(iter(label_map))
    if isinstance(sample_key, int):
        series = series.astype(int)
    else:
        series = series.astype(str).str.strip()

    unknown = set(series.unique()) - set(label_map.keys())
    if unknown:
        raise ValueError(
            f"[{dataset_name}] Unknown label values: {unknown}. "
            f"Add them to label_map in datasets.yaml."
        )

    return series.map(label_map)


def _load_excel(file_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Read an Excel file (.xlsx / .xls) into a DataFrame."""
    header = pd.read_excel(file_path, nrows=0)
    missing = [c for c in (text_col, label_col) if c not in header.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in '{os.path.basename(file_path)}'.\n"
            f"Available columns: {list(header.columns)}"
        )
    return pd.read_excel(file_path, usecols=[text_col, label_col])


def _load_csv_chunked(file_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Stream a CSV in CHUNK_SIZE-row batches and return a concatenated DataFrame.
    Validates column names on the header row before streaming to give a fast,
    actionable error instead of processing the whole file first.
    """
    # Read just the header (0 data rows) to check column names cheaply
    header = pd.read_csv(file_path, nrows=0)
    missing = [c for c in (text_col, label_col) if c not in header.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in '{os.path.basename(file_path)}'.\n"
            f"Available columns: {list(header.columns)}"
        )

    chunks = []
    for chunk in pd.read_csv(
        file_path,
        usecols=[text_col, label_col],
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",   # skip malformed rows rather than crashing
        engine="c",            # C parser is faster than python engine
        low_memory=False,      # avoids mixed-type inference warnings on large files
    ):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_all_datasets(config_path: str = "config/datasets.yaml") -> pd.DataFrame:
    """
    Main entry point. Reads the YAML config, streams each dataset in chunks,
    normalizes its labels, and concatenates everything into one DataFrame.

    Returns
    -------
    pd.DataFrame with columns: text (str), label (int: 0 or 1)
    """
    config = _load_config(config_path)
    frames = []

    for ds in config["datasets"]:
        name      = ds["name"]
        file_path = ds["file"]

        if not os.path.exists(file_path):
            print(f"  [SKIP] '{name}' — file not found: {file_path}")
            continue

        file_mb = os.path.getsize(file_path) / (1024 ** 2)
        print(f"  [LOAD] {name} ← {file_path}  ({file_mb:.1f} MB, chunk_size={CHUNK_SIZE:,})")

        if file_path.endswith((".xlsx", ".xls")):
            df = _load_excel(file_path, ds["text_column"], ds["label_column"])
        else:
            df = _load_csv_chunked(file_path, ds["text_column"], ds["label_column"])

        # Rename to the unified column names used everywhere downstream
        df = df.rename(columns={
            ds["text_column"]:  "text",
            ds["label_column"]: "label",
        })

        before  = len(df)
        df      = df.dropna(subset=["text", "label"])
        dropped = before - len(df)
        if dropped:
            print(f"         Dropped {dropped:,} rows with missing values.")

        df["label"] = _normalize_labels(df["label"], ds["label_map"], name)

        print(
            f"         {len(df):,} rows | "
            f"bullying={df['label'].sum():,} | "
            f"safe={(df['label'] == 0).sum():,}"
        )
        frames.append(df[["text", "label"]])

    if not frames:
        raise RuntimeError(
            "No datasets were loaded. Check that your CSV files exist in the "
            "'data/' folder and that 'config/datasets.yaml' points to them correctly."
        )

    combined = pd.concat(frames, ignore_index=True)

    before   = len(combined)
    combined = combined.drop_duplicates(subset=["text"])
    print(
        f"\n  Combined: {len(combined):,} unique rows "
        f"(removed {before - len(combined):,} duplicates)"
    )
    print(
        f"  Class balance → "
        f"bullying: {combined['label'].sum():,} | "
        f"safe: {(combined['label'] == 0).sum():,}\n"
    )

    return combined.reset_index(drop=True)
