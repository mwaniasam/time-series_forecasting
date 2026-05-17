"""
Data ingestion pipeline for the Telecom Italia Milan dataset.

Reads all daily .txt files from the zip archives, retains only the three
relevant columns (square_id, timestamp_ms, internet), applies dtype
optimisations to minimise RAM usage, aggregates country-level duplicate rows,
and writes the result to a Parquet file for efficient downstream access.
"""

import gc
import glob
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

BASE = Path(__file__).parent
OUT_DIR = BASE / "processed"
OUT_DIR.mkdir(exist_ok=True)
PARQUET_PATH = OUT_DIR / "milan_internet_traffic.parquet"

COL_NAMES = [
    "square_id", "timestamp_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]

DTYPES = {
    "square_id":    "uint16",
    "timestamp_ms": "int64",
    "internet":     "float32",
}


def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def read_daily_file(zf_path: str, filename: str) -> pd.DataFrame:
    """
    Open a single daily file from within a zip archive and return a
    memory-optimised DataFrame with three columns: square_id, timestamp_ms,
    and internet.  Rows without an internet reading are dropped, and
    multiple CDR rows for the same (square, timestamp) pair — arising from
    separate country-code entries — are summed.
    """
    with zipfile.ZipFile(zf_path, "r") as zf:
        with zf.open(filename) as fh:
            df = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                names=COL_NAMES,
                usecols=["square_id", "timestamp_ms", "internet"],
                dtype=DTYPES,
                na_values=[""],
            )
    df.dropna(subset=["internet"], inplace=True)
    df = df.groupby(["square_id", "timestamp_ms"], as_index=False)["internet"].sum()
    df["square_id"] = df["square_id"].astype("uint16")
    df["internet"] = df["internet"].astype("float32")
    return df


def run():
    if PARQUET_PATH.exists():
        print(f"Parquet already exists at {PARQUET_PATH} — skipping.")
        return

    zip_files = sorted(glob.glob(str(BASE / "dataverse_files*.zip")))
    print(f"Found {len(zip_files)} archives")

    t0 = time.time()
    chunks = []
    file_counter = 0

    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path, "r") as zf:
            txt_names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
        for fname in txt_names:
            df_day = read_daily_file(zf_path, fname)
            chunks.append(df_day)
            file_counter += 1
            print(
                f"  [{file_counter:3d}/62] {fname}  "
                f"rows={len(df_day):>7,}  RAM={ram_mb():.0f} MB"
            )
            del df_day
            gc.collect()

    print(f"\nConcatenating {len(chunks)} daily frames ...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df.drop(columns=["timestamp_ms"], inplace=True)
    df.sort_values(["square_id", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow", compression="snappy")
    elapsed = time.time() - t0

    pq_mb = PARQUET_PATH.stat().st_size / 1024 ** 2
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Shape: {df.shape}")
    print(f"In-memory: {mem_mb:.1f} MB")
    print(f"On-disk (Parquet/Snappy): {pq_mb:.1f} MB")
    print(df.dtypes)


if __name__ == "__main__":
    run()
