"""
Timestamp standardisation and calibration pipeline for the three Kaggle datasets
used in the Keele University parking simulation.

Steps:
  1. Load each raw CSV
  2. Normalise timestamps to a uniform 15-minute interval
  3. Filter to Keele enforcement hours (08:00–18:00, weekdays)
  4. Encode occupancy as binary (0/1)
  5. Calibrate bay IDs to the ~2,000-bay Keele campus
  6. Merge all three and export to data/processed/unified_parking.csv

Usage:
  python src/preprocess.py
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

DATASETS = [
    {
        "name": "PKLot",
        "filename": "pklot_occupancy.csv",
        "ts_col": "LastUpdated",
        "occ_col": "Occupied",
        "ts_format": "custom",   # DD/MM/YYYY HH:MM
    },
    {
        "name": "CNRPark",
        "filename": "cnrpark_occupancy.csv",
        "ts_col": "timestamp",
        "occ_col": "occupancy",
        "ts_format": "iso",      # YYYY-MM-DDTHH:MM
    },
    {
        "name": "SmartParking",
        "filename": "smart_parking.csv",
        "ts_col": "updated_at",
        "occ_col": "lot_occupied",
        "ts_format": "unix",     # Unix epoch (seconds)
    },
]

ENFORCE_START = 8
ENFORCE_END   = 18
TARGET_BAYS   = 2000


def normalise_timestamps(df, ts_col, ts_format):
    df = df.copy()
    if ts_format == "unix":
        df["timestamp"] = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.tz_localize(None)
    elif ts_format == "iso":
        df["timestamp"] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
    elif ts_format == "custom":
        df["timestamp"] = pd.to_datetime(df[ts_col], format="%d/%m/%Y %H:%M", dayfirst=True)
    else:
        raise ValueError(f"Unknown timestamp format: {ts_format!r}")
    # Round down to 15-minute intervals
    df["timestamp"] = df["timestamp"].dt.floor("15min")
    return df


def filter_enforcement(df):
    mask = (
        (df["timestamp"].dt.weekday < 5) &                # weekdays only
        (df["timestamp"].dt.hour >= ENFORCE_START) &
        (df["timestamp"].dt.hour < ENFORCE_END)
    )
    result = df[mask].copy()
    log.info("  After enforcement filter: %d rows", len(result))
    return result


def encode_occupancy(df, occ_col):
    df = df.copy()
    df["occupancy"] = df[occ_col].astype(int)
    return df


def calibrate_bays(df):
    """Map whatever bay IDs the source uses onto integers 0..TARGET_BAYS-1."""
    df = df.copy()
    df["bay_id"] = np.arange(len(df)) % TARGET_BAYS
    return df


def process_dataset(cfg):
    path = os.path.join(RAW_DIR, cfg["filename"])
    if not os.path.exists(path):
        log.warning("Not found – skipping: %s", path)
        return pd.DataFrame()

    log.info("Loading %s …", cfg["name"])
    df = pd.read_csv(path, low_memory=False)
    log.info("  Raw rows: %d", len(df))

    df = normalise_timestamps(df, cfg["ts_col"], cfg["ts_format"])
    df = filter_enforcement(df)
    df = encode_occupancy(df, cfg["occ_col"])
    df = calibrate_bays(df)

    df = df[["timestamp", "bay_id", "occupancy"]].drop_duplicates()
    df["source"] = cfg["name"]
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    frames = [process_dataset(cfg) for cfg in DATASETS]
    frames = [f for f in frames if not f.empty]

    if not frames:
        log.error(
            "No raw CSVs found in %s.\n"
            "Download the three Kaggle datasets and place them there.\n"
            "See data/README.md for links and exact filenames.",
            RAW_DIR,
        )
        return

    unified = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    out = os.path.join(PROCESSED_DIR, "unified_parking.csv")
    unified.to_csv(out, index=False)
    log.info("Saved %d records → %s", len(unified), out)


if __name__ == "__main__":
    main()
