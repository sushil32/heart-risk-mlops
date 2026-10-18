"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:  python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request

from heartrisk.data import CLEAN_PATH, RAW_PATH, UCI_URL, clean, load_raw


def download(force: bool = False) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force or not RAW_PATH.exists():
        print(f"Downloading {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, RAW_PATH)
    print(
        f"raw file  : {RAW_PATH}  sha256={hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()[:16]}"
    )

    raw = load_raw(RAW_PATH)
    df = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"raw rows  : {len(raw)}  missing cells: {int(raw.isna().sum().sum())}")
    print(f"clean rows: {len(df)}  -> {CLEAN_PATH}")
    print(f"class balance (target=1 means disease):\n{df['target'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(parser.parse_args().force)
