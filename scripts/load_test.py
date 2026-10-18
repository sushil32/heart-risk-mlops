"""Send randomised patient records to the API to generate traffic for monitoring demos.

Usage:  python scripts/load_test.py --url http://localhost:8000 --n 300 --invalid-rate 0.05
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import requests

from heartrisk.data import CLEAN_PATH, FEATURES, load_clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    p.add_argument("--invalid-rate", type=float, default=0.05, help="share of bad payloads")
    args = p.parse_args()

    rows = load_clean(CLEAN_PATH)[FEATURES].to_dict("records")
    stats: Counter = Counter()
    for _ in range(args.n):
        rec = {k: (None if v != v else v) for k, v in random.choice(rows).items()}  # NaN -> None
        if random.random() < args.invalid_rate:
            rec["age"] = 7  # out of range -> 422
        r = requests.post(f"{args.url}/predict", json=rec, timeout=5)
        stats[r.status_code] += 1
        if r.ok:
            stats[r.json()["label"]] += 1
        time.sleep(args.delay)
    print(dict(stats))


if __name__ == "__main__":
    main()
