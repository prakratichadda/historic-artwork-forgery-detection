"""
download_subset.py
-------------------
Downloads a small subset of source images directly from the wga.hu URLs
listed in data/mapped.csv, instead of pulling the full ~6.9GB Kaggle
archive. Saves images to data/historic-art/complete/artwork/{ID}.jpg,
matching the path prediction/combined.py expects.

Usage:
    python utils/download_subset.py [N]

N defaults to 1000 (the first N rows of mapped.csv).
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CSV_PATH = "data/mapped.csv"
OUT_DIR = os.path.join("data", "historic-art", "complete", "artwork")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; historic-art-demo/1.0)"}

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).head(N)


def download_one(image_id, url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return "ok"
    except Exception as e:
        return f"error: {e}"


tasks = []
for _, row in df.iterrows():
    dest = os.path.join(OUT_DIR, f"{row['ID']}.jpg")
    tasks.append((row["ID"], row["jpg url"], dest))

ok = skipped = failed = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(download_one, i, u, d): i for i, u, d in tasks}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        result = fut.result()
        if result == "ok":
            ok += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

print(f"\nDone. ok={ok} skipped={skipped} failed={failed} target_dir={OUT_DIR}")
