"""Acquire the UCI Online Retail II file into data/raw/, unchanged.

    python scripts/download_data.py            # download, or reuse a verified local copy
    python scripts/download_data.py --from PATH # copy a file you already have

If the download is not possible, place either `online_retail_ii.zip` (as distributed by UCI) or
`online_retail_II.xlsx` in data/raw/ and rerun; the script then only verifies it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qstats_planner.utils.config import load_config, resolve  # noqa: E402

MANUAL = """
Automatic download is unavailable. To continue:
  1. Open https://archive.ics.uci.edu/dataset/502/online+retail+ii
  2. Download the zip (or the xlsx inside it).
  3. Place it in {raw_dir}/ as online_retail_ii.zip or online_retail_II.xlsx.
  4. Rerun: python scripts/download_data.py
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, attempts: int = 3) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(target.with_suffix(".part"), "wb") as fh:
                shutil.copyfileobj(resp, fh)
            target.with_suffix(".part").rename(target)
            return True
        except Exception as exc:  # network errors of every kind end in the manual route
            print(f"  attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(2 * attempt)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", type=Path, help="copy an existing local file")
    args = parser.parse_args()

    cfg = load_config()
    src = cfg["source"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path, xlsx_path = raw_dir / src["zip_name"], raw_dir / src["xlsx_name"]

    if args.source:
        target = zip_path if args.source.suffix == ".zip" else xlsx_path
        shutil.copy2(args.source, target)
        print(f"copied {args.source} -> {target}")

    if not zip_path.exists() and not xlsx_path.exists():
        print(f"downloading {src['url']}")
        if not download(src["url"], zip_path):
            print(MANUAL.format(raw_dir=raw_dir))
            return 2

    if zip_path.exists():
        digest = sha256(zip_path)
        if digest != src["sha256"]:
            print(f"checksum mismatch for {zip_path}: {digest} (expected {src['sha256']})")
            return 1
        print(f"ok  {zip_path.name}  sha256 {digest[:16]}...")
    else:
        print(f"ok  {xlsx_path.name} (manual copy; no published checksum for the loose xlsx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
