"""
Download the four Inside Airbnb files for Copenhagen, plus the fastText
language-identification model.

    uv run scripts/download_data.py --snapshot 2026-06-30

The snapshot date is an argument because Inside Airbnb publishes one snapshot
per quarter, the URL contains its date, and only the last 12 months are free.
Find the date at https://insideairbnb.com/get-the-data/ (Copenhagen,
Hovedstaden, Denmark). A 404 means the date is wrong.

Downloads:

    reviews.csv.gz          the corpus: listing_id, id, date, reviewer_id, comments
    listings.csv.gz         coordinates, neighbourhood and host fields per listing
    neighbourhoods.geojson  official area polygons
    neighbourhoods.csv      the neighbourhood list
    lid.176.bin             fastText language ID, 176 languages (126 MB)

Files go to `data/raw/<snapshot label>/` (e.g. `data/raw/30 June, 2026/`); the
model goes to `models/`. Both are git-ignored: raw review text must not be
redistributed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# make `paths` importable whether this is run as a script or imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW, MODELS, ensure_dirs, snapshot_name  # noqa: E402

BASE = "https://data.insideairbnb.com/denmark/hovedstaden/copenhagen"

# (url suffix, destination filename)
AIRBNB_FILES = [
    ("data/reviews.csv.gz", "reviews.csv.gz"),
    ("data/listings.csv.gz", "listings.csv.gz"),
    ("visualisations/neighbourhoods.geojson", "neighbourhoods.geojson"),
    ("visualisations/neighbourhoods.csv", "neighbourhoods.csv"),
]

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Inside Airbnb serves 403 to requests without a browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (research download; COS-R434 coursework)"}


def download(url: str, dest: Path, force: bool = False) -> None:
    """
    Fetch one file to `dest`, via a `.part` file renamed only on success, so an
    interrupted run never leaves a half-file behind.
    """
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"  skip   {dest.name}  (already here, {size_mb:.1f} MB)")
        return

    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  get    {dest.name}  <- {url}")
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request) as response, open(part, "wb") as out:
            while chunk := response.read(1 << 20):  # 1 MB at a time
                out.write(chunk)
    except urllib.error.HTTPError as error:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"\nHTTP {error.code} for {url}\n"
            "If this is a 404, the snapshot date is wrong or that snapshot is no\n"
            "longer free. Check https://insideairbnb.com/get-the-data/ and pass\n"
            "the date you find there with --snapshot YYYY-MM-DD."
        ) from error

    part.rename(dest)
    print(f"         done, {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot date as it appears in the Inside Airbnb URL, e.g. 2026-06-30",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that are already present",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Do not fetch the 126 MB fastText language-ID model",
    )
    args = parser.parse_args()

    ensure_dirs()

    # folder named the way Inside Airbnb labels the snapshot
    dest_dir = RAW / snapshot_name(args.snapshot)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inside Airbnb — Copenhagen, snapshot {args.snapshot} -> {dest_dir.name}/")
    for suffix, filename in AIRBNB_FILES:
        download(f"{BASE}/{args.snapshot}/{suffix}", dest_dir / filename, args.force)

    if not args.skip_model:
        print("\nfastText language identification model")
        download(LID_URL, MODELS / "lid.176.bin", args.force)

    print(f"\nAll files in {dest_dir}")


if __name__ == "__main__":
    main()
