"""
Every path in the project, in one place. Notebooks and scripts import from here
instead of hardcoding paths:

    from paths import RAW, INTERIM, snapshot

`ROOT` is worked out from this file's location, so it is correct whichever
directory a notebook is launched from.

Inside Airbnb publishes one snapshot per quarter. `data/raw/` holds one folder
per snapshot, named as on the download page ("30 June, 2026"). The analysis
always names its snapshot explicitly.

Use the DETAILED files, the `.csv.gz` ones:

    reviews.csv.gz    listing_id, id, date, reviewer_id, comments   <- the corpus
    reviews.csv       listing_id, date only (no text)
    listings.csv.gz   ~75 columns including coordinates              <- the join
    listings.csv      16-column summary
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# this file is <project>/scripts/paths.py
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"              # one folder per snapshot. Never edited, never committed.
INTERIM = DATA / "interim"      # per-review / per-sentence working files (parquet)
PROCESSED = DATA / "processed"  # aggregates only
MODELS = ROOT / "models"        # fastText lid.176 lives here
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"

LID_MODEL = MODELS / "lid.176.bin"

# Inside Airbnb's own label format, like "30 June, 2026".
SNAPSHOT_FORMAT = "%d %B, %Y"


def snapshot_name(iso_date: str) -> str:
    """Turns '2026-06-30' into '30 June, 2026', the folder name used for that snapshot."""
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime(SNAPSHOT_FORMAT).lstrip("0")


def available_snapshots() -> list[str]:
    """Snapshot folder names present in data/raw/, oldest first."""
    if not RAW.exists():
        return []
    dated = []
    for path in RAW.iterdir():
        if not path.is_dir():
            continue
        try:
            dated.append((datetime.strptime(path.name, SNAPSHOT_FORMAT), path.name))
        except ValueError:
            # not a snapshot folder
            continue
    return [name for _, name in sorted(dated)]


def latest_snapshot() -> str | None:
    """The most recent snapshot on disk, or None if nothing is downloaded."""
    names = available_snapshots()
    return names[-1] if names else None


class Snapshot:
    """
    One dated download, holding its file paths.

        snap = snapshot()                  # the latest one on disk
        snap = snapshot("30 June, 2026")   # a specific one
        pd.read_csv(snap.reviews)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = RAW / name
        self.date = datetime.strptime(name, SNAPSHOT_FORMAT).date()

        # the detailed files, with review text and coordinates
        self.reviews = self.dir / "reviews.csv.gz"
        self.listings = self.dir / "listings.csv.gz"
        self.neighbourhoods_geojson = self.dir / "neighbourhoods.geojson"
        self.neighbourhoods_csv = self.dir / "neighbourhoods.csv"
        self.calendar = self.dir / "calendar.csv.gz"  # not used

    @property
    def slug(self) -> str:
        """'2026-06-30', for the filenames of anything derived from this snapshot."""
        return self.date.isoformat()

    def exists(self) -> bool:
        return all(
            p.exists()
            for p in (self.reviews, self.listings, self.neighbourhoods_geojson)
        )

    def missing(self) -> list[Path]:
        return [
            p
            for p in (self.reviews, self.listings, self.neighbourhoods_geojson,
                      self.neighbourhoods_csv)
            if not p.exists()
        ]

    def __repr__(self) -> str:
        return f"Snapshot({self.name!r})"


def snapshot(name: str | None = None) -> Snapshot:
    """A Snapshot by name, or the latest one on disk if no name is given."""
    if name is None:
        name = latest_snapshot()
        if name is None:
            raise FileNotFoundError(
                f"No snapshot folders in {RAW}. Run:\n"
                "  uv run scripts/download_data.py --snapshot YYYY-MM-DD"
            )
    return Snapshot(name)


# --- derived files ---------------------------------------------------------
# Named for the snapshot they came from, so files from two snapshots never mix.

def interim(kind: str, snap: Snapshot) -> Path:
    """For example, interim('sentences', snap) gives data/interim/sentences_2026-06-30.parquet"""
    return INTERIM / f"{kind}_{snap.slug}.parquet"


def processed(kind: str, snap: Snapshot) -> Path:
    return PROCESSED / f"{kind}_{snap.slug}.parquet"


def ensure_dirs() -> None:
    """Create every directory the project writes to. Safe to call repeatedly."""
    for d in (RAW, INTERIM, PROCESSED, MODELS, FIGURES, REPORT):
        d.mkdir(parents=True, exist_ok=True)
