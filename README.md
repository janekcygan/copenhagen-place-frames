# copenhagen-frames

How Copenhagen's neighbourhoods are described on Airbnb, in three voices:

| voice | source | |
|---|---|---|
| `host/en` | listing descriptions | marketing, written to sell a stay |
| `guest/en` | English reviews | experience, written after one |
| `guest/da` | Danish reviews | the same, in Danish |

Course project for COS-R434 (NLP for Social Science), University of Helsinki,
autumn 2026. The corpus is treated as **platform-mediated place description**.

The project is built together with Claude Code.

## Research questions

- **RQ1.** What descriptive frames characterise each Copenhagen neighbourhood in platform
  text, and how do those profiles differ between the marketing voice and the experience voice?
- **RQ2.** Do guests writing in Danish and guests writing in English describe the same areas
  using different frames?
- **RQ3.** Where does the marketing profile diverge most from the experience profile, and is
  that divergence spatially patterned?
- **RQ4.** Does a frame classifier trained on English guest text transfer to Danish guest
  text through a multilingual sentence encoder, and where does it fail?

The unit is the **area × voice** cell: eleven official areas, three voices. Work in progress:
the repository currently covers the corpus, the place filter and topic discovery. The
codebook and classification steps are not here yet.

## Data

All data comes from [Inside Airbnb](https://insideairbnb.com/get-the-data/) (Copenhagen,
Hovedstaden, Denmark). **None of it is in this repository.** Review text is written by
identifiable people and must not be redistributed, so `data/` is git-ignored and every
notebook writes text only to `data/interim/` on your own machine.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                       # core stack: notebooks 01–05 and 08
uv sync --group modelling     # adds BERTopic + PyTorch (several GB): notebooks 06–07
```

Download the data. Notebooks 01–07 use the 30 June 2026 snapshot; 08 also reads the
listings of the older snapshots, to fill in missing prices:

```bash
uv run scripts/download_data.py --snapshot 2026-06-30
uv run scripts/download_data.py --snapshot 2026-03-30 --skip-model
uv run scripts/download_data.py --snapshot 2025-12-30 --skip-model
uv run scripts/download_data.py --snapshot 2025-09-29 --skip-model
```

Only the last 12 months of snapshots are free, so the oldest one may no longer be available.

## Notebooks

[marimo](https://marimo.io) notebooks, numbered in run order. Open one with
`uv run marimo edit notebooks/<file>.py`, or run it as a script with
`uv run python notebooks/<file>.py`.

| # | notebook | what it does | needs | writes |
|---|---|---|---|---|
| 01 | `initial_exploration` | first look at reviews and listings | raw data | — |
| 02 | `data_preperation` | clean, join, language-ID, split into sentences | raw data | `interim/sentences_voices_*.parquet` |
| 03 | `place_filter` | build the place filter (King, Lam & Roberts keyword discovery) | 02 | — (the lexicon lives in `scripts/place_filter.py`) |
| 04 | `exp1_vocabulary` | experiment: place vocabulary per voice | 05 | count tables, figure |
| 05 | `corpora` | apply the filter, remove platform text and duplicates, balanced sample | 02 | `interim/corpus_*.parquet` |
| 06 | `exp2_BERTopic_test` | experiment: BERTopic defaults vs tuned settings | 05, 07 | count tables, figure |
| 07 | `discovery` | BERTopic on a 24,000-sentence pool (8,000 per voice) | 05 | topic table, marking sheet, reading file |
| 08 | `exp3_topics_distributions` | experiment: where the kept topics concentrate (voice, area, room, price, map) | 05, 07 | count tables, figures, maps |

The `exp` notebooks are side experiments; nothing in the pipeline reads their output. They are
numbered by when they were done, not by what they depend on: **04 needs 05, and 06 needs 07.**
The pipeline itself is 01 → 02 → 03 → 05 → 07.

Two notes:

- **03** takes about 8 minutes on its first run (the classifier results are cached afterwards).
  Its last cell checks that `scripts/place_filter.py` and the notebook keep exactly the same
  sentences.
- **08** needs the topic marking: one row per topic from 07, with a hand-filled `frameless`
  column (`yes`/`no`) and notes. Mine is included as
  `data/processed/discovery_marks_2026-06-30.csv`. 07 never overwrites it, and warns if a
  re-run produces different topics.

## Layout

```
notebooks/   the marimo notebooks above
scripts/     paths.py (all paths), download_data.py, place_filter.py (the shipped filter)
data/        raw/ interim/ processed/ — empty here, filled locally
figures/     written by the notebooks
docs/        research log
```
