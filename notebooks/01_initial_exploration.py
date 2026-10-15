import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 - Initial data exploration
    """)
    return


@app.cell
def _():
    # load libraries
    import sys
    import marimo as mo
    from pathlib import Path

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import pandas as pd
    import paths

    snap = paths.snapshot("30 June, 2026")
    snap
    return mo, pd, snap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reviews
    """)
    return


@app.cell
def _(pd, snap):
    reviews_raw = pd.read_csv(snap.reviews, parse_dates=["date"])
    return (reviews_raw,)


@app.cell
def _(reviews_raw):
    reviews_raw.head()
    return


@app.cell
def _(reviews_raw):
    reviews_raw.info(memory_usage="deep")
    return


@app.cell
def _(reviews_raw):
    reviews_raw.describe()
    return


@app.cell
def _(reviews_raw):
    print(reviews_raw.isnull().sum())
    print((reviews_raw.isnull().sum() / len(reviews_raw) * 100).round(2).to_frame("Missing %"))
    return


@app.cell
def _(reviews_raw):
    for t in reviews_raw["comments"].dropna().sample(15, random_state=0):
        print(repr(t[:300]), "\n")
    return


@app.cell
def _(reviews_raw):
    # screening for HTML in the comments
    # How much HTML, really?
    has_html = reviews_raw["comments"].fillna("").str.contains("<", regex=False)
    print(f"{has_html.sum():,} of {len(reviews_raw):,} reviews contain '<'  ({has_html.mean():.1%})")
    return


@app.cell
def _(reviews_raw):
    # Which tags?
    import re
    TAG = re.compile(r"<[a-zA-Z/!][^>]{0,40}>")
    comments = reviews_raw["comments"].fillna("")

    loose  = comments.str.contains("<", regex=False)
    strict = comments.str.contains(TAG)

    print(f"contains '<'      : {loose.sum():>8,}  ({loose.mean():.1%})")
    print(f"contains a tag    : {strict.sum():>8,}  ({strict.mean():.1%})")
    print(f"'<' but no tag    : {(loose & ~strict).sum():>8,}")
    return TAG, comments, re, strict


@app.cell
def _(TAG, comments, strict):
    tags = comments[strict].str.findall(TAG).explode()
    print(tags.value_counts().head(10))
    return


@app.cell
def _(reviews_raw):
    # How long is a review, really?
    lengths = reviews_raw["comments"].fillna("").str.split().str.len()
    print(lengths.describe())
    print("\nmedian:", lengths.median(), "tokens")
    print("under 3 tokens:", (lengths < 3).sum())
    return


@app.cell
def _(comments):
    patterns = {
        "automated posting": r"automated posting",
        "canceled/cancelled reservation": r"cancell?ed (this|the) reservation",
        "the host canceled": r"host cancell?ed",
        "airbnb.com address": r"@airbnb\.com",
    }
    for label, pat in patterns.items():
        hits = comments.str.contains(pat, case=False, regex=True)
        print(f"{label:<32} {hits.sum():>7,}  ({hits.mean():.2%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listings
    """)
    return


@app.cell
def _(pd, snap):
    listings_raw = pd.read_csv(snap.listings)
    print(listings_raw.shape)
    return (listings_raw,)


@app.cell
def _(listings_raw):
    print("rows:              ", len(listings_raw))
    print("unique ids:        ", listings_raw["id"].nunique())
    print("unique host_ids:   ", listings_raw["host_id"].nunique())
    return


@app.cell
def _(listings_raw):
    missing = (listings_raw.isnull().sum() / len(listings_raw) * 100).round(1)
    missing = missing.sort_values(ascending=False)
    print(missing.head(25).to_frame("Missing %"))
    return (missing,)


@app.cell
def _(missing):
    # 1. columns with no information at all
    empty = missing[missing == 100]
    print(f"{len(empty)} columns are 100% empty:\n")
    print("\n".join(empty.index))
    return


@app.cell
def _(missing):
    # 2. partially missing columns, where the missingness itself says something
    partial = missing[(missing > 0) & (missing < 100)]
    print(f"{len(partial)} columns partially missing:\n")
    print(partial.to_string())
    return


@app.cell
def _(listings_raw):
    no_reviews = listings_raw["number_of_reviews"] == 0
    print(f"listings with 0 reviews: {no_reviews.sum():,}  ({no_reviews.mean():.1%})")
    print(f"review_scores_rating missing: {listings_raw['review_scores_rating'].isna().sum():,}")

    # do the two sets coincide exactly?
    print("identical sets:", (no_reviews == listings_raw['review_scores_rating'].isna()).all())
    return


@app.cell
def _(listings_raw):
    text_cols = ["name", "description", "neighborhood_overview", "host_about", "amenities"]

    for c in text_cols:
        s = listings_raw[c].fillna("")
        nonblank = s[s.str.strip() != ""]          # the rows that actually carry text
        share = len(nonblank) / len(s)

        if len(nonblank) == 0:
            median_chars = "—"                      # nothing to take a median of
        else:
            median_chars = f"{int(nonblank.str.len().median()):,}"

        print(f"{c:<24} {len(nonblank):>7,} non-blank  ({share:>5.1%})   median {median_chars:>6} chars")
    return


@app.cell
def _(listings_raw):
    for d in listings_raw["description"].dropna().sample(15, random_state=0):
        print(repr(d[:300]), "\n")
    return


@app.cell
def _(listings_raw):
    print(listings_raw["source"].value_counts(dropna=False))
    return


@app.cell
def _(listings_raw, re):
    TAG2   = re.compile(r"<[a-zA-Z/!][^>]{0,40}>")
    BR    = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
    ZWSP  = re.compile("[\u200b\u200c\u200d\ufeff]")       # zero-width space/NJ/J, BOM
    EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    desc = listings_raw["description"].fillna("")
    n = len(desc)
    return EMOJI, TAG2, ZWSP, desc


@app.cell
def _(TAG2, desc):
    _loose  = desc.str.contains("<", regex=False)
    _strict = desc.str.contains(TAG2)
    print(f"contains '<'   : {_loose.sum():>7,}  ({_loose.mean():.1%})")
    print(f"contains a tag : {_strict.sum():>7,}  ({_strict.mean():.1%})")
    print(f"'<' but no tag : {(_loose & ~_strict).sum():>7,}")
    print()
    print(desc[_strict].str.findall(TAG2).explode().value_counts().head(10))
    return


@app.cell
def _(desc, re):
    WIDE = re.compile(r"<[a-zA-Z/!][^>]{0,300}>")
    found = desc.str.findall(WIDE).explode().dropna()
    print(found.str.slice(0, 60).value_counts().head(15))
    print()
    print(f"total tag occurrences: {len(found):,}")
    return


@app.cell
def _(desc):
    for _t in desc[desc.str.contains("<a ", regex=False)].head(3):
        print(repr(_t[:400]), "\n")
    return


@app.cell
def _(EMOJI, ZWSP, desc):
    zw = desc.str.contains(ZWSP)
    print(f"descriptions with zero-width chars: {zw.sum():,}  ({zw.mean():.1%})")

    # How many characters would stripping them remove, in total?
    before = desc.str.len().sum()
    after  = desc.str.replace(ZWSP, "", regex=True).str.len().sum()
    print(f"invisible characters in the corpus: {before - after:,}")

    em = desc.str.contains(EMOJI)
    print(f"descriptions containing emoji:      {em.sum():,}  ({em.mean():.1%})")
    return


@app.cell
def _(listings_raw, pd):
    rows = []
    for _label, grp in listings_raw.groupby("source"):
        p = grp["description"].fillna("")
        nb = p[p.str.strip() != ""]
        rows.append({
            "source": _label,
            "listings": len(grp),
            "has description": f"{len(nb)/len(grp):.1%}",
            "median chars": int(nb.str.len().median()),
            "mean chars": int(nb.str.len().mean()),
            "median words": int(nb.str.split().str.len().median()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    return


if __name__ == "__main__":
    app.run()
