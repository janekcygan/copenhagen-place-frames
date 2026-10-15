import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths
    import place_filter as pf

    snap = paths.snapshot("30 June, 2026")

    CORPUS_PATH = paths.interim("sentences_voices", snap)   # 02_data_preperation.py's output
    OUT_PATH = paths.interim("corpus", snap)                # this notebook's output

    # one seed for every random draw
    SEED = 434

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} is missing — run notebooks/02_data_preperation.py first")

    print(f"snapshot: {snap}")
    print(f"in:  {CORPUS_PATH.name}")
    print(f"out: {OUT_PATH.name}")
    return CORPUS_PATH, OUT_PATH, SEED, mo, np, paths, pd, pf, snap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 Building the three corpora

    This notebook applies the place filter (`scripts/place_filter.py`) to every sentence,
    removes Airbnb's own text and copied host text, and draws a balanced sample of English
    reviews. It writes one file, `data/interim/corpus_<slug>.parquet`, which is the sentence
    table from 02 with some columns added.

    `arm` is `host/en`, `guest/en`, `guest/da`, or empty for other languages and Danish host
    text. `is_place` is the filter's verdict, computed for every row. `is_platform_text` marks
    Airbnb's own text in the host descriptions. `is_duplicate` marks a host sentence that already
    appeared earlier, in any listing. `in_corpus` combines these: in an arm, about place, not
    platform text and not a duplicate. These rows are the three corpora. `in_balanced_sample`
    marks the rows used for comparisons, and `sample_weight` says how many corpus sentences each
    sampled row stands for.

    Keeping these as columns instead of saving a filtered file means later steps can switch
    between filtered and unfiltered data without rebuilding anything.
    """)
    return


@app.cell
def _(CORPUS_PATH, pd):
    # all columns: the output is the full table plus new columns
    sentences = pd.read_parquet(CORPUS_PATH)

    # The three arms, as in 03. Danish host copy is not an arm.
    ARM_LANG = {"host/en": ("host", "en"), "guest/en": ("guest", "en"), "guest/da": ("guest", "da")}

    # a lookup, so a row can never be in two arms; anything else gets None
    _lookup = {(v, l): name for name, (v, l) in ARM_LANG.items()}
    sentences["arm"] = [_lookup.get(pair) for pair in zip(sentences["voice"], sentences["lang"])]

    print(f"{len(sentences):,} sentences, {sentences.shape[1]} columns")
    print(sentences["arm"].value_counts(dropna=False).to_string())

    # area is the unit of comparison, so check every in-arm sentence has one
    _no_area = sentences["arm"].notna() & sentences["neighbourhood"].isna()
    print(f"\nin-arm sentences with no neighbourhood: {int(_no_area.sum()):,}")
    return (sentences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: apply the filter to everything

    `pf.is_place` is applied to all rows, including reviews in other languages, where the
    result means nothing. Which rows are studied is decided by `arm`, so `is_place` needs no
    special cases.

    The check below counts the kept sentences again and compares them with
    `03_place_filter.py`. A different number means the script, the corpus or the arms have
    changed.
    """)
    return


@app.cell
def _(pf, sentences):
    sentences["is_place"] = pf.is_place(sentences["text"])

    # counts from 03_place_filter.py's step 2
    _EXPECTED_KEPT = {"host/en": 55_346, "guest/en": 346_692, "guest/da": 16_658}

    print(f"{'arm':10} {'sentences':>11} {'is_place':>10} {'retained':>9}   expected")
    _moved = []
    for _arm, _expected in _EXPECTED_KEPT.items():
        _rows = sentences["arm"] == _arm
        _kept = int(sentences.loc[_rows, "is_place"].sum())
        _flag = "ok" if _kept == _expected else "CHANGED"
        if _kept != _expected:
            _moved.append((_arm, _kept, _expected))
        print(f"{_arm:10} {int(_rows.sum()):>11,} {_kept:>10,} "
              f"{_kept / _rows.sum():>9.2%}   {_expected:>7,} {_flag}")

    if _moved:
        raise AssertionError(f"filter output differs from 03's: {_moved}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Airbnb's text in the host descriptions

    Two kinds of platform text are removed. The first is the template flag from 02: 404
    sentences from descriptions that are entirely an Airbnb placeholder. The second is eight
    help-centre and legal phrases, 632 sentences, found by listing host sentences that appear
    under at least 10 different hosts. 308 of them pass the place filter, because "Help Center"
    contains `center` and "Central Business Register" contains `central`.

    The tagline family ("...at this centrally-located place") is kept, as part of how the
    platform shapes the marketing text. Step 3 reduces each version to one copy.

    The phrases are named one by one instead of using a threshold on hosts per sentence,
    because hosts also write things like "welcome to our home!" independently of each other.
    """)
    return


@app.cell
def _(sentences):
    # Matched on lowercased text with ’ turned into ', so "don’t" matches "don't".
    # Plain substring matching is fine for these long ASCII phrases.
    HELP_CENTRE_PHRASES = [
        "visit our help center",                           # "For (more legal) information, visit our Help Center."
        "consumer protection laws don't",                  # consumer-protection notice, two versions
        "customer support",                                # only ever inside that notice (checked: 0 other rows)
        "always use airbnb to send money",                 # fraud warning
        "business registry:",                              # "Business registry: The Central Business Register"
        "airbnb isn't",                                    # "Airbnb isn't your host."
        "it provides the platform for guests and hosts",   # platform self-description
        "support team is available 24/7",                  # support notice
    ]

    _is_host = sentences["voice"] == "host"
    _norm = sentences["text"].fillna("").str.lower().str.replace("’", "'", regex=False)

    _help = sentences["text"].notna() & False   # an all-False Series with the right index
    for _phrase in HELP_CENTRE_PHRASES:
        _help = _help | _norm.str.contains(_phrase, regex=False)
    _help = _help & _is_host   # host text only

    sentences["is_platform_text"] = sentences["is_platform_template"] | _help

    _host_en = sentences["arm"] == "host/en"
    print(f"host/en sentences:             {int(_host_en.sum()):>7,}")
    print(f"  02's template flag:          {int((sentences.is_platform_template & _host_en).sum()):>7,}")
    print(f"  help-centre / legal lines:   {int((_help & _host_en).sum()):>7,}")
    print(f"  either (is_platform_text):   {int((sentences.is_platform_text & _host_en).sum()):>7,}")
    print(f"  ... of which the filter kept:{int((sentences.is_platform_text & _host_en & sentences.is_place).sum()):>7,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: copied host text

    A host sentence that already appeared word for word in another listing is dropped, whether
    or not it is the same host. Otherwise a host who pastes one description into ten listings
    would be counted ten times.

    Sentences are compared after normalising case, spaces and apostrophes. The copy that
    survives is always the same one, the first by `listing_id` and then by position, so re-runs
    give the same result. Platform text is set aside first, so a tagline survives once instead
    of disappearing completely.

    Reviews are not deduplicated: two reviews that say "Great location!" come from two people.
    """)
    return


@app.cell
def _(sentences):
    _candidates = (sentences["arm"] == "host/en") & ~sentences["is_platform_text"]

    # Sort into survival order, then mark every occurrence after the first. The index
    # still points at the original rows.
    _hosts = sentences.loc[_candidates, ["listing_id", "sent_pos", "text"]].sort_values(
        ["listing_id", "sent_pos"]
    )
    _norm = (
        _hosts["text"].str.lower()
        .str.replace("’", "'", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    _dup = _norm.duplicated(keep="first")

    sentences["is_duplicate"] = False
    sentences.loc[_dup.index[_dup], "is_duplicate"] = True

    _n_dup = int(sentences["is_duplicate"].sum())
    print(f"host/en sentences checked:  {int(_candidates.sum()):>7,}")
    print(f"duplicates dropped:         {_n_dup:>7,} ({_n_dup / _candidates.sum():.1%})")
    print(f"... of which place-describing: "
          f"{int((sentences.is_duplicate & sentences.is_place).sum()):,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: the three corpora

    A sentence is in the corpus if it is in an arm, about place, not platform text and not a
    duplicate. The table shows what each step removed per voice and area, to catch a step that
    hits one area much harder than the others.
    """)
    return


@app.cell
def _(pd, sentences):
    sentences["in_corpus"] = (
        sentences["arm"].notna()
        & sentences["is_place"]
        & ~sentences["is_platform_text"]
        & ~sentences["is_duplicate"]
    )

    # one row per arm x area
    _rows = []
    for (_arm, _area), _g in sentences[sentences["arm"].notna()].groupby(["arm", "neighbourhood"]):
        _rows.append({
            "arm": _arm,
            "area": _area,
            "sentences": len(_g),
            "place": int(_g["is_place"].sum()),
            "minus platform": int((_g["is_place"] & _g["is_platform_text"]).sum()),
            "minus duplicate": int((_g["is_place"] & _g["is_duplicate"]).sum()),
            "in corpus": int(_g["in_corpus"].sum()),
        })
    corpus_by_area = pd.DataFrame(_rows)
    corpus_by_area["retained %"] = (100 * corpus_by_area["in corpus"] / corpus_by_area["sentences"]).round(2)

    _totals = corpus_by_area.groupby("arm")[["sentences", "place", "minus platform",
                                            "minus duplicate", "in corpus"]].sum()
    _totals["retained %"] = (100 * _totals["in corpus"] / _totals["sentences"]).round(2)
    print(_totals.to_string())

    corpus_by_area.pivot(index="area", columns="arm", values="in corpus")
    return (corpus_by_area,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: the balanced English sample

    All host sentences and all Danish sentences in the corpus are used. Danish is the smallest
    voice and is never sampled. English review sentences are sampled within each area down to
    the Danish count in that area, so the English and Danish comparison is balanced from the
    start. Past about 4,000 sentences, the uncertainty is dominated by listing clustering and
    classifier error anyway.

    Comparisons use the balanced sample. A statement about all English reviews has to weight
    each sampled English row by `sample_weight`, which is the number of English corpus sentences
    in the area divided by the number sampled there. Host and Danish rows have a weight of 1.
    Sampling is by sentence, with a fixed seed and without replacement.
    """)
    return


@app.cell
def _(SEED, np, pd, sentences):
    sentences["in_balanced_sample"] = False
    sentences["sample_weight"] = np.nan   # NaN means not in the balanced sample

    # host and Danish: everything, weight 1
    _whole = sentences["in_corpus"] & sentences["arm"].isin(["host/en", "guest/da"])
    sentences.loc[_whole, "in_balanced_sample"] = True
    sentences.loc[_whole, "sample_weight"] = 1.0

    # English: per area, as many as Danish has there
    _da_per_area = sentences[sentences["in_corpus"] & (sentences["arm"] == "guest/da")] \
        .groupby("neighbourhood").size()

    _rows = []
    for _area, _n_da in _da_per_area.items():
        _pool = sentences.index[
            sentences["in_corpus"] & (sentences["arm"] == "guest/en")
            & (sentences["neighbourhood"] == _area)
        ]
        # in case English is ever the smaller side in an area
        _n = min(_n_da, len(_pool))
        # a separate generator per area, so one area's draw does not affect another's
        _rng = np.random.default_rng([SEED, len(_rows)])
        _picked = _rng.choice(_pool, size=_n, replace=False)
        sentences.loc[_picked, "in_balanced_sample"] = True
        sentences.loc[_picked, "sample_weight"] = len(_pool) / _n
        _rows.append({"area": _area, "guest/da": int(_n_da), "guest/en in corpus": len(_pool),
                      "guest/en sampled": _n, "weight": round(len(_pool) / _n, 2)})

    balance_table = pd.DataFrame(_rows).set_index("area")

    # English sampled == Danish, area by area
    _off = balance_table[balance_table["guest/en sampled"] != balance_table["guest/da"]]
    if len(_off):
        raise AssertionError(f"English sample does not match Danish in: {list(_off.index)}")

    print("balanced sample, by arm:")
    print(sentences[sentences["in_balanced_sample"]].groupby("arm").size().to_string())
    balance_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: sample for the frame-less check

    Frame prevalence will be a share of the kept sentences. If one voice has more kept sentences
    that say nothing about the area, its frames will look smaller, and that would look like a
    difference between the voices. The check is to mark a sample from each voice on whether the
    sentence says anything about the area, and compare the rates. This cell only draws the
    sample.

    It takes 100 random corpus sentences per voice, plus up to 30 per voice that were only kept
    because of `city` or `byen`, since "explore the city" is a known case. The second group is
    reported separately and never mixed into the random one.

    The marking is blind. The sheet (`frameless_check_<slug>.csv`) holds only an id and the
    sentence, shuffled, and the key (`frameless_key_<slug>.csv`) links the ids back. Both contain
    review text and stay in `data/interim/`.
    """)
    return


@app.cell
def _(SEED, paths, pd, pf, sentences, snap):
    _N_RANDOM, _N_CITY = 100, 30
    _CITY_WORDS = {"city", "byen"}

    # sentences kept only by city/byen: kept by the full filter, but not when those two words
    # are removed (and no toponym)
    _reduced_ref = [t for t in pf.REFERENCE_WORDS if t not in _CITY_WORDS]
    _reduced_ovr = [t for t in pf.VETO_OVERRIDES if t not in _CITY_WORDS]

    _parts = []
    for _i, _arm in enumerate(["host/en", "guest/en", "guest/da"]):
        _in = sentences[sentences["in_corpus"] & (sentences["arm"] == _arm)]
        _random = _in.sample(_N_RANDOM, random_state=SEED + _i)

        _without_city = pf.place_filter(_in["text"], reference=_reduced_ref,
                                        overrides=_reduced_ovr) | pf.toponym_hit(_in["text"])
        _city_only = _in[~_without_city].drop(_random.index, errors="ignore")
        _city = _city_only.sample(min(_N_CITY, len(_city_only)), random_state=SEED + _i)

        print(f"{_arm:9} random {len(_random)}   city-only {len(_city)} "
              f"(of {len(_city_only):,} such sentences, {len(_city_only) / len(_in):.1%} of the corpus)")
        _parts.append(_random.assign(stratum="random"))
        _parts.append(_city.assign(stratum="city-only"))

    _sample = pd.concat(_parts).sample(frac=1, random_state=SEED)   # shuffle: blind
    _sample["check_id"] = range(1, len(_sample) + 1)

    FRAMELESS_SHEET = paths.INTERIM / f"frameless_check_{snap.slug}.csv"
    FRAMELESS_KEY = paths.INTERIM / f"frameless_key_{snap.slug}.csv"

    # never overwrite a sheet that may already hold marks
    if FRAMELESS_SHEET.exists():
        print(f"\n{FRAMELESS_SHEET.name} exists — left untouched (it may hold marks).")
    else:
        _sample[["check_id", "text"]].assign(says_about_area="").to_csv(FRAMELESS_SHEET, index=False)
        _sample[["check_id", "sentence_id", "arm", "stratum", "neighbourhood"]].to_csv(
            FRAMELESS_KEY, index=False)
        print(f"\nwrote {FRAMELESS_SHEET.name} ({len(_sample)} rows) and {FRAMELESS_KEY.name}")
        print("mark `says_about_area` with 1 (yes) or 0 (no).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: write

    The corpus file goes to `data/interim/`, since it contains text. The table of counts per
    voice and area goes to `data/processed/`.
    """)
    return


@app.cell
def _(OUT_PATH, corpus_by_area, paths, sentences, snap):
    sentences.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.name}: {len(sentences):,} rows, {sentences.shape[1]} columns")

    _table_path = paths.PROCESSED / f"corpus_v2_table_{snap.slug}.csv"
    corpus_by_area.to_csv(_table_path, index=False)
    print(f"wrote {_table_path.name}")
    return


if __name__ == "__main__":
    app.run()
