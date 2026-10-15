import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import re
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths
    import place_filter as pf

    snap = paths.snapshot("30 June, 2026")

    CORPUS_PATH = paths.interim("corpus", snap)   # 05_corpora.py's output

    # small count tables, written as CSV
    OUT_CONCEPTS = paths.processed("x1_concepts_by_voice", snap).with_suffix(".csv")
    OUT_TERMS = paths.processed("x1_top_terms", snap).with_suffix(".csv")
    OUT_HOST_GUEST = paths.processed("x1_host_vs_guest", snap).with_suffix(".csv")
    OUT_DA_EN = paths.processed("x1_da_vs_en", snap).with_suffix(".csv")
    OUT_FIGURE = paths.FIGURES / "x1_concepts_by_voice.png"

    ARMS = ["host/en", "guest/en", "guest/da"]

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} is missing — run notebooks/05_corpora.py first")

    print(f"snapshot: {snap}")
    print(f"in: {CORPUS_PATH.name}")
    return (
        ARMS,
        CORPUS_PATH,
        OUT_CONCEPTS,
        OUT_DA_EN,
        OUT_FIGURE,
        OUT_HOST_GUEST,
        OUT_TERMS,
        mo,
        pd,
        pf,
        plt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 Experiment 1: do the three voices describe place with the same words?

    This is a side experiment, and nothing later in the pipeline uses its output.

    Among the sentences the place filter keeps, do host/en, guest/en and guest/da use the same
    place vocabulary, and which concepts and words does each voice rely on? The method is
    simple frequency counts of the lexicon in `scripts/place_filter.py`. Every number is a share
    of the voice's kept sentences, because the voices differ in size by up to 21 times.

    The notebook only outputs counts and keywords, never sentence text.
    """)
    return


@app.cell
def _(ARMS, CORPUS_PATH, pd):
    corpus = pd.read_parquet(
        CORPUS_PATH,
        columns=["sentence_id", "arm", "text", "listing_id", "in_corpus"],
    )

    # `in_corpus` marks the three corpora built in 05
    kept = corpus[corpus["in_corpus"]].drop(columns="in_corpus").reset_index(drop=True)
    del corpus  # free ~1 GB

    # Listings too, in case a few prolific listings dominate a voice.
    size = kept.groupby("arm").agg(
        sentences=("sentence_id", "size"),
        listings=("listing_id", "nunique"),
    ).reindex(ARMS)
    size
    return kept, size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: which lexicon words does each sentence contain?

    Words are matched the same way the filter matches them. Kind B phrases are crossed out
    first, so `seating area` is not counted as `area`. The longest words come first in the
    pattern, so "city centre" is not counted as `city`. Toponyms are matched on text with the
    Danish letters folded, so `Norrebro` counts as `nørrebro`. Matching uses Python's `re` with
    `.map()`, not `.str.contains`, because RE2 breaks on ø, å and æ.
    """)
    return


@app.cell
def _(kept, pf, re):
    # lexicon words, longest first
    _words = sorted(pf.REFERENCE_WORDS, key=len, reverse=True)
    word_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in _words) + r")\b",
        flags=re.IGNORECASE,
    )

    # Toponyms: pattern from folded spellings, plus a lookup back to the lexicon
    # spelling (from 'norrebro' back to 'nørrebro').
    _folded_to_toponym = {pf.fold(t): t for t in pf.TOPONYMS}
    _topos = sorted(_folded_to_toponym, key=len, reverse=True)
    toponym_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in _topos) + r")\b",
        flags=re.IGNORECASE,
    )

    _kind_b = pf.make_pattern(pf.EXCLUSION_B)

    def words_in(sentence):
        """The set of lexicon words and toponyms in one sentence."""
        found = set()

        # lexicon words, with Kind B crossed out
        crossed = _kind_b.sub(" ", sentence)
        for match in word_pattern.findall(crossed):
            found.add(match.lower())

        # toponyms, on the folded original text (as in the filter)
        for match in toponym_pattern.findall(pf.fold(sentence)):
            found.add(_folded_to_toponym[match.lower()])

        return found

    # A set per sentence: does the sentence use the word, not how many times.
    kept["words"] = kept["text"].fillna("").map(words_in)

    # word to concept; toponyms get their own row
    concept_of = {}
    for _concept, _entry in pf.CONCEPTS.items():
        for _w in _entry["en"] + _entry["da"]:
            concept_of[_w] = _concept
    for _t in pf.TOPONYMS:
        concept_of[_t] = "toponyms"

    # long table: one row per (sentence, word)
    hits = (
        kept[["sentence_id", "arm", "listing_id", "words"]]
        .explode("words")
        .dropna(subset=["words"])
        .rename(columns={"words": "word"})
    )
    hits["concept"] = hits["word"].map(concept_of)
    print(f"{len(hits):,} (sentence, word) pairs")
    return (hits,)


@app.cell
def _(ARMS, hits, kept, size):
    # Sanity check: every kept sentence should contain a lexicon word or toponym,
    # otherwise this matcher is not the filter's matcher.
    _with_hit = hits.groupby("arm")["sentence_id"].nunique().reindex(ARMS)
    _share = _with_hit / size["sentences"] * 100
    print("share of kept sentences with at least one lexicon word or toponym:")
    for _arm in ARMS:
        print(f"  {_arm:9} {_share[_arm]:6.2f}%")
    assert (_share > 99).all(), "the matcher disagrees with the filter — investigate before reading"

    # words starting/ending with a Danish letter must be found
    for _w in ["østerbro", "område", "nørrebro"]:
        _n = int((hits["word"] == _w).sum())
        print(f"  {_w:10} {_n:>7,} sentences")
        assert _n > 0, f"{_w} never matched: a word-boundary problem"
    del kept["words"]  # free memory
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 1: concepts by voice

    The share of each voice's kept sentences that contain at least one word from a concept. A
    sentence can mention several concepts, so the columns do not add up to 100%.
    """)
    return


@app.cell
def _(ARMS, OUT_CONCEPTS, hits, size):
    # one row per (sentence, concept)
    _per_concept = hits.drop_duplicates(["sentence_id", "concept"])
    _n = _per_concept.groupby(["concept", "arm"]).size().unstack("arm").reindex(columns=ARMS).fillna(0)

    concept_table = (_n / size["sentences"] * 100).round(1)

    concept_table = concept_table.sort_values("guest/en", ascending=False)
    concept_table.to_csv(OUT_CONCEPTS)
    concept_table
    return (concept_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 2: the 15 most used words in each voice

    `%` is the share of the voice's kept sentences, and `listings` is how many listings those
    sentences come from. A high share from only a few listings would mean a few prolific
    writers.
    """)
    return


@app.cell
def _(ARMS, OUT_TERMS, hits, pd, size):
    _counts = hits.groupby(["arm", "word", "concept"]).agg(
        sentences=("sentence_id", "size"),
        listings=("listing_id", "nunique"),
    ).reset_index()
    _counts["%"] = (_counts["sentences"] / _counts["arm"].map(size["sentences"]) * 100).round(2)

    # top 15 per voice
    _tops = []
    for _arm in ARMS:
        _t = _counts[_counts["arm"] == _arm].nlargest(15, "sentences").copy()
        _t.insert(0, "rank", range(1, len(_t) + 1))
        _tops.append(_t)
    top_terms = pd.concat(_tops, ignore_index=True)
    top_terms.to_csv(OUT_TERMS, index=False)

    # the three voices side by side
    top_terms_wide = pd.concat(
        [
            top_terms[top_terms["arm"] == _arm]
            .set_index("rank")[["word", "%"]]
            .rename(columns={"word": f"{_arm} word", "%": f"{_arm} %"})
            for _arm in ARMS
        ],
        axis=1,
    )
    word_counts = _counts
    top_terms_wide
    return (word_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 3: hosts and guests, English only

    Here the language, the lexicon and the city are the same, so only the kind of text differs.
    For each word:

    $$\text{ratio} = \frac{(h + 0.5) \,/\, H}{(g + 0.5) \,/\, G}$$

    where $h$ and $g$ are the host and guest sentences containing the word, and $H$ and $G$ are
    all kept sentences in each voice. A ratio of 3 means the word is three times as common in
    host place descriptions.

    Adding 0.5 keeps the ratio finite when a word is missing on one side. A word needs at least
    30 sentences on one side, so the top of the list is not words seen 3 times against 0.
    """)
    return


@app.cell
def _(OUT_HOST_GUEST, mo, pd, size, word_counts):
    _h = word_counts[word_counts["arm"] == "host/en"].set_index("word")
    _g = word_counts[word_counts["arm"] == "guest/en"].set_index("word")

    # every word used by either voice, with 0 where it is missing
    host_guest = _h[["concept", "sentences"]].rename(columns={"sentences": "host n"}).join(
        _g[["concept", "sentences"]].rename(columns={"sentences": "guest n", "concept": "_c"}),
        how="outer",
    )
    host_guest["concept"] = host_guest["concept"].fillna(host_guest["_c"])
    host_guest = host_guest.drop(columns="_c")
    host_guest[["host n", "guest n"]] = host_guest[["host n", "guest n"]].fillna(0).astype(int)

    _H, _G = size.loc["host/en", "sentences"], size.loc["guest/en", "sentences"]
    host_guest["host %"] = (host_guest["host n"] / _H * 100).round(2)
    host_guest["guest %"] = (host_guest["guest n"] / _G * 100).round(2)
    host_guest["host/guest"] = (
        ((host_guest["host n"] + 0.5) / _H) / ((host_guest["guest n"] + 0.5) / _G)
    ).round(2)

    # at least 30 sentences on one side
    host_guest = host_guest[(host_guest["host n"] >= 30) | (host_guest["guest n"] >= 30)]
    host_guest = host_guest.sort_values("host/guest", ascending=False)
    host_guest.to_csv(OUT_HOST_GUEST)

    # 3a: all words, top 10 each way
    table_3a = pd.concat([
        host_guest.head(10).assign(leans="host"),
        host_guest.tail(10).iloc[::-1].assign(leans="guest"),
    ])

    # 3b: without toponyms. Hosts name small local places guests rarely do, so 3a's
    # host side is all street names.
    _common = host_guest[host_guest["concept"] != "toponyms"]
    table_3b = pd.concat([
        _common.head(10).assign(leans="host"),
        _common.tail(10).iloc[::-1].assign(leans="guest"),
    ])

    mo.vstack([
        mo.md(f"**Table 3a — all words** ({len(host_guest)} pass the 30-sentence minimum)"),
        table_3a,
        mo.md(f"**Table 3b — without toponyms** ({len(_common)} words)"),
        table_3b,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 3c: Danish and English guests

    The same ratio, with guest/da instead of host/en:
    $\text{ratio} = \frac{(d + 0.5) / D}{(e + 0.5) / E}$. Both are reviews, but the language
    differs, so a word's ratio also reflects how each language divides a concept into words.
    """)
    return


@app.cell
def _(OUT_DA_EN, mo, pd, size, word_counts):
    _d = word_counts[word_counts["arm"] == "guest/da"].set_index("word")
    _e = word_counts[word_counts["arm"] == "guest/en"].set_index("word")

    # every word used by either voice, with 0 where it is missing
    _da_en = _d[["concept", "sentences"]].rename(columns={"sentences": "da n"}).join(
        _e[["concept", "sentences"]].rename(columns={"sentences": "en n", "concept": "_c"}),
        how="outer",
    )
    _da_en["concept"] = _da_en["concept"].fillna(_da_en["_c"])
    _da_en = _da_en.drop(columns="_c")
    _da_en[["da n", "en n"]] = _da_en[["da n", "en n"]].fillna(0).astype(int)

    _D, _E = size.loc["guest/da", "sentences"], size.loc["guest/en", "sentences"]
    _da_en["da %"] = (_da_en["da n"] / _D * 100).round(2)
    _da_en["en %"] = (_da_en["en n"] / _E * 100).round(2)
    _da_en["da/en"] = (
        ((_da_en["da n"] + 0.5) / _D) / ((_da_en["en n"] + 0.5) / _E)
    ).round(2)

    # at least 30 sentences on one side
    _da_en = _da_en[(_da_en["da n"] >= 30) | (_da_en["en n"] >= 30)]
    _da_en = _da_en.sort_values("da/en", ascending=False)
    _da_en.to_csv(OUT_DA_EN)

    # all words, top 10 each way
    _all = pd.concat([
        _da_en.head(10).assign(leans="da"),
        _da_en.tail(10).iloc[::-1].assign(leans="en"),
    ])

    # without toponyms
    _common = _da_en[_da_en["concept"] != "toponyms"]
    _no_topo = pd.concat([
        _common.head(10).assign(leans="da"),
        _common.tail(10).iloc[::-1].assign(leans="en"),
    ])

    mo.vstack([
        mo.md(f"**Table 3c — all words** ({len(_da_en)} pass the 30-sentence minimum)"),
        _all,
        mo.md(f"**Table 3d — without toponyms** ({len(_common)} words)"),
        _no_topo,
    ])
    return


@app.cell
def _(ARMS, OUT_FIGURE, concept_table, plt):
    # figure: concepts by voice as a dot plot
    # colour-blind-safe colours, plus a marker shape per voice
    _style = {
        "host/en": ("#2a78d6", "o"),
        "guest/en": ("#eb6834", "s"),
        "guest/da": ("#1baf7a", "D"),
    }
    _rows = concept_table.index[::-1]  # largest at the top
    _fig, _ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for _y, _concept in enumerate(_rows):
        # faint line spanning the three dots
        _vals = concept_table.loc[_concept, ARMS]
        _ax.plot([_vals.min(), _vals.max()], [_y, _y], color="#d0d0cc", lw=2, zorder=1)
    for _arm in ARMS:
        _colour, _marker = _style[_arm]
        _ax.scatter(concept_table.loc[_rows, _arm], range(len(_rows)), s=45,
                    color=_colour, marker=_marker, label=_arm, zorder=2,
                    edgecolor="white", linewidth=1)
    _ax.set_yticks(range(len(_rows)), _rows)
    _ax.set_xlabel("% of the perspective's place-describing sentences")
    _ax.set_xlim(left=0)
    _ax.grid(axis="x", color="#ececea", lw=0.8)
    _ax.set_axisbelow(True)
    for _side in ["top", "right", "left"]:
        _ax.spines[_side].set_visible(False)
    _ax.tick_params(axis="y", length=0)
    _ax.legend(frameon=False, loc="lower right")
    _ax.set_title("Place concepts by perspective", loc="left", fontsize=11)
    _fig.tight_layout()
    _fig.savefig(OUT_FIGURE)
    print(f"saved {OUT_FIGURE}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Caveats

    Only Table 3 compares the kind of text with everything else held equal. Comparisons with
    guest/da differ in language too, and the two languages split concepts into words
    differently: `location` is one English word but five Danish ones.

    The lexicon decides what can be seen. These are frequencies of the words the filter was
    built from, in sentences that were selected by those same words. They show how the filter
    sees each voice, not everything people write about place.

    Long reviews contribute more sentences than short ones. The `listings` column in Table 2 is
    the check for this.
    """)
    return


if __name__ == "__main__":
    app.run()
