import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import hashlib
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths

    snap = paths.snapshot("30 June, 2026")
    CORPUS_PATH = paths.interim("corpus", snap)   # 05_corpora.py's output

    SEED = 434                  # the same seed as 05
    POOL_PER_VOICE = 8_000      # 24k in total
    # mpnet rather than MiniLM: MiniLM split the same praise into Danish and English topics
    ENCODER = "paraphrase-multilingual-mpnet-base-v2"

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} is missing — run notebooks/05_corpora.py first")
    return CORPUS_PATH, ENCODER, POOL_PER_VOICE, SEED, hashlib, mo, np, paths, pd, snap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07 Discovery

    Discovery suggests candidate frames for the codebook. It does not measure anything. As
    Carlsen & Ralund (2022) put it, a cluster hints at a category, a missing cluster is not
    evidence that something is absent, and a topic's top words are a biased sample of it. Kaur &
    Wallace (2025) describe topic models as an interpretive scaffold rather than as themes.

    Following Grootendorst (2022), each sentence is embedded with a multilingual encoder, reduced
    to 5 dimensions with UMAP, and clustered with HDBSCAN, which leaves sentences it cannot place
    as outliers. Each cluster is then described with class-based TF-IDF.

    The pool has 8,000 corpus sentences per voice, stratified by area, so the large English
    review voice cannot drown out the smaller voices. The encoder is
    `paraphrase-multilingual-mpnet-base-v2`; the smaller MiniLM kept Danish and English praise in
    separate topics, and mpnet does not. District and city names are deleted before embedding,
    because otherwise topics form around which district a sentence names.

    Instead of picking the model with the best score, the notebook fits a range of cluster sizes,
    keeps the settings with at most 100 topics and no topic over 20% of the pool, and reads the
    median one. Kaur & Wallace found that optimising for coherence makes BERTopic too coarse.
    Coherence is not used at all, since for neural topic models it matches human judgement poorly
    (Hoyle et al. 2021).

    Topic words are shown twice, as plain c-TF-IDF and as an MMR list with fewer near-duplicates
    (Grootendorst §7.2). Outliers are left unassigned, and their share is reported per voice,
    because a voice that is hard to place says something about the shared embedding space. The
    encoder gets the sentences as they are, apart from the deleted names. Stopwords are only
    removed from the topic words, since bag-of-words preprocessing makes BERTopic worse (Kaur &
    Wallace). Deciding which topics are frame-less is done by hand on a marking sheet, because
    grouping subjects into frames belongs to the codebook.
    """)
    return


@app.cell
def _(CORPUS_PATH, POOL_PER_VOICE, SEED, np, paths, pd, snap):
    # the pool: 8,000 per voice, stratified by area
    _cols = ["sentence_id", "arm", "lang", "text", "neighbourhood", "listing_id", "in_corpus"]
    _corpus = pd.read_parquet(CORPUS_PATH, columns=_cols)
    _corpus = _corpus[_corpus["in_corpus"]]

    def area_quota(counts: pd.Series, total: int) -> pd.Series:
        """
        Split `total` across areas in proportion to `counts`, in whole numbers that
        sum exactly to `total` (largest-remainder method: floor every share, then give
        the leftover places to the areas that lost most in the flooring).
        """
        exact = counts / counts.sum() * total
        quota = np.floor(exact).astype(int)
        leftover = total - quota.sum()
        biggest_losses = (exact - quota).sort_values(ascending=False).index[:leftover]
        quota[biggest_losses] += 1
        return quota

    _parts = []
    for _i, _arm in enumerate(["host/en", "guest/en", "guest/da"]):
        _voice = _corpus[_corpus["arm"] == _arm]
        _quota = area_quota(_voice["neighbourhood"].value_counts(), POOL_PER_VOICE)
        for _j, (_area, _n) in enumerate(_quota.items()):
            _in_area = _voice[_voice["neighbourhood"] == _area]
            _parts.append(_in_area.sample(_n, random_state=SEED + 100 * _i + _j))

    # sorted, so the order does not depend on the loop
    pool = pd.concat(_parts).sort_values("sentence_id").reset_index(drop=True)

    # pool membership, ids only (used by 06)
    _pool_ids_path = paths.interim("discovery_pool", snap)
    pool[["sentence_id", "arm", "neighbourhood"]].to_parquet(_pool_ids_path, index=False)

    print(f"pool: {len(pool):,} sentences -> {_pool_ids_path.name}")
    print(pool.groupby("arm").size().to_string())
    return (pool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masking district and city names

    Without masking, mpnet put 26% of the pool into a single topic of sentences that mostly just
    named a district or the city. The area of each sentence is already known from its
    coordinates, so names that say where a listing is are deleted from what the encoder sees.
    Names that say what is there, like Tivoli, Nyhavn, parks, streets and stations, are kept.

    The names are deleted rather than replaced with a placeholder like `X`, because the
    placeholder became something topics formed around. Islands Brygge, Christianshavn and
    Carlsberg Byen are kept, since reviews mostly mention them for the harbour bath, the canals
    and the new quarter. Only whole words are matched, so `Nørrebrogade` stays, and the Danish
    letters also match the spellings guests use without a Danish keyboard (`oe`, `ae`, `aa`).
    Only the encoder input and the topic words are masked. The example sentences and every later
    step use the original text.

    Among the topics that were not built around names, a median of 91% of each topic's sentences
    stayed together after masking.
    """)
    return


@app.cell
def _(pool):
    import re

    # Districts and common misspellings: place_filter.TOPONYMS minus the three kept
    # names, and minus "inner city" (usually means "central").
    MASK_AREAS = ["indre by", "nørrebro", "vesterbro", "østerbro", "frederiksberg",
                  "amager", "amagerbro", "valby", "vanløse", "brønshøj", "husum",
                  "bispebjerg", "nordvest", "enghave", "sydhavn", "sydhavnen", "nordhavn",
                  "ørestad", "fredriksberg", "fredericksberg", "vestebro"]
    # the city, including foreign spellings
    MASK_CITY = ["copenhagen", "københavn", "kobenhavn", "cph", "kbh", "kopenhagen",
                 "copenhague", "copenaghen", "københavner"]

    def _accent_tolerant(term):
        # Each Danish letter may also match its keyboard substitutes. Done per
        # character, so the rest of the sentence ("tæt på") is not folded.
        swaps = {"ø": "(?:ø|oe|o)", "æ": "(?:æ|ae|a)", "å": "(?:å|aa|a)"}
        return "".join(swaps.get(ch, re.escape(ch)) for ch in term)

    # longest first, so "amagerbro" wins over "amager"; s? catches the possessive
    _names = sorted(MASK_AREAS + MASK_CITY, key=len, reverse=True)
    MASK_PATTERN = re.compile(
        r"\b(?:" + "|".join(_accent_tolerant(n) for n in _names) + r")s?\b", re.IGNORECASE)

    def mask_names(text):
        # delete the name, then close the double space
        return re.sub(r"\s{2,}", " ", MASK_PATTERN.sub("", text)).strip()

    masked_text = [mask_names(t) for t in pool["text"]]
    _touched = sum(m != t for m, t in zip(masked_text, pool["text"]))
    print(f"sentences with a name deleted: {_touched:,} of {len(pool):,} "
          f"({_touched / len(pool):.1%})")
    return (masked_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding

    `paraphrase-multilingual-mpnet-base-v2` turns a sentence in any of 50 or so languages into a
    vector of 768 numbers, trained so that a sentence and its translation end up close together.
    That is what makes one shared clustering of English and Danish possible, and it is also what
    the language comparison tests: if Danish sentences form their own clusters, the space is not
    really shared.

    MiniLM, which is smaller, kept short Danish and English praise in separate topics. mpnet does
    not, and it still fits on a 4 GB GPU. The vectors are cached under a hash of the masked pool
    and the model name.
    """)
    return


@app.cell
def _(ENCODER, hashlib, masked_text, np, paths, pool, snap):
    import torch
    from sentence_transformers import SentenceTransformer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(ENCODER, device=_device)

    # cache key: sentence ids, in order, the masked text, and the model
    _key = hashlib.sha256(
        (ENCODER + "\n" + "\n".join(s + "\t" + t for s, t in zip(pool["sentence_id"], masked_text)))
        .encode("utf-8")
    ).hexdigest()[:16]
    _cache = paths.INTERIM / f"embeddings_pool_{snap.slug}_{_key}.npy"

    if _cache.exists():
        embeddings = np.load(_cache)
        print(f"embeddings loaded from cache: {_cache.name}")
    else:
        # batch_size 64 to stay well inside a 4 GB GPU
        embeddings = encoder.encode(masked_text, batch_size=64,
                                    show_progress_bar=False, convert_to_numpy=True)
        np.save(_cache, embeddings)
        print(f"embedded {len(pool):,} sentences on {_device} -> {_cache.name}")
    print(f"shape: {embeddings.shape}")
    return embeddings, encoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sweep

    HDBSCAN's `min_cluster_size` is the smallest group it will call a cluster. Small values give
    many narrow clusters, large values a few broad ones and more outliers. BERTopic has no right
    number of topics, so several values are fitted and the one with the median number of topics
    is read (Kaur & Wallace 2025, §4.1).

    The median of a sweep is just the middle of the grid, so two rules for which settings count
    were fixed before the sweep was run. A setting can give at most 100 topics, since Kaur &
    Wallace's participants found 50 to 150 topics overwhelming. And no topic can hold more than
    20% of the pool (4,800 sentences), because such a topic is a catch-all.

    The grid goes from BERTopic's default of 10 up to 400. A tie goes to the model with more
    topics. If fewer than two settings pass, the notebook stops. UMAP is run once with a fixed
    seed, so the models only differ in `min_cluster_size`.
    """)
    return


@app.cell
def _(SEED, embeddings, pd):
    from hdbscan import HDBSCAN
    from umap import UMAP

    # BERTopic's own UMAP defaults, written out; random_state fixes the layout
    UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                       random_state=SEED)
    reduced = UMAP(**UMAP_PARAMS).fit_transform(embeddings)

    # fixed before the first run; do not change after looking at the output
    SWEEP = [10, 15, 20, 25, 35, 50, 70, 100, 140, 200, 280, 400]
    MAX_TOPICS = 100            # more than this is too fine to read
    MAX_LARGEST_SHARE = 0.20    # a topic holding more than this share is a catch-all

    _rows = []
    for _m in SWEEP:
        _labels = HDBSCAN(min_cluster_size=_m, metric="euclidean",
                          cluster_selection_method="eom").fit_predict(reduced)
        _placed = _labels[_labels != -1]
        # largest cluster as a share of the whole pool, outliers included
        _largest = int(pd.Series(_placed).value_counts().max()) if len(_placed) else 0
        _rows.append({"min_cluster_size": _m,
                      "topics": int(len(set(_placed))),
                      "outlier %": round(100 * (_labels == -1).mean(), 1),
                      "largest topic %": round(100 * _largest / len(_labels), 1)})
    sweep_table = pd.DataFrame(_rows).set_index("min_cluster_size")

    sweep_table["admissible"] = ((sweep_table["topics"] <= MAX_TOPICS)
                                 & (sweep_table["largest topic %"] <= 100 * MAX_LARGEST_SHARE))
    _ok = sweep_table[sweep_table["admissible"]]
    if len(_ok) < 2:
        raise RuntimeError(f"only {len(_ok)} admissible setting(s): stop and review the sweep, "
                           "do not loosen the criteria here")

    # median among the admissible settings; most topics first, so with an even count
    # (n - 1) // 2 picks the finer of the two middle models
    _by_topics = _ok.sort_values("topics", ascending=False)
    MEDIAN_M = int(_by_topics.index[(len(_by_topics) - 1) // 2])

    print(f"admissible: min_cluster_size {list(_ok.index)}")
    print(f"median model: min_cluster_size = {MEDIAN_M}")
    sweep_table
    return MEDIAN_M, UMAP_PARAMS, sweep_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The median model

    This is fitted as a full BERTopic model with the same UMAP settings and seed, so it has to
    give the same number of topics as the sweep did. That is checked below.

    Topic words are given twice. c-TF-IDF is BERTopic's own representation. MMR (maximal marginal
    relevance) re-ranks the same candidates so that each word is relevant and also different from
    the ones already chosen, giving for example `metro, walk, minutes` instead of
    `metro, metro station, station`. Neither changes which sentences are in a cluster.

    English and Danish stopwords from NLTK are removed from the topic words only. The topic words
    are computed on the masked text, so district names cannot come out on top.
    """)
    return


@app.cell
def _(MEDIAN_M, SEED, UMAP_PARAMS, embeddings, encoder, masked_text, pool, sweep_table):
    import nltk
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN as _HDBSCAN
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP as _UMAP

    nltk.download("stopwords", quiet=True)
    _stop = sorted(set(stopwords.words("english")) | set(stopwords.words("danish")))

    # min_df=5 keeps one host's pet phrase or a typo off the top; bigrams for
    # "walking distance", "tæt på"
    _vectorizer = CountVectorizer(stop_words=_stop, ngram_range=(1, 2), min_df=5)

    topic_model = BERTopic(
        embedding_model=encoder,   # MMR needs it to embed candidate words
        umap_model=_UMAP(**UMAP_PARAMS),
        hdbscan_model=_HDBSCAN(min_cluster_size=MEDIAN_M, metric="euclidean",
                               cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=_vectorizer,
        representation_model={"MMR": MaximalMarginalRelevance(diversity=0.3)},
        top_n_words=10,
        calculate_probabilities=False,
    )
    # masked text, so topic words match what the encoder saw
    docs = list(masked_text)
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    pool_topics = pool.assign(topic=topics)

    # the full model must reproduce the sweep
    _n = len(set(topics)) - (1 if -1 in topics else 0)
    _expected = int(sweep_table.loc[MEDIAN_M, "topics"])
    print(f"topics: {_n} (sweep said {_expected})")
    if _n != _expected:
        raise AssertionError("the median model did not reproduce the sweep — UMAP is not deterministic here")
    print(f"seed {SEED}; outliers: {(pool_topics.topic == -1).mean():.1%}")
    return pool_topics, topic_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two things to watch

    Clusters could split by voice, with marketing and reviews never sharing a cluster. Or they
    could split by language, although a multilingual encoder should put "tæt på metroen" next to
    "close to the metro".

    The pool is one third per voice, so a topic with no voice preference would be one third of
    each. The guest sentences are half English and half Danish. A topic is called voice-pure if
    one voice makes up at least 80% of it, and language-pure if at least 90% of its guest
    sentences are in one language (only judged with at least 20 guest sentences). These are
    just ways of summarising; the table per topic is the actual evidence.
    """)
    return


@app.cell
def _(pd, pool_topics):
    _ARMS = ["host/en", "guest/en", "guest/da"]

    # outliers per voice
    _out = pool_topics.assign(outlier=pool_topics.topic == -1).groupby("arm")["outlier"].mean()
    print("outlier share by voice (baseline: equal across voices):")
    print((100 * _out.reindex(_ARMS)).round(1).to_string())

    _placed = pool_topics[pool_topics.topic != -1]
    _mix = pd.crosstab(_placed.topic, _placed.arm).reindex(columns=_ARMS, fill_value=0)
    topic_mix = pd.DataFrame({
        "size": _mix.sum(axis=1),
        **{f"{a} %": (100 * _mix[a] / _mix.sum(axis=1)).round(1) for a in _ARMS},
    })
    _guest = _mix["guest/en"] + _mix["guest/da"]
    topic_mix["da share of guest %"] = (100 * _mix["guest/da"] / _guest.where(_guest > 0)).round(1)
    topic_mix["voice-pure"] = topic_mix[[f"{a} %" for a in _ARMS]].max(axis=1) >= 80
    _judged = _guest >= 20
    topic_mix["language-pure"] = _judged & (
        (topic_mix["da share of guest %"] >= 90) | (topic_mix["da share of guest %"] <= 10)
    )

    _n = len(topic_mix)
    _vp = topic_mix["voice-pure"]
    _lp = topic_mix["language-pure"]
    print(f"\n{_n} topics")
    print(f"voice-pure (one voice >= 80%):    {int(_vp.sum()):>3} topics, "
          f"{topic_mix.loc[_vp, 'size'].sum() / topic_mix['size'].sum():.1%} of placed sentences")
    for _a in _ARMS:
        _k = int((topic_mix[f"{_a} %"] >= 80).sum())
        print(f"   dominated by {_a:9} {_k:>3}")
    print(f"language-pure (>= 90% one language, of {int(_judged.sum())} topics judged): "
          f"{int(_lp.sum())}")
    return (topic_mix,)


@app.cell
def _(pd, topic_mix, topic_model):
    # topic words: c-TF-IDF and MMR side by side
    _info = topic_model.get_topic_info().set_index("Topic")
    _info = _info[_info.index != -1]

    def _words(col):
        return _info[col].map(lambda ws: ", ".join(w for w in ws if w)[:120])

    topic_table = pd.concat(
        [topic_mix, pd.DataFrame({"c-TF-IDF words": _words("Representation"),
                                  "MMR words": _words("MMR")})],
        axis=1,
    ).sort_values("size", ascending=False)
    topic_table.index.name = "topic"
    topic_table
    return (topic_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Marking frame-less topics

    Grouping subjects such as metro, cafés or noise into frames is interpretation and belongs to
    the codebook. So the notebook writes a marking sheet instead: one row per topic with its size,
    voice mix and words, and two empty columns. `frameless` is `yes` if the topic is about the stay
    or general praise ("great location", "lovely flat") rather than about how the place is
    described, and `no` if it is about place. `note` is for anything worth keeping for the
    codebook.

    The sheet is never overwritten. If it already exists, the notebook only warns when the topics
    no longer match it. It contains words and counts, no review text.

    This marks topics. It does not replace the blind sentence-level check drawn in 05
    (`frameless_check_<slug>.csv`), because topics are not a random sample of the corpus.
    """)
    return


@app.cell
def _(paths, pd, snap, topic_table):
    marks_path = paths.PROCESSED / f"discovery_marks_{snap.slug}.csv"

    _sheet = topic_table[["size", "host/en %", "guest/en %", "guest/da %",
                          "c-TF-IDF words", "MMR words"]].copy()
    _sheet["frameless"] = ""   # yes / no
    _sheet["note"] = ""

    if not marks_path.exists():
        _sheet.to_csv(marks_path)
        print(f"wrote a blank marking sheet: {marks_path.name}")
    else:
        # hand-made marks cannot be regenerated: compare, never replace
        _old = pd.read_csv(marks_path, index_col="topic")
        _same = (sorted(_old.index) == sorted(_sheet.index)
                 and (_old["size"].reindex(_sheet.index) == _sheet["size"]).all())
        if _same:
            _n = int(_old["frameless"].notna().sum())
            print(f"marking sheet kept as it is: {marks_path.name} ({_n} of {len(_old)} marked)")
        else:
            print(f"⚠️ {marks_path.name} exists but its topics do not match this model. "
                  "It was NOT overwritten. Move it aside to get a fresh sheet.")
    return (marks_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write the reading material

    `data/interim/discovery_reading_<slug>.md` has the sweep, then every topic with its words, its
    voice mix and ten example sentences spread across the voices (up to four per voice, the most
    typical first, in the original text). It contains review text, so it stays on this machine.
    `data/processed/discovery_topics_<slug>.csv` is the topic table with words and counts only, and
    `figures/discovery_voice_mix.png` shows the voice mix of every topic.

    The most typical sentences are the ones HDBSCAN places most firmly in their cluster, measured
    from 0 at the edge to 1 at the core.
    """)
    return


@app.cell
def _(MEDIAN_M, marks_path, paths, pool_topics, snap, sweep_table, topic_model, topic_table):
    _strength = topic_model.probabilities_   # membership strength per sentence
    _pt = pool_topics.assign(strength=_strength)

    _lines = [f"# Discovery reading — {snap.slug}", "",
              "Generated by notebooks/07_discovery.py. Contains review text: never commit, "
              "never quote verbatim in the write-up.", "",
              f"Mark each topic in `data/processed/{marks_path.name}` while reading "
              "(`frameless`: yes/no, plus a `note`).", "",
              "## The sweep", "",
              f"Admissible: at most 100 topics and no topic over 20% of the pool. "
              f"Median admissible model: min_cluster_size = {MEDIAN_M}.", "",
              "```", sweep_table.to_string(), "```", ""]
    for _t, _r in topic_table.iterrows():
        _lines += [f"## Topic {_t} — {_r['size']} sentences "
                   f"(host {_r['host/en %']}% · guest/en {_r['guest/en %']}% · "
                   f"guest/da {_r['guest/da %']}%)", "",
                   f"- **c-TF-IDF:** {_r['c-TF-IDF words']}",
                   f"- **MMR:** {_r['MMR words']}", ""]
        _in = _pt[_pt.topic == _t].sort_values("strength", ascending=False)
        # up to four per voice, then fill to ten from any voice
        _picked = []
        for _arm in ["host/en", "guest/en", "guest/da"]:
            _picked += list(_in[_in.arm == _arm].index[:4])
        _picked += [i for i in _in.index if i not in _picked][: max(0, 10 - len(_picked))]
        for _i in _picked[:10]:
            _lines.append(f"- `{_pt.at[_i, 'arm']}` {_pt.at[_i, 'text']}")
        _lines.append("")

    _reading = paths.INTERIM / f"discovery_reading_{snap.slug}.md"
    _reading.write_text("\n".join(_lines), encoding="utf-8")
    topic_table.to_csv(paths.PROCESSED / f"discovery_topics_{snap.slug}.csv")

    # topic per pool sentence, ids only (used by 08)
    _assign = _pt[["sentence_id", "arm", "neighbourhood", "topic", "strength"]]
    _assign.to_parquet(paths.INTERIM / f"discovery_assignments_{snap.slug}.parquet", index=False)
    print(f"wrote {_reading.name}, discovery_topics_{snap.slug}.csv, "
          f"discovery_assignments_{snap.slug}.parquet")
    return


@app.cell
def _(paths, topic_table):
    # figure: voice mix of every topic
    # one stacked bar per topic, largest at the top; same voice colours as elsewhere
    import matplotlib

    matplotlib.use("Agg")   # headless: no window, just the file
    import matplotlib.pyplot as plt

    _COLOURS = {"host/en": "#2a78d6", "guest/en": "#eb6834", "guest/da": "#1baf7a"}
    _t = topic_table.sort_values("size")   # barh draws bottom-up
    _fig, _ax = plt.subplots(figsize=(8, max(4, 0.22 * len(_t))), dpi=150)
    _left = [0.0] * len(_t)
    for _arm, _c in _COLOURS.items():
        _vals = _t[f"{_arm} %"].tolist()
        _ax.barh([str(i) for i in _t.index], _vals, left=_left, color=_c,
                 edgecolor="white", linewidth=2, height=0.8, label=_arm)
        _left = [a + b for a, b in zip(_left, _vals)]
    _ax.set_xlim(0, 100)
    _ax.set_xlabel("share of the topic's sentences (%)")
    _ax.set_ylabel("topic (largest at top)")
    _ax.set_title("Voice composition per topic — the pool is one third per voice",
                  loc="left", fontsize=10)
    _ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8)
    for _s in ["top", "right"]:
        _ax.spines[_s].set_visible(False)
    _ax.tick_params(axis="y", labelsize=6)
    _fig.tight_layout()
    _out = paths.FIGURES / "discovery_voice_mix.png"
    _fig.savefig(_out)
    print(f"wrote {_out.name}")
    return


if __name__ == "__main__":
    app.run()
