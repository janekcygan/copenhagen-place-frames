import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import hashlib
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths

    snap = paths.snapshot("30 June, 2026")
    CORPUS_PATH = paths.interim("corpus", snap)          # 05_corpora.py's output
    POOL_IDS_PATH = paths.interim("discovery_pool", snap)  # 07_discovery.py's pool (ids only)

    SEED = 434   # the same seed as 05 and 07

    # counts, shares and topic words only
    OUT_SUMMARY = paths.processed("x2_arms", snap).with_suffix(".csv")
    OUT_FIGURE = paths.FIGURES / "x2_arms.png"

    for _p in [CORPUS_PATH, POOL_IDS_PATH]:
        if not _p.exists():
            raise FileNotFoundError(f"{_p} is missing — run notebooks/05 and 07 first")
    return (
        CORPUS_PATH,
        OUT_FIGURE,
        OUT_SUMMARY,
        POOL_IDS_PATH,
        SEED,
        hashlib,
        mo,
        np,
        paths,
        pd,
        snap,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 Experiment 2: do BERTopic's default settings fit this corpus?

    This is a side experiment, and nothing later in the pipeline uses its output. It uses the
    sentence pool that `07_discovery.py` saves, so 07 has to be run first.

    How do BERTopic's standard settings need to change for this collection: its size, its short
    sentences and its two languages? BERTopic is fitted three times on the same 24,000
    sentences, changing one group of settings at a time. Topic words use BERTopic's default
    c-TF-IDF in all three.

    Arm A is BERTopic out of the box: the English encoder `all-MiniLM-L6-v2`, a minimum cluster
    size of 10, and single words without stopword removal as topic words. Arm B changes only
    the encoder, to `paraphrase-multilingual-MiniLM-L12-v2`, which is what
    `BERTopic(language="multilingual")` uses. Arm C keeps that encoder, raises the minimum
    cluster size to 70, removes English and Danish stopwords from the topic words, adds word
    pairs and sets `min_df=5`. Arm C is the first tuned version of 07, before it moved to mpnet
    and name masking.

    The clusters depend only on the encoder and the minimum cluster size. The vectorizer only
    changes the words that describe a cluster (Grootendorst 2022, §3.3). So A against B shows
    the effect of the encoder, and in B against C the two changes affect different outputs.

    Every arm fixes UMAP's `random_state`, which BERTopic does not do by default, so the
    differences are not just run-to-run noise.

    For each arm the notebook counts topics and outliers, outliers per voice, and language
    isolation: the share of guest/da sentences in topics that are at least 90% Danish, and the
    same for English. In a shared space both would be close to zero. It also measures how many
    topic words are stopwords, which is zero by construction in arm C.
    """)
    return


@app.cell
def _(CORPUS_PATH, POOL_IDS_PATH, pd):
    # rebuild 07's pool from its saved ids
    _ids = pd.read_parquet(POOL_IDS_PATH)
    _corpus = pd.read_parquet(CORPUS_PATH, columns=["sentence_id", "text"])
    pool = (
        _ids.merge(_corpus, on="sentence_id", how="left", validate="one_to_one")
        .sort_values("sentence_id")   # 07's order, so the embedding cache key matches
        .reset_index(drop=True)
    )
    if pool["text"].isna().any():
        raise ValueError("some pool ids are no longer in the corpus — 05 was re-run since 07")
    docs = pool["text"].tolist()

    VOICES = ["host/en", "guest/en", "guest/da"]
    print(f"pool: {len(pool):,} sentences")
    print(pool.groupby("arm").size().reindex(VOICES).to_string())
    return VOICES, docs, pool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three arms

    BERTopic's own UMAP and HDBSCAN defaults are written out, and only the lines marked
    *changed* differ between arms. The embeddings are cached under a hash of the pool and the
    encoder, in the same way as in 07.
    """)
    return


@app.cell
def _():
    from nltk.corpus import stopwords as _sw
    import nltk

    nltk.download("stopwords", quiet=True)
    # one bilingual list: cleans arm C's topic words and measures metric 4 in every arm
    STOPWORDS = sorted(set(_sw.words("english")) | set(_sw.words("danish")))

    MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
    ARM_SETTINGS = {
        "A default": dict(
            encoder="all-MiniLM-L6-v2",          # BERTopic's English default
            min_cluster_size=10,                 # BERTopic's default
            vectorizer=dict(),                   # CountVectorizer() as-is: unigrams, no stopwords
        ),
        "B multilingual": dict(
            encoder=MULTILINGUAL,                # changed: language
            min_cluster_size=10,
            vectorizer=dict(),
        ),
        "C tuned (07)": dict(
            encoder=MULTILINGUAL,
            min_cluster_size=70,                 # changed: median model of the sweep (25 to 200)
            vectorizer=dict(stop_words=STOPWORDS,   # changed: topic-word settings
                            ngram_range=(1, 2), min_df=5),
        ),
    }
    return ARM_SETTINGS, STOPWORDS


@app.cell
def _(ARM_SETTINGS, docs, hashlib, np, paths, pool, snap, time):
    # embed the pool once per encoder (cached)
    import torch
    from sentence_transformers import SentenceTransformer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = {}     # encoder name to loaded model
    embeddings = {}   # encoder name to (24000, dim) array

    for _name in sorted({s["encoder"] for s in ARM_SETTINGS.values()}):
        encoders[_name] = SentenceTransformer(_name, device=_device)
        # same key as 07: encoder name + every (id, text) pair in pool order
        _key = hashlib.sha256(
            (_name + "\n" + "\n".join(pool["sentence_id"] + "\t" + pool["text"])).encode("utf-8")
        ).hexdigest()[:16]
        _cache = paths.INTERIM / f"embeddings_pool_{snap.slug}_{_key}.npy"
        if _cache.exists():
            embeddings[_name] = np.load(_cache)
            print(f"{_name}: loaded from cache ({_cache.name})")
        else:
            _t0 = time.time()
            embeddings[_name] = encoders[_name].encode(
                docs, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
            np.save(_cache, embeddings[_name])
            print(f"{_name}: embedded on {_device} in {time.time() - _t0:.0f}s -> {_cache.name}")
        print(f"   shape {embeddings[_name].shape}")
    return embeddings, encoders


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit the three models

    Each arm is a complete BERTopic fit. This takes a few minutes, because UMAP with a fixed seed
    runs on one thread.
    """)
    return


@app.cell
def _(ARM_SETTINGS, SEED, docs, embeddings, encoders, time):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    models = {}   # arm to fitted BERTopic
    labels = {}   # arm to topic id per sentence (-1 is an outlier)
    fit_seconds = {}

    for _arm, _s in ARM_SETTINGS.items():
        _t0 = time.time()
        _model = BERTopic(
            embedding_model=encoders[_s["encoder"]],
            # BERTopic's default UMAP, plus random_state
            umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                            metric="cosine", random_state=SEED),
            # BERTopic's default HDBSCAN; only min_cluster_size varies
            hdbscan_model=HDBSCAN(min_cluster_size=_s["min_cluster_size"], metric="euclidean",
                                  cluster_selection_method="eom", prediction_data=True),
            vectorizer_model=CountVectorizer(**_s["vectorizer"]),
            top_n_words=10,
            calculate_probabilities=False,
        )
        labels[_arm], _ = _model.fit_transform(docs, embeddings=embeddings[_s["encoder"]])
        models[_arm] = _model
        fit_seconds[_arm] = round(time.time() - _t0)
        print(f"{_arm:15} fitted in {fit_seconds[_arm]}s")

    # Arm C reproduces 07's first tuned model, which found 37 topics. A different number
    # means the pool or the settings have drifted.
    _c = labels["C tuned (07)"]
    _n_c = len(set(_c)) - (1 if -1 in _c else 0)
    if _n_c != 37:
        raise AssertionError(f"arm C found {_n_c} topics, 07 found 37 — not the same model")
    print("arm C reproduces 07 (37 topics)")
    return fit_seconds, labels, models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The comparison

    For every topic, the notebook takes the Danish share of its guest sentences. A topic counts
    as Danish-only if at least 90% of its guest sentences are Danish, and English-only if at most
    10% are. Topics with fewer than 10 guest sentences are not judged. 10 is the smallest cluster
    arms A and B can make, so the rule is the same in every arm.

    Half of the pool's guest sentences are English and half are Danish, so a topic with no
    language preference would be about 50/50. The 90% cut-off is a way of summarising, not a
    statistical test.
    """)
    return


@app.cell
def _(ARM_SETTINGS, STOPWORDS, VOICES, fit_seconds, labels, models, pd, pool):
    _stop = set(STOPWORDS)

    def language_isolation(df: pd.DataFrame, min_guest: int = 10) -> dict:
        """
        Share of each guest language's placed sentences that sit in a topic
        where that language is >= 90% of the topic's guest sentences.
        """
        placed = df[(df["topic"] != -1) & (df["arm"] != "host/en")]
        per_topic = pd.crosstab(placed["topic"], placed["arm"]).reindex(
            columns=["guest/en", "guest/da"], fill_value=0)
        guest_n = per_topic.sum(axis=1)
        judged = per_topic[guest_n >= min_guest]
        da_share = judged["guest/da"] / judged.sum(axis=1)
        da_isolated = da_share[da_share >= 0.9].index
        en_isolated = da_share[da_share <= 0.1].index
        return {
            "guest/da in Danish-only topics %": round(
                100 * per_topic.loc[da_isolated, "guest/da"].sum() / per_topic["guest/da"].sum(), 1),
            "guest/en in English-only topics %": round(
                100 * per_topic.loc[en_isolated, "guest/en"].sum() / per_topic["guest/en"].sum(), 1),
            "guest sentences in unjudged topics %": round(
                100 * per_topic[guest_n < min_guest].sum().sum() / per_topic.sum().sum(), 1),
        }

    def stopword_share(model) -> float:
        """Mean share of a topic's top-10 words that are stopwords (a bigram
        counts only if both its words are stopwords)."""
        shares = []
        for topic_id in model.get_topics():
            if topic_id == -1:
                continue
            words = [w for w, _ in model.get_topic(topic_id) if w]
            if words:
                n_stop = sum(all(tok in _stop for tok in w.split()) for w in words)
                shares.append(n_stop / len(words))
        return round(100 * sum(shares) / len(shares), 1)

    _rows = []
    for _arm in ARM_SETTINGS:
        _df = pool.assign(topic=labels[_arm])
        _outlier_by_voice = _df.assign(out=_df["topic"] == -1).groupby("arm")["out"].mean()
        _rows.append({
            "arm": _arm,
            "topics": _df.loc[_df["topic"] != -1, "topic"].nunique(),
            "outliers %": round(100 * (_df["topic"] == -1).mean(), 1),
            **{f"outliers {v} %": round(100 * _outlier_by_voice[v], 1) for v in VOICES},
            **language_isolation(_df),
            "stopword share of topic words %": stopword_share(models[_arm]),
            "fit seconds": fit_seconds[_arm],
        })
    summary = pd.DataFrame(_rows).set_index("arm")
    summary.T   # one column per arm
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Topic words in each arm

    The words of the ten largest topics in each arm.
    """)
    return


@app.cell
def _(ARM_SETTINGS, models, pd):
    _rows = []
    for _arm in ARM_SETTINGS:
        _info = models[_arm].get_topic_info()
        _info = _info[_info["Topic"] != -1].sort_values("Count", ascending=False).head(10)
        for _, _r in _info.iterrows():
            _rows.append({"arm": _arm, "topic": _r["Topic"], "size": _r["Count"],
                          "words": ", ".join(w for w in _r["Representation"] if w)})
    top_topics = pd.DataFrame(_rows)
    top_topics
    return (top_topics,)


@app.cell
def _(OUT_SUMMARY, paths, snap, summary, top_topics):
    summary.to_csv(OUT_SUMMARY)
    _out_words = paths.processed("x2_top_topics", snap).with_suffix(".csv")
    top_topics.to_csv(_out_words, index=False)
    print(f"wrote {OUT_SUMMARY.name}, {_out_words.name}")
    return


@app.cell
def _(OUT_FIGURE, VOICES, summary):
    # figure: outliers by voice, and language isolation, per arm
    # two panels with their own y-ranges; same voice colours as the other figures
    import matplotlib

    matplotlib.use("Agg")   # headless: no window, just the file
    import matplotlib.pyplot as plt

    COLOURS = {"host/en": "#2a78d6", "guest/en": "#eb6834", "guest/da": "#1baf7a"}
    _arms = list(summary.index)
    _x = range(len(_arms))
    _w = 0.26

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # panel 1: outlier share per voice
    for _k, _v in enumerate(VOICES):
        _vals = summary[f"outliers {_v} %"].tolist()
        _bars = _ax1.bar([i + (_k - 1) * _w for i in _x], _vals, width=_w, color=COLOURS[_v],
                         edgecolor="white", linewidth=2, label=_v)
        _ax1.bar_label(_bars, fmt="%.0f", fontsize=7, color="#333333", padding=2)
    _ax1.set_title("Sentences HDBSCAN could not place (%)", loc="left", fontsize=10)

    # panel 2: language isolation among guest sentences
    _iso = {"guest/en": "guest/en in English-only topics %",
            "guest/da": "guest/da in Danish-only topics %"}
    for _k, (_v, _col) in enumerate(_iso.items()):
        _vals = summary[_col].tolist()
        _bars = _ax2.bar([i + (_k - 0.5) * _w for i in _x], _vals, width=_w, color=COLOURS[_v],
                         edgecolor="white", linewidth=2, label=_v)
        _ax2.bar_label(_bars, fmt="%.0f", fontsize=7, color="#333333", padding=2)
    _ax2.set_title("Guest sentences in single-language topics (%)", loc="left", fontsize=10)

    for _ax in (_ax1, _ax2):
        _ax.set_xticks(list(_x))
        _ax.set_xticklabels(_arms, fontsize=8)
        _ax.legend(frameon=False, fontsize=8)
        _ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        _ax.set_axisbelow(True)
        for _s in ["top", "right"]:
            _ax.spines[_s].set_visible(False)
    _fig.tight_layout()
    _fig.savefig(OUT_FIGURE)
    print(f"wrote {OUT_FIGURE.name}")
    return


if __name__ == "__main__":
    app.run()
