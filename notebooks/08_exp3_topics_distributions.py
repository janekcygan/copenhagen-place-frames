import marimo

__generated_with = "0.24.2"
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

    snap = paths.snapshot("30 June, 2026")
    CORPUS_PATH = paths.interim("corpus", snap)                         # 05's output
    ASSIGN_PATH = paths.interim("discovery_assignments", snap)          # 07: ids + topics, no text
    MARKS_PATH = paths.processed("discovery_marks", snap).with_suffix(".csv")  # the filled-in marking sheet

    # All snapshots on disk, newest first. Only their listings files are read
    # (room type and price).
    SNAPSHOTS_NEWEST_FIRST = [paths.snapshot(n) for n in reversed(paths.available_snapshots())]

    VOICES = ["host/en", "guest/en", "guest/da"]

    # A group with fewer kept sentences gets no ranking (~4 per topic on average,
    # below which one listing can make a topic look distinctive).
    MIN_CELL = 100

    # Hexagons with a 400 m corner radius. Inside Airbnb moves listings up to ~150 m at
    # random, so much smaller hexagons would map the jitter.
    HEX_RADIUS_M = 400
    MIN_HEX = 30          # kept sentences a hexagon needs before it is coloured at all
    MIN_DOMINANT = 5      # sentences a topic needs in a hexagon before it can "win" it

    for _p in [CORPUS_PATH, ASSIGN_PATH, MARKS_PATH]:
        if not _p.exists():
            raise FileNotFoundError(f"{_p} is missing — run notebooks/05 and 07 first")

    import matplotlib

    matplotlib.use("Agg")   # headless: write files, open no window
    import matplotlib.pyplot as plt
    return (
        ASSIGN_PATH,
        CORPUS_PATH,
        HEX_RADIUS_M,
        MARKS_PATH,
        MIN_CELL,
        MIN_DOMINANT,
        MIN_HEX,
        SNAPSHOTS_NEWEST_FIRST,
        VOICES,
        mo,
        np,
        paths,
        pd,
        plt,
        snap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 08 Experiment 3: where do the kept discovery topics concentrate?

    This is a side experiment, and nothing later in the pipeline uses its output. It needs 07's
    marking sheet with the `frameless` column filled in.

    Of the 24 topics marked as not frame-less, which are the most common and which are the most
    distinctive when 07's pool is grouped by voice, by voice and area, by room type, by price
    (entire homes only) and by location on a hexagon map?

    Two numbers are given for each topic and group. The share is the fraction of the group's kept
    sentences that are in the topic, which answers what the group talks about most. It is
    dominated by the big general topics. The lift is the share in the group divided by the share
    that would be expected, which answers what the group talks about more than usual. A lift of
    2 means twice as often and 0.5 means half as often. The figures colour the lift on a log
    scale.

    The expected shares are computed within each voice. 07's pool took 8,000 sentences from each
    voice, which shrank each voice by a different factor, so the mix of areas is correct inside a
    voice but not across voices.

    Every group also reports `n` (kept sentences), `n_listings`, and `coverage`, the kept
    sentences as a share of all pool sentences in the group. The notebook only outputs counts,
    never sentence text.
    """)
    return


@app.cell
def _(
    ASSIGN_PATH,
    CORPUS_PATH,
    MARKS_PATH,
    SNAPSHOTS_NEWEST_FIRST,
    VOICES,
    pd,
):
    # 1. the pool, its topics and the marking
    assign = pd.read_parquet(ASSIGN_PATH)   # sentence_id, arm, neighbourhood, topic, strength

    marks = pd.read_csv(MARKS_PATH)
    _kept_marks = marks[marks["frameless"] == "no"].sort_values("size", ascending=False)
    KEPT_TOPICS = _kept_marks["topic"].tolist()   # biggest first: the row order of every table

    # label per topic: id + first three c-TF-IDF words
    TOPIC_LABEL = {}
    for _, _row in _kept_marks.iterrows():
        _words = [w.strip() for w in _row["c-TF-IDF words"].split(",")][:3]
        TOPIC_LABEL[_row["topic"]] = f"{_row['topic']}: {', '.join(_words)}"

    # 2. listing and coordinates from the corpus (no text column)
    _corpus = pd.read_parquet(
        CORPUS_PATH, columns=["sentence_id", "listing_id", "latitude", "longitude"]
    )
    pool = assign.merge(_corpus, on="sentence_id", how="left", validate="one_to_one")
    if pool["listing_id"].isna().any():
        raise ValueError("some pool ids are not in the corpus — 05 was re-run since 07")

    # 3. room type and price from the listings files
    # Room type from the newest snapshot. Price is missing for 30-40% of pool listings in
    # June; those get the most recent older snapshot's price, recorded in `price_snapshot`.
    def parse_price(s):
        """
        Turns '$1,159.00' into 1159.0 (Danish kroner per night, despite the dollar sign).

        astype("string") first, because a price column that is completely empty is read as numbers.
        """
        return pd.to_numeric(s.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce")

    _listing_info = None
    for _snap in SNAPSHOTS_NEWEST_FIRST:
        _l = pd.read_csv(_snap.listings, usecols=["id", "room_type", "price"])
        _l = _l.rename(columns={"id": "listing_id"})
        _l["price"] = parse_price(_l["price"])
        _l["price_snapshot"] = _snap.slug
        if _listing_info is None:
            _listing_info = _l   # newest snapshot
        else:
            # fill only missing prices, only for listings already known
            _older = _l.set_index("listing_id")
            _info = _listing_info.set_index("listing_id")
            _gap = _info["price"].isna() & _info.index.isin(_older.index)
            _gap_ids = _info.index[_gap]
            _found = _older.loc[_gap_ids, "price"].dropna()
            _info.loc[_found.index, "price"] = _found
            _info.loc[_found.index, "price_snapshot"] = _snap.slug
            _listing_info = _info.reset_index()
    _listing_info.loc[_listing_info["price"].isna(), "price_snapshot"] = pd.NA

    pool = pool.merge(_listing_info, on="listing_id", how="left", validate="many_to_one")
    if pool["room_type"].isna().any():
        raise ValueError("some pool listings have no room type in the newest snapshot")

    # 4. the kept sentences
    pool["is_kept"] = pool["topic"].isin(KEPT_TOPICS)
    kept = pool[pool["is_kept"]].copy()

    # expected: 24 kept topics, 9,450 sentences
    assert len(KEPT_TOPICS) == 24, len(KEPT_TOPICS)
    assert len(kept) == 9_450, len(kept)

    print(f"pool {len(pool):,} sentences · kept {len(kept):,} in {len(KEPT_TOPICS)} topics")
    _cov = pool.groupby("arm")["is_kept"].agg(["sum", "mean"]).reindex(VOICES)
    _cov.columns = ["kept", "coverage"]
    print(_cov.round(3).to_string())
    print("\nprice source (pool sentences):")
    print(pool["price_snapshot"].value_counts(dropna=False).to_string())
    return KEPT_TOPICS, TOPIC_LABEL, kept, pool


@app.cell
def _(KEPT_TOPICS, MIN_CELL, VOICES, np, pd):
    # Both functions return one row per group x topic:
    #   n_topic   sentences of this topic in the group (observed)
    #   n         kept sentences in the group;  n_listings  distinct listings
    #   coverage  kept / all pool sentences in the group
    #   share     n_topic / n
    #   expected  sentences of this topic a typical group of this size would have
    #   lift      n_topic / expected
    #   enough    n >= MIN_CELL
    # They differ only in what "typical" means.

    def prevalence(kept_rows, pool_rows, group_col):
        """Within each voice, a typical group has the voice's own topic shares. Used for voice and area."""
        rows = []
        for voice in VOICES:
            k_voice = kept_rows[kept_rows["arm"] == voice]
            p_voice = pool_rows[pool_rows["arm"] == voice]
            # reindex so a topic with zero sentences still gets a row
            baseline = k_voice["topic"].value_counts(normalize=True).reindex(KEPT_TOPICS, fill_value=0)

            for group, k_group in k_voice.groupby(group_col, observed=True):
                n = len(k_group)
                n_pool = int((p_voice[group_col] == group).sum())
                counts = k_group["topic"].value_counts().reindex(KEPT_TOPICS, fill_value=0)
                for topic in KEPT_TOPICS:
                    expected = n * baseline[topic]
                    rows.append({
                        "voice": voice, "group": group, "topic": topic,
                        "n_topic": int(counts[topic]), "n": n,
                        "n_listings": k_group["listing_id"].nunique(),
                        "coverage": n / n_pool if n_pool else np.nan,
                        "share": counts[topic] / n,
                        "expected": expected,
                        "lift": counts[topic] / expected if expected > 0 else np.nan,
                        "enough": n >= MIN_CELL,
                    })
        return pd.DataFrame(rows)

    def pooled_prevalence(kept_rows, pool_rows, group_col):
        """
        All voices together, with the typical group built from the group's own mix of voices:

            expected(topic, group) = sum over voices of
                                     (voice's sentences in the group x voice's share of the topic)

        Private rooms have more guest sentences than entire homes, so comparing with all voices
        pooled would make guest topics look distinctive only because of who writes there.
        """
        p_voice = (
            pd.crosstab(kept_rows["arm"], kept_rows["topic"], normalize="index")
            .reindex(index=VOICES, columns=KEPT_TOPICS, fill_value=0)
        )                                                     # p(topic | voice)
        rows = []
        for group, k_group in kept_rows.groupby(group_col, observed=True):
            n = len(k_group)
            n_pool = int((pool_rows[group_col] == group).sum())
            n_by_voice = k_group["arm"].value_counts().reindex(VOICES, fill_value=0)
            counts = k_group["topic"].value_counts().reindex(KEPT_TOPICS, fill_value=0)
            for topic in KEPT_TOPICS:
                expected = float((n_by_voice * p_voice[topic]).sum())
                rows.append({
                    "group": group, "topic": topic,
                    "n_topic": int(counts[topic]), "n": n,
                    "n_host": int(n_by_voice["host/en"]), "n_guest_en": int(n_by_voice["guest/en"]),
                    "n_guest_da": int(n_by_voice["guest/da"]),
                    "n_listings": k_group["listing_id"].nunique(),
                    "coverage": n / n_pool if n_pool else np.nan,
                    "share": counts[topic] / n,
                    "expected": expected,
                    "lift": counts[topic] / expected if expected > 0 else np.nan,
                    "enough": n >= MIN_CELL,
                })
        return pd.DataFrame(rows)

    return pooled_prevalence, prevalence


@app.cell
def _(KEPT_TOPICS, MIN_CELL, TOPIC_LABEL, np, pd, plt):
    # the table figure that every grouping uses
    # Per group: share (plain number) and lift (coloured, log2, clipped at 1/4x and 4x).
    # Blue is less than expected, red is more, and grey in the middle is no difference.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    LIFT_CMAP = LinearSegmentedColormap.from_list(
        "lift", ["#184f95", "#6da7ec", "#f0efec", "#ef8b8a", "#b52f2e"]
    )
    LIFT_NORM = Normalize(-2, 2)
    INK, MUTED, RARE_GREY = "#0b0b0b", "#52514e", "#e2e1dc"

    # With fewer than 2 expected sentences, 0 or 1 observed gives an extreme lift by
    # chance. Those cells show a dash on grey (the map uses the same rule).
    MIN_EXPECTED = 2

    def topic_table(long, groups, headers, title, path, note=""):
        """
        long    : table from prevalence()/pooled_prevalence(), with a `group` column
        groups  : column order; headers: the header text above each group's two columns
        Writes the figure to `path`.
        """
        topics = KEPT_TOPICS
        gap = 0.35                                   # space between groups
        step = 2 + gap
        fig_w = 3.6 + 1.25 * len(groups)
        fig_h = 1.9 + 0.27 * len(topics)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

        for i, t in enumerate(topics):
            ax.text(-0.2, i, TOPIC_LABEL[t], ha="right", va="center", fontsize=7, color=INK)
            if i % 2 == 0:   # zebra stripe
                ax.add_patch(Rectangle((-0.5, i - 0.5), step * len(groups) - gap + 0.5, 1,
                                       color="#f6f5f2", zorder=0, linewidth=0))

        for j, g in enumerate(groups):
            x0 = j * step
            d = long[long["group"] == g].set_index("topic").reindex(topics)
            enough = bool(d["enough"].iloc[0])
            ax.text(x0 + 1, -1.75, headers[j], ha="center", va="bottom", fontsize=7, color=INK)
            ax.text(x0 + 0.5, -0.9, "share", ha="center", va="center", fontsize=6, color=MUTED)
            ax.text(x0 + 1.5, -0.9, "lift", ha="center", va="center", fontsize=6, color=MUTED)
            if not enough:
                ax.text(x0 + 1, len(topics) / 2, f"too few\n(< {MIN_CELL}\nsentences)",
                        ha="center", va="center", fontsize=7, color=MUTED)
                continue
            for i, t in enumerate(topics):
                r = d.loc[t]
                ax.text(x0 + 0.5, i, f"{r['share'] * 100:.1f}%", ha="center", va="center",
                        fontsize=6.5, color=INK)
                if r["expected"] < MIN_EXPECTED or np.isnan(r["lift"]):
                    ax.add_patch(Rectangle((x0 + 1.04, i - 0.46), 0.92, 0.92, color=RARE_GREY, linewidth=0))
                    ax.text(x0 + 1.5, i, "–", ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                with np.errstate(divide="ignore"):
                    z = float(np.clip(np.log2(r["lift"]), -2, 2))   # log2(0) is minus infinity, which shows as 1/4x
                ax.add_patch(Rectangle((x0 + 1.04, i - 0.46), 0.92, 0.92,
                                       color=LIFT_CMAP(LIFT_NORM(z)), linewidth=0))
                ax.text(x0 + 1.5, i, f"{r['lift']:.1f}×", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(z) > 1.3 else INK)

        ax.set_xlim(-0.6, step * len(groups) - gap + 0.1)
        ax.set_ylim(len(topics) - 0.5, -3.4)   # room for title and headers
        ax.set_axis_off()
        fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK)

        # colour key and footnote
        cax = fig.add_axes([0.40, 0.055, 0.4, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=LIFT_NORM, cmap=LIFT_CMAP), cax=cax, orientation="horizontal")
        cb.set_ticks([-2, -1, 0, 1, 2])
        cb.set_ticklabels(["¼×", "½×", "as expected", "2×", "4×"])
        cb.ax.tick_params(labelsize=6, colors=MUTED, length=0)
        cb.outline.set_visible(False)
        fig.text(0.02, 0.0, "share = % of the group's kept sentences · lift = observed ÷ expected · "
                 f"– = fewer than {MIN_EXPECTED} expected, too little to tell" + (f"\n{note}" if note else ""),
                 fontsize=6, color=MUTED, ha="left", va="bottom")
        fig.subplots_adjust(left=0.30 if len(groups) <= 3 else 0.22, right=0.98, top=0.95, bottom=0.10)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def wide_table(long, groups, short_names):
        """The figure's numbers as a table: share % and lift per group."""
        out = pd.DataFrame({"topic": [TOPIC_LABEL[t] for t in KEPT_TOPICS]})
        for g, s in zip(groups, short_names):
            d = long[long["group"] == g].set_index("topic").reindex(KEPT_TOPICS)
            ok = d["enough"].to_numpy() & (d["expected"].to_numpy() >= MIN_EXPECTED)
            out[f"{s} share %"] = (d["share"] * 100).round(1).to_numpy()
            out[f"{s} lift"] = np.where(ok, d["lift"].round(2), np.nan)
        return out

    return INK, LIFT_CMAP, MIN_EXPECTED, MUTED, topic_table, wide_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. By voice

    For each voice, the share and lift of every topic. The expected count is the voice's number of
    kept sentences times the topic's share across the three voices together. That works here
    because the pool holds the three voices in equal thirds.
    """)
    return


@app.cell
def _(KEPT_TOPICS, VOICES, kept, np, paths, pd, pool, snap, topic_table, wide_table):
    _counts = pd.crosstab(kept["topic"], kept["arm"]).reindex(index=KEPT_TOPICS, columns=VOICES, fill_value=0)
    _share = _counts / _counts.sum()                     # each column sums to 1
    _pooled = _counts.sum(axis=1) / _counts.to_numpy().sum()

    assert np.allclose(_share.sum(), 1.0)

    by_voice = []
    for _v in VOICES:
        _n = int(_counts[_v].sum())
        for _t in KEPT_TOPICS:
            by_voice.append({
                "group": _v, "topic": _t, "n_topic": int(_counts.at[_t, _v]), "n": _n,
                "n_listings": kept.loc[kept["arm"] == _v, "listing_id"].nunique(),
                "coverage": _n / (pool["arm"] == _v).sum(),
                "share": _share.at[_t, _v], "pooled_share": _pooled[_t],
                "expected": _n * _pooled[_t], "lift": _counts.at[_t, _v] / (_n * _pooled[_t]),
                "enough": True,
            })
    by_voice = pd.DataFrame(by_voice)
    by_voice.to_csv(paths.processed("x3_by_voice", snap).with_suffix(".csv"), index=False)

    _headers = [f"{_v}\nn={int(_counts[_v].sum()):,}" for _v in VOICES]
    topic_table(by_voice, VOICES, _headers, "Kept topics by voice", paths.FIGURES / "x3_voice.png",
                note="expected = the voice's kept sentences × the topic's share in the three voices together")
    voice_table = wide_table(by_voice, VOICES, VOICES)
    print("wrote x3_voice.png")
    return (voice_table,)


@app.cell(hide_code=True)
def _(mo, voice_table):
    mo.ui.table(voice_table, selection=None, page_size=24)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Voice and area

    One table per area, built like section 1, but here the lift compares with that voice across
    the whole city. A lift of 10 for hosts in Amager Vest means hosts there use the topic ten times
    as often as hosts in general. A voice with fewer than 100 kept sentences in an area gets no
    numbers. The figures are saved in `figures/x3_area/`.
    """)
    return


@app.cell
def _(VOICES, kept, paths, pool, prevalence, snap, topic_table, wide_table):
    by_area = prevalence(kept, pool, "neighbourhood")

    # inside each voice, the areas' n add up to the voice's kept count
    for _v in VOICES:
        _n_areas = by_area[by_area["voice"] == _v].drop_duplicates("group")["n"].sum()
        assert _n_areas == (kept["arm"] == _v).sum(), _v

    by_area.to_csv(paths.processed("x3_by_voice_area", snap).with_suffix(".csv"), index=False)

    # one figure per area, with the three voices as the groups
    _dir = paths.FIGURES / "x3_area"
    _dir.mkdir(exist_ok=True)
    AREAS = pool["neighbourhood"].value_counts().index.tolist()   # biggest first
    area_tables = {}
    for _a in AREAS:
        _d = by_area[by_area["group"] == _a].drop(columns="group").rename(columns={"voice": "group"})
        _g = _d.drop_duplicates("group").set_index("group").reindex(VOICES)
        _headers = [f"{_v}\nn={int(_g.at[_v, 'n'])} · {int(_g.at[_v, 'n_listings'])} listings" for _v in VOICES]
        _slug = _a.lower().replace(" ", "_").replace("ø", "oe").replace("-", "_")
        topic_table(_d, VOICES, _headers, f"{_a} — kept topics by voice",
                    _dir / f"x3_area_{_slug}.png",
                    note="expected = the voice's sentences in this area × the voice's citywide share of the topic")
        area_tables[_a] = wide_table(_d, VOICES, VOICES)

    # most distinctive topic per voice x area, among readable cells
    _best = (
        by_area[by_area["enough"] & (by_area["expected"] >= 2)]
        .sort_values("lift", ascending=False)
        .groupby(["voice", "group"]).head(1)
        .sort_values(["group", "voice"])
    )
    print(_best[["group", "voice", "topic", "n_topic", "n", "n_listings", "share", "lift"]].round(3).to_string(index=False))
    print(f"wrote {len(AREAS)} figures to figures/x3_area/")
    return (area_tables,)


@app.cell(hide_code=True)
def _(area_tables, mo):
    mo.ui.tabs({_a: mo.ui.table(_t, selection=None, page_size=24) for _a, _t in area_tables.items()})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Room type, all voices together

    Entire homes against private rooms. Shared rooms (31 pool sentences) are left out. A guest in a
    private room usually stays with the person who lives there, so they might describe the place
    differently. The expected counts are built from each room type's own mix of voices
    (`pooled_prevalence`).
    """)
    return


@app.cell
def _(kept, mo, paths, pool, pooled_prevalence, snap, topic_table, wide_table):
    ROOMS = ["Entire home/apt", "Private room"]
    _k = kept[kept["room_type"].isin(ROOMS)]
    _p = pool[pool["room_type"].isin(ROOMS)]
    by_room = pooled_prevalence(_k, _p, "room_type")
    by_room.to_csv(paths.processed("x3_by_room", snap).with_suffix(".csv"), index=False)

    _g = by_room.drop_duplicates("group").set_index("group").reindex(ROOMS)
    _headers = [
        f"{_r}\nn={int(_g.at[_r, 'n']):,} · {int(_g.at[_r, 'n_listings']):,} listings\n"
        f"host {int(_g.at[_r, 'n_host'])} · en {int(_g.at[_r, 'n_guest_en'])} · da {int(_g.at[_r, 'n_guest_da'])}"
        for _r in ROOMS
    ]
    topic_table(by_room, ROOMS, _headers, "Kept topics by room type, all voices",
                paths.FIGURES / "x3_room.png",
                note="expected = built from each room type's own voice mix (host / guest-en / guest-da)")
    print(_g[["n", "n_host", "n_guest_en", "n_guest_da", "n_listings", "coverage"]].round(3).to_string())
    print("wrote x3_room.png")
    mo.ui.table(wide_table(by_room, ROOMS, ["entire", "private"]), selection=None, page_size=24)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Price, all voices together (entire homes only)

    This is a side observation, not one of the research questions. The voices are pooled with the
    same baseline as in section 3. Only entire homes are used, since otherwise price would partly
    measure room type again. The quartiles are taken over listings rather than sentences, so a
    listing with many sentences does not shift them. Missing prices are filled in from older
    snapshots where possible, recorded in `price_snapshot`, and prices that are still missing are
    reported rather than guessed.
    """)
    return


@app.cell
def _(kept, mo, paths, pd, pool, pooled_prevalence, snap, topic_table, wide_table):
    _entire = pool[pool["room_type"] == "Entire home/apt"]
    print(f"entire-home pool sentences without any price: {_entire['price'].isna().mean():.1%}")

    # one row per listing, then quartile cut-points
    _listings = _entire.dropna(subset=["price"]).drop_duplicates("listing_id")
    _cuts = _listings["price"].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy()
    PRICE_BANDS = [
        f"Q1 ≤{_cuts[1]:,.0f}", f"Q2 {_cuts[1]:,.0f}–{_cuts[2]:,.0f}",
        f"Q3 {_cuts[2]:,.0f}–{_cuts[3]:,.0f}", f"Q4 >{_cuts[3]:,.0f}",
    ]
    print(f"price quartiles over {len(_listings):,} entire-home listings (DKK/night): {PRICE_BANDS}")

    _p = _entire.dropna(subset=["price"]).copy()
    # include_lowest so the cheapest listing lands in Q1
    _p["price_band"] = pd.cut(_p["price"], bins=_cuts, labels=PRICE_BANDS, include_lowest=True)
    _k = _p[_p["is_kept"]]
    by_price = pooled_prevalence(_k, _p, "price_band")
    by_price.to_csv(paths.processed("x3_by_price", snap).with_suffix(".csv"), index=False)

    _g = by_price.drop_duplicates("group").set_index("group").reindex(PRICE_BANDS)
    _headers = [
        f"{_b}\nn={int(_g.at[_b, 'n']):,}\nhost {int(_g.at[_b, 'n_host'])} · en {int(_g.at[_b, 'n_guest_en'])}"
        f" · da {int(_g.at[_b, 'n_guest_da'])}"
        for _b in PRICE_BANDS
    ]
    topic_table(by_price, PRICE_BANDS, _headers, "Kept topics by nightly price quartile (DKK), entire homes, all voices",
                paths.FIGURES / "x3_price.png",
                note="expected = built from each quartile's own voice mix (host / guest-en / guest-da)")
    print(_g[["n", "n_host", "n_guest_en", "n_guest_da", "n_listings", "coverage"]].round(3).to_string())
    print("wrote x3_price.png")
    mo.ui.table(wide_table(by_price, PRICE_BANDS, ["Q1", "Q2", "Q3", "Q4"]), selection=None, page_size=24)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The map

    Each sentence is placed at its listing's coordinates and counted into hexagons. Hexagons are
    used because all six neighbours are the same distance away.

    Each hexagon's expected count for a topic is built from its own mix of voices. Otherwise a
    voice's topics would light up wherever that voice happens to be common. The expected count is
    the sum, over the three voices, of that voice's sentences in the hexagon times that voice's
    overall share of the topic, and the lift is the observed count divided by the expected one.

    Map 5a shows the topic with the highest lift in each hexagon. The 8 topics that win the most
    hexagons get a colour and the others are grey, and the topic number is written in every
    hexagon. Map 5b has one small map per topic, with every hexagon coloured by that topic's lift.

    Hexagons with fewer than 30 kept sentences stay blank, and a topic needs at least 5 sentences
    in a hexagon to win it.
    """)
    return


@app.cell
def _(HEX_RADIUS_M, KEPT_TOPICS, MIN_DOMINANT, MIN_HEX, VOICES, kept, np, pd):
    import geopandas as gpd
    from shapely.geometry import Polygon

    # degrees to metres: EPSG:25832 (UTM 32N), the metric system used for Copenhagen
    _pts = gpd.GeoSeries(gpd.points_from_xy(kept["longitude"], kept["latitude"]), crs="EPSG:4326").to_crs(25832)
    _x, _y = _pts.x.to_numpy(), _pts.y.to_numpy()

    # point to hexagon: axial coordinates for pointy-top hexagons (Red Blob Games),
    # with cube rounding (q + r + s = 0) so points near the corners land correctly.
    _R = HEX_RADIUS_M
    _qf = (np.sqrt(3) / 3 * _x - _y / 3) / _R
    _rf = (2 / 3 * _y) / _R
    _sf = -_qf - _rf
    _q, _r, _s = np.round(_qf), np.round(_rf), np.round(_sf)
    _dq, _dr, _ds = np.abs(_q - _qf), np.abs(_r - _rf), np.abs(_s - _sf)
    _fix_q = (_dq > _dr) & (_dq > _ds)
    _fix_r = ~_fix_q & (_dr > _ds)
    _q[_fix_q] = -_r[_fix_q] - _s[_fix_q]
    _r[_fix_r] = -_q[_fix_r] - _s[_fix_r]

    hexed = kept[["sentence_id", "arm", "topic", "listing_id"]].copy()
    hexed["hex_q"], hexed["hex_r"] = _q.astype(int), _r.astype(int)
    hexed["hex_id"] = hexed["hex_q"].astype(str) + "_" + hexed["hex_r"].astype(str)

    # expected counts from each hexagon's own mix of voices
    _voice_share = (
        pd.crosstab(kept["arm"], kept["topic"], normalize="index")
        .reindex(index=VOICES, columns=KEPT_TOPICS, fill_value=0)
    )                                                       # p(topic | voice)
    _voice_n = pd.crosstab(hexed["hex_id"], hexed["arm"]).reindex(columns=VOICES, fill_value=0)
    _expected = _voice_n @ _voice_share                     # hex x topic: sum_v n(v,hex) * p(t|v)
    _observed = (
        pd.crosstab(hexed["hex_id"], hexed["topic"])
        .reindex(index=_expected.index, columns=KEPT_TOPICS, fill_value=0)
    )
    _n = _voice_n.sum(axis=1)
    # expected must add up to the observed total in every hexagon
    assert np.allclose(_expected.sum(axis=1), _n)

    _lift = _observed / _expected
    _hex_listings = hexed.groupby("hex_id")["listing_id"].nunique()

    # one row per hexagon x topic; centres in metres (grid points, not listing locations)
    _rows = []
    for _h in _expected.index:
        _hq, _hr = map(int, _h.split("_"))
        _cx = _R * np.sqrt(3) * (_hq + _hr / 2)
        _cy = _R * 1.5 * _hr
        for _t in KEPT_TOPICS:
            _rows.append({
                "hex_id": _h, "x_25832": round(_cx), "y_25832": round(_cy),
                "n": int(_n[_h]), "n_listings": int(_hex_listings[_h]),
                "n_host": int(_voice_n.at[_h, "host/en"]), "n_guest_en": int(_voice_n.at[_h, "guest/en"]),
                "n_guest_da": int(_voice_n.at[_h, "guest/da"]),
                "topic": _t, "observed": int(_observed.at[_h, _t]),
                "expected": round(float(_expected.at[_h, _t]), 3), "lift": float(_lift.at[_h, _t]),
                "enough": _n[_h] >= MIN_HEX,
            })
    by_hex = pd.DataFrame(_rows)

    # 5a: per hexagon, the highest-lift topic among those with >= MIN_DOMINANT sentences
    dominant = (
        by_hex[by_hex["enough"] & (by_hex["observed"] >= MIN_DOMINANT)]
        .sort_values("lift", ascending=False)
        .groupby("hex_id").head(1)
        .reset_index(drop=True)
    )

    def hex_polygon(cx, cy, radius):
        """Pointy-top hexagon around (cx, cy): corners at 30°, 90°, ..., 330°."""
        angles = np.deg2rad(np.arange(30, 390, 60))
        return Polygon(zip(cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

    _hexes = by_hex.drop_duplicates("hex_id")[["hex_id", "x_25832", "y_25832", "n", "enough"]]
    hex_gdf = gpd.GeoDataFrame(
        _hexes,
        geometry=[hex_polygon(cx, cy, _R) for cx, cy in zip(_hexes["x_25832"], _hexes["y_25832"])],
        crs=25832,
    )
    print(f"{len(hex_gdf)} hexagons hold kept sentences; {int(hex_gdf['enough'].sum())} have ≥ {MIN_HEX} "
          f"({int(_n[_n >= MIN_HEX].sum()):,} of {len(kept):,} sentences); {len(dominant)} have a winner")
    print("\nhow often each topic wins a hexagon:")
    print(dominant["topic"].value_counts().to_string())
    return by_hex, dominant, gpd, hex_gdf


@app.cell
def _(
    INK,
    KEPT_TOPICS,
    LIFT_CMAP,
    MIN_EXPECTED,
    MUTED,
    TOPIC_LABEL,
    by_hex,
    dominant,
    gpd,
    hex_gdf,
    np,
    paths,
    plt,
    snap,
):
    _areas = gpd.read_file(snap.neighbourhoods_geojson).to_crs(25832)
    _xmin, _ymin, _xmax, _ymax = hex_gdf[hex_gdf["enough"]].total_bounds
    _pad = 800

    def _frame(ax):
        _areas.boundary.plot(ax=ax, color="#9a9994", linewidth=0.5, zorder=3)
        ax.set_xlim(_xmin - _pad, _xmax + _pad)
        ax.set_ylim(_ymin - _pad, _ymax + _pad)
        ax.set_axis_off()
        ax.set_aspect("equal")

    # 5a: winning topic per hexagon. 23 topics win somewhere, too many to tell apart by
    # colour, so the 8 that win most hexagons (ties: summed lift) get the categorical
    # palette and the rest are grey. The id is written in every hexagon.
    from matplotlib.patches import Patch

    _CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    _DARK_FILLS = {"#2a78d6", "#008300", "#4a3aa7", "#e34948"}   # white text on these
    _OTHER_GREY = "#d6d5d0"

    _rank = (
        dominant.groupby("topic")
        .agg(hexes=("hex_id", "size"), strength=("lift", "sum"))
        .sort_values(["hexes", "strength"], ascending=False)
    )
    _coloured = _rank.index[: len(_CATEGORICAL)].tolist()
    TOPIC_COLOUR = dict(zip(_coloured, _CATEGORICAL))

    _win = hex_gdf.merge(dominant[["hex_id", "topic", "lift", "observed"]], on="hex_id")
    _win["fill"] = [TOPIC_COLOUR.get(_t, _OTHER_GREY) for _t in _win["topic"]]

    _fig, _ax = plt.subplots(figsize=(10, 10), dpi=150)
    hex_gdf[hex_gdf["enough"]].plot(ax=_ax, color="#f0efec", edgecolor="#ffffff", linewidth=1.5, zorder=1)
    _win.plot(ax=_ax, color=_win["fill"], edgecolor="#ffffff", linewidth=1.5, zorder=2)
    for _, _h in _win.iterrows():
        _c = _h.geometry.centroid
        _ax.text(_c.x, _c.y, str(_h["topic"]), ha="center", va="center", fontsize=6.5, zorder=4,
                 color="#ffffff" if _h["fill"] in _DARK_FILLS else INK)
    _frame(_ax)

    # key to the right of the map: coloured topics, then the grey ones by id
    _handles = [Patch(facecolor=TOPIC_COLOUR[_t], edgecolor="none",
                      label=f"{TOPIC_LABEL[_t]}  ({int(_rank.at[_t, 'hexes'])})") for _t in _coloured]
    _others = [_t for _t in _rank.index if _t not in TOPIC_COLOUR]
    _handles.append(Patch(facecolor=_OTHER_GREY, edgecolor="none",
                          label="other winning topics (id in the hexagon):"))
    _leg = _ax.legend(handles=_handles, loc="upper left", bbox_to_anchor=(1.01, 0.98), frameon=False,
                      fontsize=7, title="topic (hexagons won)", title_fontsize=7.5, alignment="left")
    for _txt in _leg.get_texts():
        _txt.set_color(INK)
    _ax.text(1.03, 0.98 - 0.032 * (len(_handles) + 1.6),
             "\n".join(f"{TOPIC_LABEL[_t]}  ({int(_rank.at[_t, 'hexes'])})" for _t in _others),
             transform=_ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)
    _ax.set_title("Most over-represented kept topic per hexagon (≥ 30 sentences; the topic ≥ 5 of them)\n"
                  "Lift against each hexagon's own voice mix; three voices pooled (equal thirds by design). "
                  "How strong each win is: figure 5b and x3_by_hex.csv.", loc="left", fontsize=8.5, color=INK)
    _fig.savefig(paths.FIGURES / "x3_map_dominant.png", bbox_inches="tight")
    plt.close(_fig)

    # 5b: one panel per kept topic, same scale in every panel
    _ncol = 6
    _nrow = int(np.ceil(len(KEPT_TOPICS) / _ncol))
    _fig, _axes = plt.subplots(_nrow, _ncol, figsize=(18, 3.6 * _nrow), dpi=130)
    for _ax, _t in zip(_axes.flat, KEPT_TOPICS):
        _d = hex_gdf.merge(by_hex[by_hex["topic"] == _t][["hex_id", "lift", "observed", "expected"]], on="hex_id")
        _d = _d[_d["enough"]]
        # fewer than 2 expected sentences: grey, "too little to tell"
        _readable = _d["expected"] >= MIN_EXPECTED
        if (~_readable).any():
            _d[~_readable].plot(ax=_ax, color="#e2e1dc", edgecolor="#ffffff", linewidth=0.4)
        if _readable.any():
            with np.errstate(divide="ignore"):
                _z = np.clip(np.log2(_d.loc[_readable, "lift"]), -2, 2)
            _d[_readable].plot(ax=_ax, column=_z, cmap=LIFT_CMAP, vmin=-2, vmax=2,
                               edgecolor="#ffffff", linewidth=0.4)
        _frame(_ax)
        _short = TOPIC_LABEL[_t] if len(TOPIC_LABEL[_t]) <= 36 else TOPIC_LABEL[_t][:35] + "…"
        _ax.set_title(f"{_short}\nn={int(_d['observed'].sum())} · readable hexagons: {int(_readable.sum())}",
                      fontsize=6.5, color=INK, loc="left")
    for _ax in list(_axes.flat)[len(KEPT_TOPICS):]:
        _ax.set_axis_off()
    _sm = plt.cm.ScalarMappable(cmap=LIFT_CMAP, norm=plt.Normalize(-2, 2))
    _cb = _fig.colorbar(_sm, ax=_axes, shrink=0.3, pad=0.01, aspect=30)
    _cb.set_ticks([-2, -1, 0, 1, 2])
    _cb.set_ticklabels(["¼×", "½×", "as expected", "2×", "4×"])
    _cb.ax.tick_params(labelsize=7, colors=MUTED)
    _cb.outline.set_visible(False)
    _fig.suptitle("Each kept topic's lift per hexagon (≥ 30 sentences), against the hexagon's own voice mix. "
                  "Grey: fewer than 2 sentences expected, too little to tell.",
                  x=0.01, ha="left", fontsize=11)
    _fig.savefig(paths.FIGURES / "x3_map_topics.png", bbox_inches="tight")
    plt.close(_fig)
    print("wrote x3_map_dominant.png, x3_map_topics.png")
    return


@app.cell
def _(by_hex, paths, snap):
    # counts per grid cell only
    by_hex.to_csv(paths.processed("x3_by_hex", snap).with_suffix(".csv"), index=False)
    print("wrote x3_by_voice, x3_by_voice_area, x3_by_room, x3_by_price, x3_by_hex (.csv)")
    return


if __name__ == "__main__":
    app.run()
