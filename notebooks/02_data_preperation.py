import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 Preparing the corpus

    This notebook turns the raw 30 June 2026 snapshot into one sentence-level table that holds
    all three voices, and saves it as `data/interim/sentences_voices_2026-06-30.parquet`.

    The host voice comes from `listings.description`, the marketing text written to sell a stay.
    The two guest voices come from `reviews.comments`, split into English and Danish reviews
    written after a stay.

    Nothing here filters for place. That happens in `03_place_filter.py`, kept separate so the
    filter can be re-run without parsing all 471,532 reviews again.
    """)
    return


@app.cell
def _():
    # imports and paths
    import html
    import re
    import sys
    from pathlib import Path

    import fasttext
    import marimo as mo
    import matplotlib.pyplot as plt
    import nltk
    import pandas as pd
    from nltk.tokenize import sent_tokenize
    from tqdm import tqdm

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths

    snap = paths.snapshot("30 June, 2026")
    if not snap.exists():
        raise FileNotFoundError(f"missing from {snap.dir}: {snap.missing()}")

    paths.ensure_dirs()

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print(f"snapshot: {snap} -> {snap.dir}")
    return fasttext, html, mo, paths, pd, plt, re, sent_tokenize, snap, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clean, but do not reduce

    Cleaning means repairing what is wrong in the data: platform HTML, zero-width characters,
    neighbourhood names with their Danish letters stripped, empty rows. Reducing would mean
    throwing away information a later method could use, such as stopwords, punctuation, case
    or word endings. This notebook only cleans.

    There are two reasons. BERTopic works on a sentence encoder, which expects natural
    sentences. And the place filter relies on small function words: "just around the corner
    from the lakes", "a bit far from the centre". Removing stopwords would make "far from the
    centre" and "in the centre" look the same. Denny & Spirling (2018) show that preprocessing
    choices change unsupervised results, so they belong to the method.

    Keeping only ASCII characters, as `str.replace("[^ -~]", "")` does, would turn `Nørrebro`
    into `Nrrebro`, and only in the Danish half of the corpus. Text in other languages is
    handled in step 6 instead, by language identification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps 1 and 2: load, and count before dropping
    """)
    return


@app.cell
def _(pd, snap):
    # reviews.csv.gz, not reviews.csv: the summary file has no `comments` column.
    # Dates parsed at load so `.dt.year` works and parquet stores them as datetimes.
    reviews_raw = pd.read_csv(snap.reviews, parse_dates=["date"])
    print(f"reviews:  {reviews_raw.shape[0]:,} rows x {reviews_raw.shape[1]} cols")
    return (reviews_raw,)


@app.cell
def _(pd, snap):
    LISTINGS_COLS = [
        "id",                      # join key to reviews.listing_id
        "description",             # the host voice
        "neighbourhood_cleansed",  # the only usable place column
        "latitude",                # jittered by Inside Airbnb
        "longitude",
        "host_id",
        "source",                  # flags `previous scrape` listings
        "first_review",            # proxy for when a listing entered the market (step 9c)
    ]
    # neighborhood_overview, neighbourhood and neighbourhood_group_cleansed are 100% empty.
    listings_raw = pd.read_csv(
        snap.listings, usecols=LISTINGS_COLS, parse_dates=["first_review"]
    )
    print(f"listings: {listings_raw.shape[0]:,} rows x {listings_raw.shape[1]} cols")
    return (listings_raw,)


@app.cell
def _(listings_raw, reviews_raw):
    # Work on copies: marimo does not re-run a cell when an object is mutated in place.
    reviews = reviews_raw.copy()
    listings = listings_raw.copy()
    return listings, reviews


@app.cell
def _(listings, reviews):
    # Null vs blank. Counted here, dropped after cleaning, because cleaning creates
    # new empties (a description that is only `<br /><br />`).
    def blank_counts(frame, column):
        """Null vs null-or-blank for one text column. Returns a dict, prints nothing."""
        n_null = int(frame[column].isna().sum())
        n_blank = int((frame[column].fillna("").str.strip() == "").sum())
        return {
            "rows": len(frame),
            "null": n_null,
            "null_or_blank": n_blank,
            "blank_but_not_null": n_blank - n_null,
            "usable": len(frame) - n_blank,
        }

    load_counts = {
        "reviews.comments": blank_counts(reviews, "comments"),
        "listings.description": blank_counts(listings, "description"),
    }
    for _name, _counts in load_counts.items():
        print(f"{_name}:")
        for _k, _v in _counts.items():
            print(f"    {_k:>20}: {_v:>9,}")
    return (load_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: clean the text

    The markup is different in the two sources (measured in `01_initial_exploration.py`).
    About 24% of the 471,532 reviews contain markup, almost all of it `<br/>` line breaks.
    In the 23,144 host descriptions it is 60%: line breaks written as `<br />`, a few hundred
    list tags (`<li>`, `<ul>`) and 260 links.

    The cleaner names the tags it expects rather than matching anything shaped like a tag. A
    general tag pattern with a short length limit missed the long `<a href=...>` tags, and with
    a long limit it swallowed real text after a stray `<`.

    Block tags such as `<br>` and `<li>` become a newline, so they still mark where a sentence
    ends. Inline tags such as `<b>` and `<a>` are removed and their text is kept.
    """)
    return


@app.cell
def _(html, re):
    # Block-level tags mark a boundary, so they become a newline.
    BLOCK_RE = re.compile(r"</?(br|li|ul|ol|p|div|tr)\s*/?>", flags=re.IGNORECASE)
    # Inline tags are emphasis and links. Remove the tag, keep the text.
    INLINE_RE = re.compile(r"</?(a|b|i|em|strong|span|u)\b[^>]{0,300}>", flags=re.IGNORECASE)
    # Zero-width space / non-joiner / joiner / BOM.
    ZWSP_RE = re.compile("[​‌‍﻿]")
    # Anything tag-shaped the patterns above did not handle.
    LEFTOVER_RE = re.compile(r"<[a-zA-Z/!][^>]{0,300}>")
    # Airbnb's help-centre link, inserted into the host's field by the platform.
    PLATFORM_LINK_RE = re.compile(r"airbnb\.com/help/article/3866", flags=re.IGNORECASE)
    # Not a raw string: pandas' Arrow backend (RE2) rejects `\U` escapes.
    EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")

    def clean_text(text: str) -> str:
        """Repair platform damage in one string. Boundaries become newlines first."""
        if not isinstance(text, str):
            return ""
        text = BLOCK_RE.sub("\n", text)
        text = INLINE_RE.sub("", text)
        text = ZWSP_RE.sub("", text)
        text = html.unescape(text)        # turns &amp; into & and &#39; into '
        # Collapse spaces and blank lines, but keep single newlines for step 7.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    return EMOJI_RE, LEFTOVER_RE, PLATFORM_LINK_RE, clean_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zero-width characters

    Some host descriptions contain the zero-width space (U+200B), which is invisible when
    printed. `.split()` does not split on it, so `in​Vesterbro` becomes a single token and a
    token match for `Vesterbro` fails, although a `\b` regex would still find it. Removing the
    character once, here, means no later step has to think about it.
    """)
    return


@app.cell
def _(clean_text, reviews):
    reviews_clean = reviews.assign(text=reviews["comments"].map(clean_text))

    # Drop the empties now, after cleaning.
    _before = len(reviews_clean)
    reviews_clean = reviews_clean[reviews_clean["text"].str.strip() != ""].copy()
    reviews_clean = reviews_clean.reset_index(drop=True)

    print(f"reviews in:      {_before:,}")
    print(f"reviews kept:    {len(reviews_clean):,}")
    print(f"dropped (empty): {_before - len(reviews_clean):,}")
    return (reviews_clean,)


@app.cell
def _(EMOJI_RE, PLATFORM_LINK_RE, clean_text, listings):
    listings_clean = listings.assign(
        text=listings["description"].map(clean_text),
    )

    # All 260 links point at the same Airbnb help article: platform text inside the
    # host field. Flagged, not stripped.
    listings_clean["has_platform_link"] = (
        listings["description"].fillna("").map(lambda t: bool(PLATFORM_LINK_RE.search(t)))
    )
    # Emoji are kept; in marketing copy they are part of the register.
    listings_clean["has_emoji"] = listings_clean["text"].map(
        lambda t: bool(EMOJI_RE.search(t))
    )

    # Airbnb's auto-generated placeholder descriptions ("Keep it simple at this
    # peaceful and centrally-located place."). A text is a template if it is short AND
    # used by more than one host. Duplicate descriptions across hosts are either
    # <= 87 or >= 240 characters, so any cutoff between 88 and 239 gives the same rows.
    # Flagged, not dropped.
    TEMPLATE_MAX_CHARS = 150
    _nonblank = listings_clean["text"].str.strip() != ""
    _hosts_per_text = listings_clean[_nonblank].groupby("text")["host_id"].nunique()
    _templates = {
        text
        for text, n_hosts in _hosts_per_text.items()
        if n_hosts > 1 and len(text) < TEMPLATE_MAX_CHARS
    }
    listings_clean["has_platform_template"] = (
        listings_clean["text"].isin(_templates) & _nonblank
    )

    print(f"listings with a platform link: {listings_clean['has_platform_link'].sum():,}"
          f" ({listings_clean['has_platform_link'].mean():.1%})")
    print(f"platform-template descriptions: {listings_clean['has_platform_template'].sum():,}"
          f" ({listings_clean['has_platform_template'].mean():.1%})"
          f" from {len(_templates)} distinct templates")
    print(f"listings with an emoji:        {listings_clean['has_emoji'].sum():,}"
          f" ({listings_clean['has_emoji'].mean():.1%})")
    return (listings_clean,)


@app.cell
def _(LEFTOVER_RE, listings_clean, reviews_clean):
    # Check for markup the cleaner did not handle. Printed rather than asserted:
    # "<3" or "rooms <30m2" are real text.
    def leftover_markup(series, label):
        residue = series.map(lambda t: LEFTOVER_RE.findall(t)).explode().dropna()
        if len(residue):
            print(f"!! {label}: {len(residue):,} unexpected tag-like strings")
            print(residue.value_counts().head(10).to_string())
        else:
            print(f"ok {label}: no markup left")
        return residue

    _r = leftover_markup(reviews_clean["text"], "reviews")
    _h = leftover_markup(listings_clean["text"], "host descriptions")
    return


@app.cell
def _(listings_clean):
    # Inside Airbnb deletes ø/æ/å from `neighbourhood_cleansed` ("Nrrebro", "sterbro").
    # The review text itself is intact. Same 11 names in all four snapshots.
    NEIGHBOURHOOD_REPAIR = {
        "Nrrebro": "Nørrebro",
        "sterbro": "Østerbro",
        "Vanlse": "Vanløse",
        "Brnshj-Husum": "Brønshøj-Husum",
        "Amager st": "Amager Øst",
    }
    # neighbourhoods.geojson has the same broken names, so the original column is kept
    # as the join key and `neighbourhood` holds the repaired name for reading.
    listings_prepared = listings_clean.assign(
        neighbourhood=listings_clean["neighbourhood_cleansed"].replace(NEIGHBOURHOOD_REPAIR)
    )

    # Exact match against the broken names, in case a future snapshot changes them.
    _unrepaired = sorted(
        set(NEIGHBOURHOOD_REPAIR) & set(listings_prepared["neighbourhood"].dropna())
    )
    print(f"areas: {listings_prepared['neighbourhood'].nunique()}")
    print(sorted(listings_prepared["neighbourhood"].dropna().unique()))
    if _unrepaired:
        print(f"!! still ASCII-stripped: {_unrepaired}")
    return (listings_prepared,)


@app.cell
def _(listings_prepared):
    # The host corpus: listings with a description. `listings_prepared` stays complete,
    # because the join in step 4 needs geography for every listing.
    hosts = listings_prepared[listings_prepared["text"].str.strip() != ""].copy()
    hosts = hosts.reset_index(drop=True)
    print(f"listings total:            {len(listings_prepared):,}")
    print(f"host corpus (has text):    {len(hosts):,}")
    print(f"dropped (no description):  {len(listings_prepared) - len(hosts):,}")
    return (hosts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: join reviews to listings
    """)
    return


@app.cell
def _(listings_prepared, reviews_clean):
    # Rename `id` on both sides first, so pandas does not create id_x / id_y.
    _geo = listings_prepared[
        [
            "id",
            "neighbourhood",
            "neighbourhood_cleansed",
            "latitude",
            "longitude",
            "host_id",
            "source",
        ]
    ].rename(columns={"id": "listing_id"})

    joined = reviews_clean.rename(columns={"id": "review_id"}).merge(
        _geo,
        on="listing_id",
        how="inner",
        # raises if a listing_id appears twice, which would duplicate its reviews
        validate="many_to_one",
    )

    # Reviews of listings no longer in the snapshot are lost here (survivorship).
    print(f"reviews before join: {len(reviews_clean):,}")
    print(f"reviews after join:  {len(joined):,}")
    print(f"lost (delisted):     {len(reviews_clean) - len(joined):,}"
          f" ({1 - len(joined) / len(reviews_clean):.2%})")
    return (joined,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: duplicates

    A repeated `review_id` would be an error in the snapshot, so it is checked.

    The more important case is host descriptions. A host with six listings may paste the same
    text into all of them, which turns one person's phrasing into six data points. This is only
    measured here. The duplicates are removed in `05_corpora.py`.
    """)
    return


@app.cell
def _(hosts, joined):
    _dup_reviews = int(joined["review_id"].duplicated().sum())
    print(f"duplicate review_ids: {_dup_reviews:,}")

    # keep=False marks every member of a duplicate group, including the first.
    _dup_mask = hosts["text"].duplicated(keep=False)
    print(f"non-unique descriptions:   {_dup_mask.sum():,} of {len(hosts):,}"
          f" ({_dup_mask.mean():.1%})")
    print(f"  involving distinct hosts: {hosts.loc[_dup_mask, 'host_id'].nunique():,}")
    print(f"  distinct texts overall:   {hosts['text'].nunique():,}")

    # Number and size of the duplicate groups.
    _counts = hosts["text"].value_counts()
    _groups = _counts[_counts > 1]
    print(f"  duplicate groups:         {len(_groups):,}")
    print(f"  redundant rows:           {int((_groups - 1).sum()):,}")
    print(f"  largest group:            {int(_groups.max()):,} listings share one text")
    print("  group sizes (size -> how many groups):")
    print("   ", _groups.value_counts().sort_index().to_dict())

    # Reuse within one host vs. the same text from unrelated hosts.
    _group_hosts = hosts[_dup_mask].groupby("text")["host_id"].nunique()
    print(f"  groups within ONE host:   {int((_group_hosts == 1).sum()):,}")
    print(f"  groups ACROSS hosts:      {int((_group_hosts > 1).sum()):,}")

    _within_host_dup = hosts.duplicated(subset=["host_id", "text"], keep=False)
    print(f"listings carrying copy reused by their own host: {_within_host_dup.sum():,}")
    print(f"  hosts doing it: {hosts.loc[_within_host_dup, 'host_id'].nunique():,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: language identification

    This uses the fastText `lid.176` model, which `scripts/download_data.py` downloads.

    The model is unreliable on very short texts, so anything under `MIN_TOKENS_FOR_LID` tokens
    is labelled `too-short` instead of being given a guess. Danish and Norwegian are easily
    confused, and since Danish is only about 5% of the corpus, a small error rate matters more
    for it than for English. fastText also stops reading at a newline, so newlines are replaced
    by spaces for the language check only. The stored text keeps them.
    """)
    return


@app.cell
def _(fasttext, paths):
    MIN_TOKENS_FOR_LID = 5

    # fastText prints a harmless deprecation warning on load.
    _lid = fasttext.load_model(str(paths.LID_MODEL))

    def identify_language(texts):
        """Label a list of strings with fastText lid.176. Returns labels, confidences and token counts."""
        flat = [" ".join(t.split()) for t in texts]
        n_tokens = [len(t.split()) for t in flat]
        raw_labels, raw_scores = _lid.predict(flat, k=1)
        labels = [lab[0].replace("__label__", "") for lab in raw_labels]
        confs = [float(s[0]) for s in raw_scores]
        labels = [
            "too-short" if n < MIN_TOKENS_FOR_LID else lab
            for lab, n in zip(labels, n_tokens)
        ]
        return labels, confs, n_tokens

    return (identify_language,)


@app.cell
def _(hosts, identify_language, joined, pd):
    _g_lang, _g_conf, _g_tok = identify_language(joined["text"].tolist())
    guest_langed = joined.assign(lang=_g_lang, lang_conf=_g_conf, n_tokens_approx=_g_tok)

    # Host descriptions too: the sentence splitter needs the language.
    _h_lang, _h_conf, _h_tok = identify_language(hosts["text"].tolist())
    host_langed = hosts.assign(lang=_h_lang, lang_conf=_h_conf, n_tokens_approx=_h_tok)

    _composition = pd.concat(
        [
            guest_langed["lang"].value_counts().rename("guest reviews"),
            host_langed["lang"].value_counts().rename("host descriptions"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    _composition["guest %"] = (_composition["guest reviews"] / len(guest_langed) * 100).round(2)
    _composition["host %"] = (_composition["host descriptions"] / len(host_langed) * 100).round(2)
    print(_composition.head(12).to_string())
    return guest_langed, host_langed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: split into sentences

    The sentence is the unit of analysis. One review can say both "the flat was spotless" and
    "Nørrebro is lively but noisy at night", and only the second is about the place.

    Danish text is split with the Danish punkt model, so `d. 3. maj` and `kl. 14` are not cut in
    two. The text is split on newlines first, because punkt cannot split
    `"Great location\nVery clean"` on its own. Fragments shorter than `MIN_SENT_TOKENS`, like
    "Great!", are dropped.

    All three voices go through the same splitter, so differences between the voices cannot
    come from how they were split. Every sentence gets a stable `sentence_id`.
    """)
    return


@app.cell
def _(re, sent_tokenize):
    MIN_SENT_TOKENS = 3

    def split_into_sentences(text: str, lang: str) -> list[str]:
        """Split one document into its sentences. Non-Danish text uses the English model."""
        language = "danish" if lang == "da" else "english"
        out = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sentence in sent_tokenize(line, language=language):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                if len(sentence.split()) >= MIN_SENT_TOKENS:
                    out.append(sentence)
        return out

    return (split_into_sentences,)


@app.cell
def _(guest_langed, host_langed, pd, split_into_sentences, tqdm):
    def explode_to_sentences(frame, voice, doc_id_col, date_col=None):
        """Turn a document-level frame into a sentence-level frame, one row per sentence."""
        records = []
        columns = frame.columns
        for row in tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc=f"splitting {voice}",
        ):
            row_dict = dict(zip(columns, row))
            doc_id = row_dict[doc_id_col]
            for pos, sentence in enumerate(
                split_into_sentences(row_dict["text"], row_dict["lang"])
            ):
                records.append(
                    {
                        # deterministic, so re-runs reproduce the same ids
                        "sentence_id": f"{voice[0]}{doc_id}-{pos}",
                        "doc_id": doc_id,
                        "voice": voice,
                        "sent_pos": pos,
                        "text": sentence,
                        "n_tokens_approx": len(sentence.split()),
                        "lang": row_dict["lang"],
                        "lang_conf": row_dict["lang_conf"],
                        "listing_id": row_dict["listing_id"],
                        "neighbourhood": row_dict["neighbourhood"],
                        # the unrepaired name, which matches neighbourhoods.geojson
                        "neighbourhood_key": row_dict["neighbourhood_cleansed"],
                        # host rows only; guest rows get False
                        "is_platform_template": bool(
                            row_dict.get("has_platform_template", False)
                        ),
                        "latitude": row_dict["latitude"],
                        "longitude": row_dict["longitude"],
                        "host_id": row_dict["host_id"],
                        "source": row_dict["source"],
                        # host descriptions have no date
                        "date": row_dict[date_col] if date_col else pd.NaT,
                    }
                )
        return pd.DataFrame.from_records(records)

    _guest_sentences = explode_to_sentences(
        guest_langed, voice="guest", doc_id_col="review_id", date_col="date"
    )
    # Give the host frame a `listing_id` so both halves share a schema.
    _host_sentences = explode_to_sentences(
        host_langed.assign(listing_id=host_langed["id"]),
        voice="host",
        doc_id_col="id",
    )

    sentences = pd.concat([_guest_sentences, _host_sentences], ignore_index=True)
    sentences["year"] = sentences["date"].dt.year.astype("Int64")

    print(f"guest sentences: {len(_guest_sentences):,}")
    print(f"host sentences:  {len(_host_sentences):,}")
    print(f"total:           {len(sentences):,}")
    return (sentences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8: the derived data table

    These are properties computed from the texts, one column per voice. The number of sentences
    per document shows that marketing and reviews are different kinds of text. The number of
    distinct listings shows coverage: if Danish reviews reach far fewer listings than English
    ones, an English and Danish difference is partly a difference in places. Sentence length is
    given as a median because it has a long right tail.

    The last column measures the three voices pooled together instead of adding up the other
    columns, since medians and ratios cannot be added and one listing can appear in all three
    voices. It is called "all three voices" and not "total" because 26% of the sentences are in
    other languages or too short to identify (step 8b).
    """)
    return


@app.cell
def _(pd, sentences):
    def voice_profile(subset):
        """The derived properties of one voice. Empty subsets return NaN, not 0."""
        if len(subset) == 0:
            return {k: float("nan") for k in (
                "documents", "sentences", "sentences per document (mean)",
                "tokens, total", "tokens per sentence (median)",
                "distinct listings", "areas covered (of 11)",
            )}
        # Count (voice, doc_id) pairs: a review id and a listing id can be the same number.
        n_docs = len(subset[["voice", "doc_id"]].drop_duplicates())
        return {
            "documents": n_docs,
            "sentences": len(subset),
            "sentences per document (mean)": round(len(subset) / n_docs, 2),
            "tokens, total": int(subset["n_tokens_approx"].sum()),
            "tokens per sentence (median)": float(subset["n_tokens_approx"].median()),
            "distinct listings": subset["listing_id"].nunique(),
            "areas covered (of 11)": subset["neighbourhood"].nunique(),
        }

    _host = sentences[sentences["voice"] == "host"]
    _guest_en = sentences[
        (sentences["voice"] == "guest") & (sentences["lang"] == "en")
    ]
    _guest_da = sentences[
        (sentences["voice"] == "guest") & (sentences["lang"] == "da")
    ]
    _all_three = pd.concat([_host, _guest_en, _guest_da])

    corpus_table = pd.DataFrame(
        {
            "host (marketing)": voice_profile(_host),
            "guest EN": voice_profile(_guest_en),
            "guest DA": voice_profile(_guest_da),
            # measured on the pooled rows, never summed across the row
            "all three voices": voice_profile(_all_three),
        }
    )
    print(corpus_table.to_string())
    return (corpus_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8b: what is actually in the file

    `corpus_table` compares the voices. This second table describes the whole file, including
    what falls outside the three voices.

    The data is nested: 1.8 million sentences in 480,248 documents, in 23,010 listings, in 11
    areas. Sentences from the same listing are not independent observations. 451,717 sentences
    are in 44 other languages and 8,992 are too short to identify. 358 listings have reviews but
    no host description, so the marketing and review voices do not cover exactly the same flats.
    """)
    return


@app.cell
def _(pd, sentences):
    _host = sentences[sentences["voice"] == "host"]
    _guest = sentences[sentences["voice"] == "guest"]
    _three = sentences[
        (sentences["voice"] == "host")
        | ((sentences["voice"] == "guest") & sentences["lang"].isin(["en", "da"]))
    ]
    _too_short = int((sentences["lang"] == "too-short").sum())

    # A Series rather than a DataFrame: the rows mix counts, a percentage and dates.
    frame_facts = {
        "sentences (rows)": len(sentences),
        "columns": sentences.shape[1],
        "documents": len(sentences[["voice", "doc_id"]].drop_duplicates()),
        "  host descriptions": _host["doc_id"].nunique(),
        "  guest reviews": _guest["doc_id"].nunique(),
        "distinct listings, any voice": sentences["listing_id"].nunique(),
        "  with a host description": _host["listing_id"].nunique(),
        "  with at least one review": _guest["listing_id"].nunique(),
        "  reviewed, no description": (
            sentences["listing_id"].nunique() - _host["listing_id"].nunique()
        ),
        "distinct hosts": sentences["host_id"].nunique(),
        "areas (of 11)": sentences["neighbourhood"].nunique(),
        # `too-short` is a label, not a language
        "languages identified": int(
            sentences.loc[sentences["lang"] != "too-short", "lang"].nunique()
        ),
        "sentences, three studied voices": len(_three),
        "  share of all sentences (%)": round(100 * len(_three) / len(sentences), 1),
        "sentences, other languages": len(sentences) - len(_three) - _too_short,
        "sentences, language unidentified": _too_short,
        "platform-template sentences": int(_host["is_platform_template"].sum()),
        "tokens, total": int(sentences["n_tokens_approx"].sum()),
        "tokens per sentence (median)": float(sentences["n_tokens_approx"].median()),
        # Guest voice only. The latest date is a few days after 30 June: Inside Airbnb
        # scrapes over a window and labels the file with one date.
        "earliest review": str(_guest["date"].min().date()),
        "latest review": str(_guest["date"].max().date()),
    }
    frame_table = pd.Series(frame_facts, name="value").to_frame()

    print(frame_table.to_string())
    return (frame_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 9: sentences over time

    The design is cross-sectional, so this plot only describes the data. Most of the rise in
    volume is survivorship: the snapshot only contains listings that were still active in June
    2026, so earlier years are under-represented. 2026 is left out because it is only half a
    year, and Danish gets its own panel as a share, since at about 5% it would sit on the zero
    line.
    """)
    return


@app.cell
def _(paths, plt, sentences):
    _guest = sentences[
        (sentences["voice"] == "guest") & (sentences["lang"].isin(["en", "da"]))
    ]
    per_year = (
        _guest.groupby(["year", "lang"]).size().unstack(fill_value=0).sort_index()
    )
    per_year = per_year[per_year.index < 2026]  # 30 June snapshot: 2026 is half a year

    fig, _axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    _axes[0].plot(per_year.index, per_year["en"], marker="o", label="English")
    _axes[0].plot(per_year.index, per_year["da"], marker="o", label="Danish")
    _axes[0].set_ylabel("review sentences")
    _axes[0].set_title(
        "Review sentences per year by language, Copenhagen\n"
        "(snapshot of 30 June 2026; host descriptions carry no date and are not shown)"
    )
    _axes[0].legend()
    _axes[0].grid(alpha=0.3)

    _share = per_year["da"] / per_year.sum(axis=1) * 100
    _axes[1].plot(_share.index, _share, marker="o", color="tab:orange")
    _axes[1].set_ylabel("Danish share (%)")
    _axes[1].set_xlabel("year the review was written")
    _axes[1].grid(alpha=0.3)
    _axes[1].set_title("Danish as a share of dated review sentences")

    fig.tight_layout()
    fig.savefig(paths.FIGURES / "sentences_per_year.png", dpi=150)
    print(f"saved {paths.FIGURES / 'sentences_per_year.png'}")
    print(per_year.tail(12).to_string())
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 9b: three voices across eleven areas

    The left panel shows sentences per area and voice on a log axis, because English sentences
    outnumber Danish ones about 17 to 1. It uses points instead of bars, since a log axis has no
    zero for a bar to start from.

    The right panel shows, for the listings in each area that have an English review sentence,
    the share that also have a Danish one. It goes from 68.6% in Brønshøj-Husum down to 39.9% in
    Nørrebro and falls as areas get bigger (Spearman's rho of minus 0.89, descriptive only). The
    language comparison is therefore partly mixed up with how central an area is.
    """)
    return


@app.cell
def _(paths, plt, sentences):
    # One label per arm; other languages and too-short rows become NaN and are left out.
    _arm = sentences["lang"].map({"en": "guest EN", "da": "guest DA"})
    _arm = _arm.where(sentences["voice"].ne("host"), "host")
    _plot_data = sentences.assign(arm=_arm).dropna(subset=["arm"])

    per_area = _plot_data.pivot_table(
        index="neighbourhood",
        columns="arm",
        values="sentence_id",
        aggfunc="count",
        fill_value=0,
    )
    # distinct listings per area and arm
    listings_per_area = _plot_data.pivot_table(
        index="neighbourhood",
        columns="arm",
        values="listing_id",
        aggfunc="nunique",
        fill_value=0,
    )

    # Same order (by corpus size) in both panels.
    _order = per_area.sum(axis=1).sort_values().index
    per_area = per_area.loc[_order]
    listings_per_area = listings_per_area.loc[_order]

    coverage = listings_per_area["guest DA"] / listings_per_area["guest EN"] * 100
    _rho = per_area.sum(axis=1).corr(coverage, method="spearman")

    fig_area, _ax = plt.subplots(1, 2, figsize=(13, 6))

    # left panel: sentences per area per voice
    # a thin grey line joins the three voices in each area
    _y = list(range(len(_order)))
    for _i, _area in enumerate(_order):
        _row = per_area.loc[_area]
        _ax[0].plot(
            [_row.min(), _row.max()], [_i, _i],
            color="lightgrey", linewidth=1, zorder=1,
        )
    for _arm_name, _colour, _marker in (
        ("host", "tab:green", "s"),
        ("guest EN", "tab:blue", "o"),
        ("guest DA", "tab:orange", "D"),
    ):
        _ax[0].scatter(
            per_area[_arm_name], _y,
            label=_arm_name, color=_colour, marker=_marker, s=55, zorder=2,
        )
    _ax[0].set_xscale("log")
    _ax[0].set_yticks(_y)
    _ax[0].set_yticklabels(_order)
    _ax[0].set_xlabel("sentences (log scale)")
    _ax[0].set_title("Sentences per area, by voice")
    _ax[0].legend(loc="lower right")
    _ax[0].grid(axis="x", alpha=0.3)
    _ax[0].set_ylim(-0.7, len(_y) - 0.3)

    # right panel: Danish listing coverage
    _ax[1].barh(_y, coverage.loc[_order], height=0.6, color="tab:orange")
    _overall = (
        listings_per_area["guest DA"].sum() / listings_per_area["guest EN"].sum() * 100
    )
    _ax[1].axvline(_overall, color="black", linestyle="--", linewidth=1)
    _ax[1].text(
        _overall + 0.7, len(_y) - 0.6, f"overall {_overall:.0f}%",
        fontsize=9, va="center",
    )
    _ax[1].set_yticks(_y)
    _ax[1].set_yticklabels([])  # shares the left panel's labels
    _ax[1].set_xlabel("% of English-reviewed listings that also have a Danish review")
    _ax[1].set_title(f"Danish coverage of listings (Spearman ρ = {_rho:.2f} with size)")
    _ax[1].grid(axis="x", alpha=0.3)
    _ax[1].set_ylim(-0.7, len(_y) - 0.3)

    fig_area.suptitle(
        "Corpus composition by area and voice — Copenhagen, 30 June 2026 snapshot",
        fontsize=12,
    )
    fig_area.tight_layout()
    fig_area.savefig(paths.FIGURES / "corpus_by_area_and_voice.png", dpi=150)
    print(f"saved {paths.FIGURES / 'corpus_by_area_and_voice.png'}")
    print(per_area.to_string())
    print()
    print("Danish listing coverage as % of English:")
    print(coverage.loc[_order].round(1).to_string())
    fig_area
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 9c: where the survivorship comes from

    The top panel shows reviews per year. The dip in 2020 and 2021 is COVID. The bottom panel
    shows listings by the year of their first review, which stands in for when a listing
    entered the market.

    Most of the listings still active are recent, so early years are only represented by the
    few long-lived ones. The rise in reviews over time is the sample filling up towards the
    snapshot date, not the city being reviewed more.
    """)
    return


@app.cell
def _(joined, listings_prepared, paths, plt):
    _reviews_per_year = joined["date"].dt.year.value_counts().sort_index()
    _entry_per_year = listings_prepared["first_review"].dt.year.value_counts().sort_index()

    fig_time, _axt = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 2026 is a half-year, drawn hatched rather than dropped.
    for _i, _series, _colour, _label in (
        (0, _reviews_per_year, "tab:blue", "reviews"),
        (1, _entry_per_year, "tab:green", "listings, by year of first review"),
    ):
        _full = _series[_series.index < 2026]
        _part = _series[_series.index == 2026]
        _axt[_i].bar(_full.index, _full.to_numpy(), color=_colour, label=_label)
        _axt[_i].bar(
            _part.index, _part.to_numpy(), color=_colour, alpha=0.45,
            hatch="//", edgecolor="white", label="2026 (partial year)",
        )
        _axt[_i].legend()
        _axt[_i].grid(axis="y", alpha=0.3)

    _axt[0].set_ylabel("reviews written")
    _axt[0].set_title("Reviews per year")
    _axt[1].set_ylabel("listings entering")
    _axt[1].set_xlabel("year")
    _axt[1].set_title(
        "Listings by year of first review — the surviving stock is recent, "
        "which is why the early years look thin"
    )
    # integer year ticks (the default locator prints 2012.5)
    _years = sorted(set(_reviews_per_year.index) | set(_entry_per_year.dropna().index))
    _axt[1].set_xticks([y for y in _years if y % 2 == 0])
    _axt[1].set_xticklabels([str(int(y)) for y in _years if y % 2 == 0])

    fig_time.suptitle(
        "Corpus coverage over time — Copenhagen, 30 June 2026 snapshot", fontsize=12
    )
    fig_time.tight_layout()
    fig_time.savefig(paths.FIGURES / "coverage_over_time.png", dpi=150)
    print(f"saved {paths.FIGURES / 'coverage_over_time.png'}")

    _recent = int(_entry_per_year[_entry_per_year.index >= 2022].sum())
    _total = int(_entry_per_year.sum())
    print(f"listings whose first review is 2022 or later: {_recent:,} of {_total:,}"
          f" ({_recent / _total:.1%})")
    print(f"listings with no review at all: "
          f"{int(listings_prepared['first_review'].isna().sum()):,}")
    fig_time
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 10: save

    Parquet keeps the column types and is fast to read one column at a time. The sentence table
    goes to `data/interim/`, which is git-ignored because it contains review text. The count
    tables go to `data/processed/`.
    """)
    return


@app.cell
def _(corpus_table, frame_table, load_counts, paths, sentences, snap):
    _sentences_path = paths.INTERIM / f"sentences_voices_{snap.slug}.parquet"
    sentences.to_parquet(_sentences_path, index=False)

    _table_path = paths.PROCESSED / f"corpus_table_{snap.slug}.csv"
    corpus_table.to_csv(_table_path)

    _frame_path = paths.PROCESSED / f"frame_table_{snap.slug}.csv"
    frame_table.to_csv(_frame_path)

    print(f"wrote {_sentences_path}  ({len(sentences):,} sentences)")
    print(f"wrote {_table_path}")
    print(f"wrote {_frame_path}")
    print()
    print("load counts, for the record:")
    for _name, _counts in load_counts.items():
        print(f"  {_name}: {_counts}")
    return


if __name__ == "__main__":
    app.run()
