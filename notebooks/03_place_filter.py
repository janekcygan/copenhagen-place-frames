import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import hashlib
    import re
    import sys
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    import marimo as mo
    import nltk
    import numpy as np
    import pandas as pd
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from scipy.special import gammaln
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    from tqdm import tqdm

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(NOTEBOOK_DIR.parent / "scripts"))

    import paths

    snap = paths.snapshot("30 June, 2026")

    CORPUS_PATH = paths.interim("sentences_voices", snap)
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{CORPUS_PATH} is missing — run notebooks/02_data_preperation.py first"
        )

    nltk.download("stopwords", quiet=True)

    print(f"snapshot: {snap}")
    print(f"corpus:   {CORPUS_PATH.name}")
    return (
        CORPUS_PATH,
        CountVectorizer,
        Counter,
        LogisticRegressionCV,
        SnowballStemmer,
        defaultdict,
        gammaln,
        hashlib,
        mo,
        np,
        pd,
        re,
        stopwords,
        unicodedata,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 Building the place filter

    The filter keeps sentences that refer to somewhere in Copenhagen outside the property
    itself. It is built with the keyword discovery method of King, Lam & Roberts, run separately
    on the three arms (`host/en`, `guest/en`, `guest/da`). The words found in the three runs are
    merged into one bilingual lexicon, which is then applied in the same way to all three voices.

    ## The matcher

    Text is never matched with `.str.contains`. pandas 3 stores strings with PyArrow, and
    `.str.contains(regex=True)` then runs in the RE2 engine, where `\b` only knows ASCII word
    boundaries. `ø`, `å` and `æ` do not count as letters there. In the Danish reviews, `\bsø\b`
    finds 2,764 sentences with RE2 but 15 with Python's `re`, because RE2 also matches inside
    `søen` and `søerne`. `\bpå\b` finds 64 instead of 9,130, and `\bøsterbro\b` in English
    reviews finds 0 instead of 419. `make_matcher` compiles patterns with Python's `re` and
    applies them with `.map()`.

    `fold` is only used for toponyms. Guests without a Danish keyboard write `Norrebro`, and
    folding raises the hits for `nørrebro` in English text from 2,813 to 4,972. Folding `område`
    in Danish text adds one hit. Folding is only used for matching; the stored text keeps the
    Danish letters.
    """)
    return


@app.cell
def _(pd, re, unicodedata):
    def fold(s: str) -> str:
        """Turns 'Nørrebro' into 'norrebro'. ø, æ and å have no NFKD decomposition, so they are replaced by hand."""
        s = s.lower().replace("ø", "o").replace("æ", "ae").replace("å", "aa")
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))

    def make_pattern(terms, fold_accents=False):
        """
        One compiled, word-bounded pattern for a list of terms. Returns None for an
        empty list, because \\b(?:)\\b would match every sentence.
        """
        if not terms:
            return None

        prepared = [fold(t) if fold_accents else t for t in terms]

        return re.compile(
            r"\b(?:" + "|".join(re.escape(t) for t in prepared) + r")\b",
            flags=re.IGNORECASE,
        )

    def make_matcher(terms, fold_accents=False):
        """Returns a function that tests a Series for any of the terms (True/False per row)."""
        pattern = make_pattern(terms, fold_accents=fold_accents)

        # an empty term list matches nothing
        if pattern is None:
            return lambda series: pd.Series(False, index=series.index, dtype=bool)

        if fold_accents:
            return lambda series: (
                series.fillna("").map(lambda t: bool(pattern.search(fold(t)))).astype(bool)
            )
        return lambda series: (
            series.fillna("").map(lambda t: bool(pattern.search(t))).astype(bool)
        )

    return fold, make_matcher, make_pattern


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0: the working set

    There are three arms. `host/en` has 105,491 sentences from 22,608 listings, marketing text
    that names places and gives distances. `guest/en` has 1,165,874 sentences from 18,888
    listings, describing what guests did. `guest/da` has 67,285 sentences from 9,006 listings,
    the same in Danish.

    Danish host descriptions are left out: 63 sentences across 13 listings are too few to train
    a classifier on or to compare with anything.

    The three arms make up 74.4% of the 1,799,485 sentences. The rest are reviews in other
    languages and sentences too short to identify.
    """)
    return


@app.cell
def _(CORPUS_PATH, pd):
    CORPUS_COLS = [
        "sentence_id",
        "doc_id",                # step 9: review length
        "n_tokens_approx",       # step 9: sentence length
        "voice", "lang",         # define an arm
        "text",
        "listing_id",            # step 7: one host's copy, or a corpus pattern?
        "host_id",
        "neighbourhood",         # step 9: retention per area
        "is_platform_template",  # Airbnb boilerplate in host text
    ]

    sentences = pd.read_parquet(CORPUS_PATH, columns=CORPUS_COLS)

    # Danish host text is not an arm.
    ARM_LANG = {"host/en": "en", "guest/en": "en", "guest/da": "da"}

    arms = {}
    for _name, _lang in ARM_LANG.items():
        _voice = _name.split("/")[0]
        arms[_name] = sentences[(sentences.voice == _voice) & (sentences.lang == _lang)]

    print(f"{'arm':10} {'sentences':>12} {'listings':>10} {'hosts':>8} {'template':>9}")
    for _name, _df in arms.items():
        print(f"{_name:10} {len(_df):>12,} {_df.listing_id.nunique():>10,} "
              f"{_df.host_id.nunique():>8,} {int(_df.is_platform_template.sum()):>9,}")

    _kept = sum(len(_df) for _df in arms.values())
    print(f"\nthree arms: {_kept:,} of {len(sentences):,} sentences ({_kept/len(sentences):.1%})")
    return ARM_LANG, arms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regression check

    This re-counts three terms with known values every time the notebook runs, so it stops if
    the word boundary bug ever comes back.
    """)
    return


@app.cell
def _(arms, make_matcher):
    # The RE2 (wrong) values are in the comments.
    _CHECKS = [
        ("sø",       "guest/da",    15),   # RE2: 2,764
        ("på",       "guest/da", 9_130),   # RE2:    64
        ("østerbro", "guest/en",   419),   # RE2:     0
    ]

    # measure all three before judging any
    _results = []
    for _term, _arm, _expected in _CHECKS:
        _got = int(make_matcher([_term])(arms[_arm]["text"]).sum())
        _results.append((_term, _arm, _got, _expected))

    for _term, _arm, _got, _expected in _results:
        _flag = "ok" if _got == _expected else "CHANGED"
        print(f"  {_term!r:10} {_arm:9} {_got:>7,}  expected {_expected:>7,}   {_flag}")

    _moved = [(t, a, g, e) for t, a, g, e in _results if g != e]
    if _moved:
        raise AssertionError(
            f"matcher regression — {len(_moved)} of {len(_results)} counts moved: {_moved}. "
            "Expected values are pinned to the 30 June 2026 snapshot. If they read "
            "2,764 / 64 / 0, the pattern is being executed by PyArrow's RE2."
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the seed set

    A good seed word almost never appears in a sentence that is not about place. `central`
    fails because of "central heating", and `quiet` fails because flats are quiet too. It also
    has to be frequent enough to make a reference set. And the seeds are balanced by concept
    rather than by count: every English seed has a Danish one that covers the same idea.

    Toponyms are kept out of the classifier and applied next to the lexicon, folded. Otherwise
    the classifier would spend its effort recognising sentences about Nørrebro instead of
    finding new words.
    """)
    return


@app.cell
def _(arms, make_matcher, pd):
    # tools for checking a term (steps 1 and 7)

    def see_in_context(term, arm="guest/en", n=10, fold_accents=False, seed=1):
        """
        Print counts for a term (sentences, listings, hosts, platform template), then
        return n sample sentences to read. Many hits from few listings means one host's copy.
        """
        df = arms[arm]
        hit = make_matcher([term], fold_accents=fold_accents)(df["text"])
        sub = df.loc[hit]
        print(f"{term!r} in {arm}: {len(sub):,} sentences ({hit.mean():.2%} of arm), "
              f"{sub.listing_id.nunique():,} listings, {sub.host_id.nunique():,} hosts, "
              f"{int(sub.is_platform_template.sum()):,} platform-template")
        if len(sub) == 0:
            print("  -> no hits. Check the spelling, and check the arm's language.")
        return sub["text"].sample(min(n, len(sub)), random_state=seed)

    def count_across_arms(terms, fold_accents=False):
        """One row per term: its rate in all three arms."""
        # a bare string would be looped over letter by letter
        if isinstance(terms, str):
            terms = [terms]
        rows = []
        for term in terms:
            matcher = make_matcher([term], fold_accents=fold_accents)
            row = {"term": term}
            for name, df in arms.items():
                hit = matcher(df["text"])
                row[f"{name} n"] = int(hit.sum())
                row[f"{name} %"] = round(100 * hit.mean(), 3)
                row[f"{name} listings"] = int(df.loc[hit, "listing_id"].nunique())
            rows.append(row)
        return pd.DataFrame(rows).set_index("term")

    def count_concepts(concepts, leak_threshold=0.005):
        """
        Per-concept coverage parity, the acceptance test.

        The rates use both languages' terms together, because the shipped filter is one
        merged lexicon applied to all text.

          en %, da %             retention of the combined terms on each guest arm
          en/da                  their ratio
          host %                 the combined terms on host/en
          en-list %, da-list %   each side on its own arm, to spot gaps
          flags                  empty sides, and leakage into the other language above
                                 leak_threshold (Danish 'by' means town, English 'by' does not)
        """
        rows = []
        for _name, _c in concepts.items():
            _en_terms, _da_terms = _c["en"], _c["da"]

            _m_union = make_matcher(_en_terms + _da_terms)
            _m_en = make_matcher(_en_terms)
            _m_da = make_matcher(_da_terms)

            _en_txt = arms["guest/en"]["text"]
            _da_txt = arms["guest/da"]["text"]

            # what the filter keeps
            _en_rate = _m_union(_en_txt).mean()
            _da_rate = _m_union(_da_txt).mean()
            _ho_rate = _m_union(arms["host/en"]["text"]).mean()

            # each side on its own arm
            _en_list = _m_en(_en_txt).mean()
            _da_list = _m_da(_da_txt).mean()

            # each side on the other language's arm
            _leak_en = _m_da(_en_txt).mean()
            _leak_da = _m_en(_da_txt).mean()

            _flags = []
            if not _en_terms:
                _flags.append("en side EMPTY")
            if not _da_terms:
                _flags.append("da side EMPTY")
            if _leak_en > leak_threshold:
                _flags.append(f"da->en leak {_leak_en:.1%}")
            if _leak_da > leak_threshold:
                _flags.append(f"en->da leak {_leak_da:.1%}")

            rows.append({
                "concept": _name,
                "en n": len(_en_terms), "da n": len(_da_terms),
                "en %": round(100 * _en_rate, 3),
                "da %": round(100 * _da_rate, 3),
                # inf when one side of the concept is empty
                "en/da": round(_en_rate / _da_rate, 2) if _da_rate else float("inf"),
                "host %": round(100 * _ho_rate, 3),
                "en-list %": round(100 * _en_list, 3),
                "da-list %": round(100 * _da_list, 3),
                "flags": "; ".join(_flags),
            })

        return pd.DataFrame(rows).set_index("concept")

    return count_across_arms, count_concepts, see_in_context


@app.cell
def _():
    # The original hand-written keyword list (v1), frozen for comparison in step 9.
    # Not used as a filter.

    V1_AREA_WORDS = [      # English
            "neighbourhood", "neighborhood", "area", "district", "quarter",
            "part of town", "part of the city", "location", "surroundings",
            # Danish
            "kvarter", "kvarteret", "bydel", "område", "omrade", "nabolag",
            "beliggenhed", "omgivelser",]
    V1_TOPONYMS = [        "nørrebro", "norrebro", "vesterbro", "østerbro", "osterbro",
            "amager", "indre by", "frederiksberg", "christianshavn", "christiania",
            "valby", "sydhavn", "sydhavnen", "nordvest", "islands brygge",
            "refshaleøen", "refshaleoen", "vanløse", "vanlose",
            "brønshøj", "bronshoj", "bispebjerg", "kongens nytorv", "nyhavn",
            "meatpacking", "kødbyen", "kodbyen", "carlsberg", "sluseholmen",
            "ørestad", "orestad", "husum", "sydhavn", "holmen", "nordhavn",
            "frederiksstaden", "latinerkvarteret", "strøget", "stroget",]
    V1_PLACE_ADJACENT = [        # English
            "cafe", "café", "coffee", "bar", "restaurant", "shop", "supermarket",
            "walk", "walking distance", "metro", "bus", "bike", "cycling",
            "street", "streets", "park", "lakes", "canal", "harbour", "harbor",
            "locals", "quiet", "lively", "safe", "central", "nearby", "around the corner",
            # Danish
            "gade", "gaden", "park", "sø", "søerne", "soerne", "havn", "cykel", "metro",
            "butik", "supermarked", "restaurant", "lokale", "roligt", "centralt",]

    V1_PLACE_TERMS = sorted(set(V1_AREA_WORDS + V1_TOPONYMS + V1_PLACE_ADJACENT))
    print(f"v1 lexicon frozen: {len(V1_PLACE_TERMS)} terms "
          f"({len(V1_AREA_WORDS)} area, {len(V1_TOPONYMS)} toponym, "
          f"{len(V1_PLACE_ADJACENT)} place-adjacent)")
    return (V1_PLACE_TERMS,)


@app.cell
def _(Counter, arms, defaultdict, pd, re):
    # toponym sweep
    # Lists capitalised words and word pairs across all three arms, side by side, to
    # find place names. It cannot tell a place from a first name (many guests thank
    # their host by name), so the table has to be read.

    # Two or more letters, starting with a letter.
    _TOKEN = re.compile(r"[A-Za-zÆØÅæøåÉé][\wÆØÅæøåÉé'’-]{1,}")

    # A capitalised word right after one of these is likely a place.
    _PLACE_PREPS = {
        "in", "at", "near", "to", "from", "around", "by", "towards", "through", "into",
        "i", "på", "ved", "til", "fra", "omkring", "gennem", "mod", "nær",
    }

    def sweep_proper_nouns(min_unigram=25, min_bigram=40):
        """
        Every capitalised token and capitalised pair across the three arms.
        Pairs matter because many toponyms are two words (Kongens Nytorv).
        Sentence-initial and ALL-CAPS tokens are skipped.

        Returns (unigrams, bigrams). Columns: count per arm, `arms` (how many
        arms contain it), `listings` (distinct listings), `prep %` (share of hits
        right after a place preposition).
        """
        _uni = {arm: Counter() for arm in arms}
        _bi = {arm: Counter() for arm in arms}
        _prep = {arm: Counter() for arm in arms}
        _lst = {arm: defaultdict(set) for arm in arms}

        for _arm, _df in arms.items():
            for _text, _lid in zip(_df["text"].fillna(""), _df["listing_id"]):
                _toks = _TOKEN.findall(_text)
                for _i, _tok in enumerate(_toks):
                    if _i == 0 or len(_tok) < 3:
                        continue
                    if not _tok[0].isupper() or _tok.isupper():
                        continue

                    _key = _tok.lower()
                    _uni[_arm][_key] += 1
                    _lst[_arm][_key].add(_lid)
                    _prep[_arm][_key] += _toks[_i - 1].lower() in _PLACE_PREPS

                    # followed by another capitalised word: a multi-word name
                    _nxt = _toks[_i + 1] if _i + 1 < len(_toks) else ""
                    if _nxt and _nxt[0].isupper() and not _nxt.isupper():
                        _b = f"{_tok} {_nxt}".lower()
                        _bi[_arm][_b] += 1
                        _lst[_arm][_b].add(_lid)
                        _prep[_arm][_b] += _toks[_i - 1].lower() in _PLACE_PREPS

        def _build(counters, floor):
            _total = Counter()
            for _c in counters.values():
                _total.update(_c)
            _rows = []
            for _term, _n in _total.items():
                if _n < floor:
                    continue
                _row = {"term": _term, "total": _n}
                for _arm in arms:
                    _row[_arm] = counters[_arm][_term]
                _row["arms"] = sum(counters[_arm][_term] > 0 for _arm in arms)
                _row["listings"] = sum(len(_lst[_arm][_term]) for _arm in arms)
                _row["prep %"] = round(
                    100 * sum(_prep[_arm][_term] for _arm in arms) / _n, 1
                )
                _rows.append(_row)
            return pd.DataFrame(_rows).set_index("term").sort_values(
                "total", ascending=False
            )

        return _build(_uni, min_unigram), _build(_bi, min_bigram)

    unigrams, bigrams = sweep_proper_nouns()
    print(f"{len(unigrams):,} single tokens and {len(bigrams):,} pairs above the floor")
    return bigrams, unigrams


@app.cell
def _(bigrams):
    # multi-word names present in all three arms
    bigrams[bigrams["arms"] == 3].head(50)
    return


@app.cell
def _(unigrams):
    # single tokens, sorted by how often they follow a place preposition
    unigrams[(unigrams["arms"] == 3) & (unigrams["total"] >= 50)] \
        .sort_values("prep %", ascending=False).head(50)
    return


@app.cell
def _():
    # STEP 1: THE SEED SET
    # Organised by concept, so each English entry has a Danish counterpart.

    seed_concepts = {
        "identity": {
            "en": [
                "neighbourhood", "neighbourhoods",
                "neighborhood", "neighborhoods",
                "district", "districts",
                "quarter", "quarters",
                "area", "areas",
            ],
            "da": [
                "kvarter", "kvarteret", "kvarterer", "kvartererne",
                "bydel", "bydelen",
                "område", "området", "områder", "områderne",
                "nabolag", "nabolaget", "nabolagene",
            ],
            "why": "delimits an area as a specific place in the city",
        },
        "character": {
            "en": [
                "lively", "green",
            ],
            "da": [
                "livlig", "livligt", "livlige",
                "grøn", "grønt", "grønne",
            ],
            "why": "describes the area's character, not a flat's features",
        },
        "location": {
            "en": [
                "location", "locations",
                "located",
                "centre", "centres", "center", "centers",
                "distance", "distances", "surroundings",
                "street", "streets",
                "road", "roads",
            ],
            "da": [
                "beliggenhed", "beliggenheden",
                "beliggende",
                "omgivelse", "omgivelser", "omgivelserne",
                "gade", "gaden", "gader", "gaderne",
                "vej", "vejen", "veje", "vejene",
                "centrum", "centret",
                "afstand", "afstanden",
            ],
            "why": "describes the placement in relation to the urban space; 'central' is deliberately absent because it may refer to the flat's features rather than the area's character",
        },
        "amenities": {
            "en": [
                "bar", "bars",
                "cafe", "cafes", "café", "cafés",
                "restaurant", "restaurants",
            ],
            "da": [
                "bar", "baren", "barer",
                "café", "caféen", "caféer", "caféerne",
                "restaurant", "restauranten", "restauranter", "restauranterne",
            ],
            "why": "describes amenities of the area, not the flat",
        },
        "connections": {
            "en": [
                "metro",
                "subway",
                "tube",
                "station", "stations",
                "bus", "buses",
                "stop", "stops",
            ],
            "da": [
                "bus", "bussen", "busser", "busserne",
                "metro", "metroen",
                "metrostation", "metrostationen", "metrostationer",
                "station", "stationen", "stationer", "stationerne",
                "stoppested",
                "togstation", "togstationen", "togstationer",
            ],
            "why": "describes how the flat connects to the city",
        },
    }

    # flat list for the matcher; set() because some terms are the same in both languages
    reference_words_seed = sorted({
        t for _c in seed_concepts.values() for t in _c["en"] + _c["da"]
    })

    # Exclusions, also in both languages.
    # Kind A names an out-of-scope place and vetoes the sentence.
    # Kind B is a seed used in a non-place sense and only cancels its own match.
    exclusion_concepts = {
        "inside the property": {
            "en": [
                # private outdoor space. Not 'garden(s)': mostly Tivoli or the
                # Botanical Garden.
                "backyard", "backyards", "back yard", "back yards",
                "courtyard", "courtyards",

                # parts of a building
                "stairs", "stair", "staircase", "staircases",
                "attic",
                "basement",
                "entrance", "entrances",
                "corridor", "corridors", "hallway", "hallways",
                "elevator", "elevators", "lift", "lifts",
                "reception", "receptions",

                # access. Not bare 'key(s)': often 'key location'.
                "key box", "key boxes", "keybox", "lockbox", "lock box", "key code",

            ],
            "da": [
                "baghave", "baghaven", "baghaver", "baghaverne",
                "baggård", "baggården", "baggårde", "baggårdene",
                "gårdhave", "gårdhaven", "gårdhaver", "gårdhaverne",
                "gård", "gården", "gårde", "gårdene",
                "siddeplads", "siddepladsen", "siddepladser", "siddepladserne",
                "siddeområde", "siddeområdet", "siddeområder", "siddeområderne",
                "legeområde", "legeområdet", "legeområder", "legeområderne",
                "trappe", "trappen", "trapper", "trapperne",
                "kælder", "kælderen", "kældre", "kældrene",
                "indgang", "indgangen", "indgange", "indgangene",
                "korridor", "korridoren", "korridorer", "korridorerne",
                "elevatoren", "elevatorer", "elevatorerne",
                "receptionen",
                "nøgle", "nøglen", "nøgler", "nøglerne", "nøgleboks", "nøgleboksen",
            ],
            "why": (
                "places inside the property. Bare 'garden', 'key', 'parking', "
                "'outdoor' and 'have' were tested and rejected: each removed more "
                "place text than non-place text. 'elevator' and 'reception' are "
                "spelled the same in both languages and sit on the en side only."
            ),
            "kind": "A",
        },

        "outside copenhagen": {
            "en": ["malmo", "sweden", "roskilde", "helsingor", "elsinore", "louisiana", "berlin"],
            "da": ["malmø", "malmö", "sverige", "helsingør"],
            "why": (
                "places outside Copenhagen, split by spelling rather than language. "
                "Small: 0.11% of retained English, 0.03% of retained Danish."
            ),
            "kind": "A",
        },

        "seed word contradictions": {
            "en": [
                "green garden", "green plants",
                "stop by", "stop me",
                "seating area", "seating areas",
                "play area", "play areas",
                "outside area", "outside areas",
                "outdoor area", "outdoor areas",
                "dining area", "dining areas",
                "living area", "living areas",
                "common area", "common areas",
                "kitchen area", "kitchen areas",
                "storage area", "storage areas",
                "bathroom area", "bathroom areas",
                "garden area", "garden areas",
            ],
            "da": [
                "hele vejen",          # 'all the way', not a road
                "grøn gårdhave", "grøn baggård", "grønne planter",
                "bar præg af", "bar lidt præg af", "bar tydeligt præg af", "bar over",
                "på en vej",
            ],
            "why": (
                "a seed used in a non-place sense, found by collocation sweep. "
                "Bare 'stop' is not here: 80% of its hits are 'bus stop'/'metro stop'."
            ),
            "kind": "B",
        },
    }

    # Kind A and Kind B are combined by different rules, so they stay separate.
    _a, _b = [], []
    for _name, _c in exclusion_concepts.items():
        _terms = _c["en"] + _c["da"]
        if _c["kind"] == "A":
            _a += _terms
        elif _c["kind"] == "B":
            _b += _terms
        else:
            raise ValueError(f"{_name!r}: kind must be 'A' or 'B', got {_c['kind']!r}")

    exclusion_a_seed = sorted(set(_a))
    exclusion_b_seed = sorted(set(_b))

    # Terms proposed by discovery and adopted, one entry per round, organised by concept.
    # A new concept must carry a "why".
    reference_words_expanded = {
        # Round 1: discovery on seeds + toponyms, min_df 10, threshold 0.25.
        "round 1 (2026-09-28)": {
            "location": {
                "en": ["central", "city"],
                "da": ["central", "centralt", "centrale", "byen"],
                # the flat sense of 'central' is 'central heating', now Kind B
            },
            "connections": {
                "en": ["public", "transport"],
                "da": ["offentlig", "offentligt", "offentlige", "transport"],
            },
            "amenities": {
                "en": ["bakery", "bakeries"],
                "da": ["bageri", "bagerier"],
            },
            "proximity": {
                "en": ["close", "near", "nearby", "around the corner",
                        "walkable", "walkability"],
                "da": ["tæt", "i nærheden", "om hjørnet", "gåafstand"],
                "why": "states that somewhere is near or reachable on foot — the "
                        "relation to a place rather than the place itself",
            },
            "green and blue space": {
                "en": ["beach", "lake", "lakes", "parks"],
                "da": ["strand", "stranden", "sø", "søen", "søer", "søerne",
                       "parker"],
                "why": "names public green or water space in the city",
            },
        },
        # Round 2: discovery on seeds + toponyms + round 1, same settings.
        "round 2 (2026-09-28)": {
            "location": {
                "en": ["heart of"],                       # 'in the heart of'
                "da": ["hjertet af", "placering", "placeringen", "placeret"],
            },
        },
        # Step-9 form sweep (Danish only): words containing a Danish term, not yet in
        # the lexicon, in >= 5 sentences. Rejected: 'buss' (Norwegian/Swedish),
        # city-level names, and 'parkering*' (a flat feature).
        "sweeps (2026-09-28)": {
            "identity": {
                "da": ["nærområdet", "nærområde", "lokalområdet", "villakvarter",
                       "boligkvarter", "boligområde"],
            },
            "location": {
                "da": ["velbeliggende", "centraltbeliggende", "velplaceret",
                       "byens", "midtbyen", "storbyen", "storbyens",
                       "villavej", "sidegade", "sidevej", "cykelafstand",
                       "supercentralt", "motorvej", "motorvejen",
                       # street noise describes the area, unlike 'quiet'
                       "gadestøj"],
            },
            "connections": {
                "da": ["havnebus", "havnebussen", "vandbussen", "busstop",
                       "busstoppested", "busstoppestedet", "busstoppesteder",
                       "busforbindelse", "busforbindelser", "transportmuligheder",
                       "transportmidler", "transporten", "kollektivtransport",
                       "bustransport", "togsstation"],
            },
            "proximity": {
                "da": ["tættere", "tætte", "supertæt"],
            },
            "green and blue space": {
                "da": ["badestrand"],
            },
            "amenities": {
                # 'cafeer' and 'spisesteder' were found by hand, not by the scan
                "da": ["kaffebar", "kaffebarer", "cafeer", "spisested",
                       "spisesteder", "spisestederne"],
            },
        },
    }
    exclusion_a_expanded = {}
    exclusion_b_expanded = {
        # Kind B because it contains the seed 'central'
        "round 1 (2026-09-28)": [
            "central heating",
        ],
    }

    # Toponyms: areas, landmarks, squares, streets, stations and parks. Not businesses,
    # and not city-level names (Copenhagen, København, cph), since the whole corpus is
    # about Copenhagen. Matched folded, so one spelling per entry; the misspellings at
    # the end are ones folding cannot reach.
    toponyms = [
        # the 11 official areas (neighbourhoods.csv) and their parts
        "indre by", "inner city", "nørrebro", "vesterbro", "østerbro", "frederiksberg",
        "amager", "amagerbro", "valby", "vanløse", "brønshøj", "husum", "bispebjerg",
        "nordvest", "enghave", "sydhavn", "sydhavnen", "christianshavn",
        # sub-areas guests and hosts name
        "islands brygge", "island brygge", "nordhavn", "ørestad", "refshaleøen", "reffen",
        "kødbyen", "meatpacking", "meat packing", "carlsberg", "carlsbergbyen",
        "carlsberg byen", "sluseholmen", "frederiksstaden", "latinerkvarteret",
        "latin quarter", "holmen", "christiania", "brumleby",
        # landmarks
        "tivoli", "nyhavn", "strøget", "rosenborg", "amalienborg", "christiansborg",
        "little mermaid", "den lille havfrue", "round tower", "rundetårn",
        "marble church", "marmorkirken", "torvehallerne", "town hall", "city hall",
        "royal arena", "bella center", "bella centre", "fisketorvet",
        # squares and streets
        "kongens nytorv", "kgs nytorv", "rådhuspladsen", "enghave plads",
        "sankt hans torv", "skjolds plads", "nuuks plads", "bopa plads", "gammel strand",
        "trianglen", "nørrebrogade", "vesterbrogade", "østerbrogade", "amagerbrogade",
        "jægersborggade", "værnedamsvej", "istedgade", "elmegade", "gammel kongevej",
        "sønder boulevard", "fasanvej",
        # stations
        "nørreport", "østerport", "hovedbanegården", "svanemøllen", "dybbølsbro",
        "flintholm",
        # parks, green and blue space
        "fælledparken", "kongens have", "king's garden", "kings garden", "assistens",
        "superkilen", "utterslev mose", "amager strand", "amager strandpark",
        "amager fælled", "søndermarken", "lergravsparken", "valbyparken",
        "nørrebroparken",
        # misspellings that folding cannot reach
        "nyhaven", "christiana", "fredriksberg", "fredericksberg", "vestebro",
        "christianhavn",
    ]
    return (
        exclusion_a_expanded,
        exclusion_a_seed,
        exclusion_b_expanded,
        exclusion_b_seed,
        reference_words_expanded,
        reference_words_seed,
        seed_concepts,
        toponyms,
    )


@app.cell
def _(
    exclusion_a_expanded,
    exclusion_a_seed,
    exclusion_b_expanded,
    exclusion_b_seed,
    re,
    reference_words_expanded,
    reference_words_seed,
    seed_concepts,
    toponyms,
):
    # the working lists: seeds plus the adopted rounds
    # Everything below uses reference_words, exclusion_a and exclusion_b.

    # Seeds plus every round, merged. A copy, so seed_concepts never changes.
    all_concepts = {
        _name: {"en": list(_c["en"]), "da": list(_c["da"]), "why": _c["why"]}
        for _name, _c in seed_concepts.items()
    }
    for _round_name, _round in reference_words_expanded.items():
        for _name, _c in _round.items():
            if _name not in all_concepts:
                # also catches a misspelled concept name
                if "why" not in _c:
                    raise ValueError(
                        f"{_round_name}: concept {_name!r} is new and has no 'why' "
                        f"— or is it a misspelling of one of {list(seed_concepts)}?"
                    )
                all_concepts[_name] = {"en": [], "da": [], "why": _c["why"]}
            all_concepts[_name]["en"] += _c.get("en", [])
            all_concepts[_name]["da"] += _c.get("da", [])

    reference_words = sorted({
        t for _c in all_concepts.values() for t in _c["en"] + _c["da"]
    })

    exclusion_a = exclusion_a_seed + [
        t for _round in exclusion_a_expanded.values() for t in _round
    ]
    exclusion_b = exclusion_b_seed + [
        t for _round in exclusion_b_expanded.values() for t in _round
    ]

    # A Kind A term must never contain a seed: then it could be the reason a sentence
    # was kept, and vetoing on it is contradictory. Such terms belong in Kind B
    # ('play area', 'seating area').
    _misfiled = [
        (_term, _seed)
        for _term in exclusion_a
        for _seed in reference_words
        if re.search(rf"\b{re.escape(_seed)}\b", _term, flags=re.IGNORECASE)
    ]
    if _misfiled:
        raise AssertionError(
            f"{len(_misfiled)} Kind A term(s) contain a seed and must move to "
            f"Kind B: {_misfiled}"
        )

    print(f"reference:   {len(reference_words_seed)} seed "
            f"+ {len(reference_words) - len(reference_words_seed)} accepted "
            f"over {len(reference_words_expanded)} rounds = {len(reference_words)}")
    print(f"exclusion A: {len(exclusion_a_seed)} seed "
            f"+ {len(exclusion_a) - len(exclusion_a_seed)} accepted "
            f"over {len(exclusion_a_expanded)} rounds = {len(exclusion_a)}   (vetoes the sentence)")
    print(f"exclusion B: {len(exclusion_b_seed)} seed "
            f"+ {len(exclusion_b) - len(exclusion_b_seed)} accepted "
            f"over {len(exclusion_b_expanded)} rounds = {len(exclusion_b)}   (cancels its own match)")
    print(f"toponyms:    {len(toponyms)} (applied separately, folded)")

    print(f"\nKind A / seed guard: ok "
            f"({len(exclusion_a)} x {len(reference_words)} pairs checked)")

    # seeds that can overrule a Kind A veto
    # "great location, lovely courtyard" should not be dropped for the courtyard.
    # A seed from `location` or `identity` overrules the veto, unless it also has an
    # in-property sense (recorded as a Kind B entry, like 'area' in 'seating area').
    # Derived from the lists, never typed.
    _candidates = sorted(set(
        all_concepts["location"]["en"] + all_concepts["location"]["da"]
        + all_concepts["identity"]["en"] + all_concepts["identity"]["da"]
    ))
    _has_kind_b = {
        _seed for _seed in _candidates
        if any(re.search(rf"\b{re.escape(_seed)}\b", _b, flags=re.IGNORECASE)
               for _b in exclusion_b)
    }
    veto_overrides = [_s for _s in _candidates if _s not in _has_kind_b]
    print(f"veto overrides: {len(veto_overrides)} location/identity seeds; "
          f"left out for having an in-property sense: {sorted(_has_kind_b)}")

    if not reference_words:
        print("\n-> step 1 not done yet. Everything below is inert until there are seeds.")
    return (
        all_concepts,
        exclusion_a,
        exclusion_b,
        reference_words,
        veto_overrides,
    )


@app.cell
def _(all_concepts, count_concepts):
    concept_table = count_concepts(all_concepts)
    concept_table
    return (concept_table,)


@app.cell
def _(make_matcher, make_pattern, pd):
    def place_filter(text, reference, kind_a, kind_b, overrides=()):
        """
        Keep a sentence unless the thing we object to is the reason it is here.

        Kind A (a stairwell, Malmö) vetoes the sentence.
        Kind B (a seed misfiring: 'hele vejen', 'green garden') is crossed out, and
        the sentence is kept if another seed is still present.
        overrides: area-naming seeds that, if still present, cancel the Kind A veto.

        The crossed-out text is only used for the test; the stored text is unchanged.
        """
        ref_pattern = make_pattern(reference)
        b_pattern = make_pattern(kind_b)
        o_pattern = make_pattern(list(overrides))
        kind_a_hit = make_matcher(kind_a)(text)

        # Cross out Kind B first, then test seeds and overrides on the same text.
        if b_pattern is None:
            crossed = text.fillna("")
        else:
            crossed = text.fillna("").map(lambda s: b_pattern.sub(" ", s))

        still_a_place = crossed.map(lambda s: bool(ref_pattern.search(s))).astype(bool)

        if o_pattern is None:
            names_the_area = pd.Series(False, index=text.index, dtype=bool)
        else:
            names_the_area = crossed.map(lambda s: bool(o_pattern.search(s))).astype(bool)

        # Keep: a seed survives, AND (no Kind A word, OR an area word overrules it).
        return still_a_place & (~kind_a_hit | names_the_area)

    return (place_filter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: reference set and search set, per arm

    The reference set is what the filter keeps at this point: the result of `place_filter()`
    with the exclusions, plus every sentence that names a toponym. Everything else in the arm is
    the search set.

    Toponym sentences go in the reference set because the filter does keep them. Like the
    seeds, the toponyms are deleted from the text before it is turned into a matrix, so the
    classifier has to learn from the remaining words.
    """)
    return


@app.cell
def _(
    arms,
    exclusion_a,
    exclusion_b,
    make_matcher,
    place_filter,
    reference_words,
    toponyms,
    veto_overrides,
):
    # one boolean Series per arm, True for the reference set and False for the search set
    _topo_hit = make_matcher(toponyms, fold_accents=True)

    reference_sets = {}
    print(f"{'arm':10} {'reference':>10} {'of arm':>8}   {'search':>10}   {'by toponym only':>16}")
    for _arm, _df in arms.items():
        _by_seed = place_filter(_df["text"], reference_words, exclusion_a, exclusion_b,
                                veto_overrides)
        _by_topo = _topo_hit(_df["text"])
        reference_sets[_arm] = _by_seed | _by_topo
        print(f"{_arm:10} {int(reference_sets[_arm].sum()):>10,} "
              f"{reference_sets[_arm].mean():>8.2%}   {int((~reference_sets[_arm]).sum()):>10,}   "
              f"{int((_by_topo & ~_by_seed).sum()):>16,}")

    _en = reference_sets["guest/en"].mean()
    _da = reference_sets["guest/da"].mean()
    print(f"\nen/da retention ratio: {_en / _da:.3f}   (reported, not a target)")
    return (reference_sets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps 3 to 6: discovery

    `discover()` is called once per arm, so the settings are the same for every arm. It follows
    King, Lam & Roberts with a few changes.

    In step 3 the lexicon terms are deleted, the text is tokenised, stopwords are dropped and
    the words are stemmed. This uses Python's `re` because of the RE2 problem, and a Snowball
    stemmer in the arm's own language. Stopwords are dropped before stemming so the list and
    the tokens match. Step 4 builds a document-term matrix of single words and word pairs. Step
    5 trains a lasso logistic regression on a sample, drawn without replacement and with a fixed
    seed, and then scores every sentence. Step 6 splits the search set into target and
    non-target and ranks words by the log-likelihood statistic. `reads_as` shows the most common
    real word behind each stem.
    """)
    return


@app.cell
def _(
    CountVectorizer,
    Counter,
    LogisticRegressionCV,
    SnowballStemmer,
    defaultdict,
    fold,
    gammaln,
    hashlib,
    make_pattern,
    np,
    pd,
    re,
    stopwords,
):
    import pickle
    import warnings

    import paths as _paths

    STEMMERS = {"en": SnowballStemmer("english"), "da": SnowballStemmer("danish")}
    STOPWORDS = {"en": set(stopwords.words("english")), "da": set(stopwords.words("danish"))}

    # a token is two or more letters in any alphabet, so ø, æ and å count
    _WORD = re.compile(r"[^\W\d_]{2,}")

    def preprocess(texts, lang, strip_pattern):
        """
        Step 3. Returns the stemmed strings and the forms behind each stem.

        strip_pattern deletes every lexicon term, so the classifier has to learn from other
        words. forms maps each stem to the real words behind it, for example
        forms["kvart"] gives {"kvarter": 869, "kvarteret": 80, ...}.
        """
        stem, stops = STEMMERS[lang].stem, STOPWORDS[lang]
        memo = {}                      # word to stem, so each distinct word is stemmed only once
        forms = defaultdict(Counter)
        out = []
        for text in texts:
            if strip_pattern is not None:
                text = strip_pattern.sub(" ", text)
            tokens = []
            for word in _WORD.findall(text.lower()):
                if word in stops:
                    continue
                s = memo.get(word)
                if s is None:
                    s = memo[word] = stem(word)
                forms[s][word] += 1
                tokens.append(s)
            out.append(" ".join(tokens))
        return out, forms

    def llik(t, nt, T, NT):
        """
        King, Lam & Roberts p. 979, vectorised. Higher means more unevenly split.

        t, nt   sentences in target and not-target that contain the term
        T, NT   all sentences in target and not-target
        Computed in logs (gammaln) to avoid overflow.
        """
        x1 = gammaln(t + 1) + gammaln(nt + 1) - gammaln(t + nt + 2)
        x2 = (gammaln(T - t + 1) + gammaln(NT - nt + 1)
              - gammaln(T - t + NT - nt + 2))
        return x1 + x2

    # bump when discover() changes, to ignore old cached results
    DISCOVER_VERSION = 3

    def discover(arm, df, lang, reference, strip_terms, min_df, threshold,
                 n_ref_max=25_000, seed=42):
        """
        Steps 3 to 6 for one arm. Returns a dict with:
          ranking   one row per stem or word pair: counts, lasso coefficient, llik, reads_as
          meta      sizes, chosen penalty, standard deviation of the predictions, ...
          target    boolean array aligned with df
          forms     the real words behind each stem
        """
        _key = hashlib.sha256(repr((
            DISCOVER_VERSION, arm, len(df), lang, sorted(strip_terms),
            min_df, threshold, n_ref_max, seed,
            hashlib.sha256(np.asarray(reference, dtype=bool).tobytes()).hexdigest(),
        )).encode()).hexdigest()[:16]
        _cache = _paths.INTERIM / f"klr_cache_{arm.replace('/', '-')}_{_key}.pkl"
        if _cache.exists():
            with open(_cache, "rb") as f:
                return pickle.load(f)

        # step 3: preprocess
        # delete both spellings of toponyms ('nørrebro' and 'norrebro')
        _strip = make_pattern(sorted(set(strip_terms) | {fold(t) for t in strip_terms}))
        stems, forms = preprocess(df["text"].fillna(""), lang, _strip)

        # step 4: document-term matrix
        # already tokenised and lowercased by preprocess
        vec = CountVectorizer(min_df=min_df, max_df=0.999, ngram_range=(1, 2),
                              token_pattern=r"\S+", lowercase=False)
        X = vec.fit_transform(stems)

        # step 5: lasso, trained on a sample and scored on everything
        ref = np.asarray(reference, dtype=bool)
        ref_idx, search_idx = np.flatnonzero(ref), np.flatnonzero(~ref)
        rng = np.random.default_rng(seed)
        # 2 search sentences per reference sentence, reference capped at n_ref_max
        n_ref = min(len(ref_idx), n_ref_max)
        n_search = min(2 * n_ref, len(search_idx))
        train = np.concatenate([rng.choice(ref_idx, n_ref, replace=False),
                                rng.choice(search_idx, n_search, replace=False)])

        # sklearn's default grid minus its three weakest penalties, which were never
        # chosen and took most of the fitting time
        _Cs = np.logspace(-4, 4, 10)[:7]         # 0.0001 ... 21.5

        clf = LogisticRegressionCV(
            Cs=_Cs, cv=5,
            l1_ratios=(1,),            # pure L1 (lasso)
            solver="liblinear",
            scoring="accuracy",
            max_iter=1000, random_state=seed, n_jobs=-1,
            use_legacy_attributes=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            clf.fit(X[train], ref[train].astype(int))

        p = clf.predict_proba(X)[:, 1]

        # step 6: target, not-target and the ranking
        target = ~ref & (p >= threshold)
        not_target = ~ref & (p < threshold)

        present = (X > 0).astype(np.int32)       # sentences containing a term, not term counts
        t_docs = np.asarray(present[target].sum(0)).ravel()
        nt_docs = np.asarray(present[not_target].sum(0)).ravel()
        r_docs = np.asarray(present[ref].sum(0)).ravel()
        T, NT = int(target.sum()), int(not_target.sum())
        coef = clf.coef_.ravel()

        ranking = pd.DataFrame({
            "target_docs": t_docs, "not_target_docs": nt_docs, "ref_docs": r_docs,
            "target_%": 100 * t_docs / max(T, 1),
            "not_target_%": 100 * nt_docs / max(NT, 1),
            "coef": coef,
        }, index=vec.get_feature_names_out())
        ranking = ranking[(ranking.target_docs > 0) | (ranking.not_target_docs > 0)].copy()
        ranking["target_larger"] = ranking["target_%"] >= ranking["not_target_%"]
        ranking["llik"] = llik(ranking.target_docs.values, ranking.not_target_docs.values, T, NT)
        # the commonest real word behind each stem
        ranking["reads_as"] = [
            " ".join(forms[s].most_common(1)[0][0] if forms[s] else s for s in term.split())
            for term in ranking.index
        ]

        result = {
            "ranking": ranking,
            "target": target,
            "forms": {s: dict(c.most_common(8)) for s, c in forms.items()},
            "meta": {
                "sentences": len(df), "reference": int(ref.sum()), "trained_on": len(train),
                "vocabulary": X.shape[1], "nonzero_coefs": int((coef != 0).sum()),
                "C": float(np.ravel(clf.C_)[0]),
                # ~0 means the model predicts a constant
                "p_std": round(float(p.std()), 3),
                "target": T, "not_target": NT,
            },
        }
        with open(_cache, "wb") as f:
            pickle.dump(result, f)
        return result

    return (discover,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The two settings

    `MIN_DF` is how many sentences a word must appear in before it becomes a column. It is the
    same number for every arm. `THRESHOLD` is the predicted probability above which a search set
    sentence counts as target.

    Before choosing them I checked that the top 50 candidates barely change with either setting.
    """)
    return


@app.cell
def _():
    # Top-50 candidates shared with this setting:
    #   min_df 5: 50/49/48 of 50 (host/en, guest/en, guest/da), min_df 15: 50/47/50
    #   threshold 0.5: 42/45/41 of 50
    MIN_DF = 10
    THRESHOLD = 0.25     # generous, since every candidate is read
    return MIN_DF, THRESHOLD


@app.cell
def _(
    ARM_LANG,
    MIN_DF,
    THRESHOLD,
    arms,
    discover,
    exclusion_a,
    exclusion_b,
    reference_sets,
    reference_words,
    toponyms,
):
    # First run ~8 minutes; later runs with the same inputs load from the cache.
    # Everything in the lexicon is deleted before vectorising, or it would be
    # proposed right back.
    _strip_terms = reference_words + exclusion_a + exclusion_b + toponyms

    discovery = {}
    for _arm, _lang in ARM_LANG.items():
        discovery[_arm] = discover(
            _arm, arms[_arm], _lang, reference_sets[_arm].values, _strip_terms,
            min_df=MIN_DF, threshold=THRESHOLD,
        )
        print(_arm, discovery[_arm]["meta"])
    return (discovery,)


@app.cell
def _(discovery, pd):
    # step 6: candidates, the three arms side by side
    def top_candidates(n=40):
        _cols = {}
        for _arm, _res in discovery.items():
            _r = _res["ranking"]
            _top = _r[_r.target_larger].sort_values("llik", ascending=False).head(n)
            _cols[_arm] = [f"{s}  ({w})" if s != w else s
                           for s, w in zip(_top.index, _top.reads_as)]
        return pd.DataFrame(_cols, index=range(1, n + 1))

    top_candidates()
    return


@app.cell
def _(discovery, pd):
    # step 6: words that mark absence (candidates for exclusion)
    # Most are topic words (host, clean, check-in). Only ones naming a place inside the
    # property or outside Copenhagen are useful exclusions.
    _cols = {}
    for _arm, _res in discovery.items():
        _r = _res["ranking"]
        _top = _r[~_r.target_larger].sort_values("llik", ascending=False).head(30)
        _cols[_arm] = list(_top.reads_as)
    pd.DataFrame(_cols, index=range(1, 31))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: read the candidates in context

    Four tools help with each candidate. `forms_of(stem, arm)` shows the real words behind a
    stem, because the lexicon holds words, not stems, and one stem can hide two words (`kvart`
    is both `kvarter` and "a quarter of an hour"). `count_across_arms([...])` shows where a word
    is used, since one lexicon is applied to all voices and a word that is fine in host text
    still has to be read in reviews. `see_in_context(term, arm)` and `what_it_adds(term, arm)`
    show sample sentences. `neighbours(term, arm)` shows the words next to a term, which is how
    Kind B phrases are found.

    Adopted words go into `reference_words_expanded`, or into an exclusion list, in the seed
    cell.

    Some things to watch for. Evaluative words like `great`, `lovely` or `dejlig` rank high
    because place sentences are often "great location, lovely flat", but they would also keep
    "lovely host". Words about the flat, like `apartment` or `lejlighed`, come along with "the
    apartment is close to...". And sometimes the phrase matters more than the word: `tæt` can
    mean close in any sense, while `tæt på` is about location.
    """)
    return


@app.cell
def _(discovery, pd):
    def forms_of(stem, arm):
        """The real words behind a stem (or each stem of a bigram), with counts."""
        _forms = discovery[arm]["forms"]
        return pd.DataFrame(
            [(s, w, n) for s in stem.split() for w, n in _forms.get(s, {}).items()],
            columns=["stem", "word", "count"],
        )

    return (forms_of,)


@app.cell
def _(forms_of):
    forms_of("away", "host/en")
    return


@app.cell
def _(count_across_arms):
    count_across_arms("convenient")
    return


@app.cell
def _(see_in_context):
    see_in_context("hjemmet", "guest/da")
    return


@app.cell
def _(arms, make_matcher, reference_sets):

    def what_it_adds(term, arm="guest/da", n=20, seed=1):
            """
            Sentences this term would ADD: it matches them and the filter does not
            already keep them. This is the sample to judge a candidate on.
            """
            df = arms[arm]
            hit = make_matcher([term])(df["text"])
            new = hit & ~reference_sets[arm]          # matches, and not already kept
            sub = df.loc[new]
            print(f"{term!r} in {arm}: {int(hit.sum()):,} sentences match, "
                    f"{int((hit & reference_sets[arm]).sum()):,} already kept, "
                    f"{len(sub):,} would be NEW "
                    f"({len(sub) / max(int(hit.sum()), 1):.0%} of its matches)")
            print(f"  the new ones come from {sub.listing_id.nunique():,} listings, "
                    f"{sub.host_id.nunique():,} hosts, "
                    f"{int(sub.is_platform_template.sum()):,} platform-template")
            if len(sub) == 0:
                    print("  -> adds nothing: every sentence it matches is already kept.")
            return sub["text"].sample(min(n, len(sub)), random_state=seed)

    return (what_it_adds,)


@app.cell
def _(what_it_adds):
    what_it_adds("placeret", "guest/da")
    return


@app.cell
def _(Counter, arms, make_matcher, make_pattern, pd, re, reference_sets):
    def neighbours(term, arm, scope="new", k=25):
        """
        The words that most often come directly before and after a term.
        If 'central' is followed by 'heating' 300 times, 'central heating' is a
        Kind B candidate.

        scope is "new" (matches not already kept), "kept" (matches already kept)
        or "all". '<start>' and '<end>' mean the term began or ended the sentence.
        """
        df = arms[arm]
        hit = make_matcher([term])(df["text"])
        if scope == "new":
            mask = hit & ~reference_sets[arm]
        elif scope == "kept":
            mask = hit & reference_sets[arm]
        elif scope == "all":
            mask = hit
        else:
            raise ValueError(f"scope must be 'new', 'kept' or 'all', got {scope!r}")

        # the same pattern the filter uses
        pattern = make_pattern([term])
        word = re.compile(r"\w+")

        before, after = Counter(), Counter()
        for text in df.loc[mask, "text"].fillna(""):
            # a sentence can contain the term more than once
            for m in pattern.finditer(text):
                left = word.findall(text[: m.start()])
                right = word.findall(text[m.end():])
                before[left[-1].lower() if left else "<start>"] += 1
                after[right[0].lower() if right else "<end>"] += 1

        n = int(mask.sum())
        print(f"{term!r} in {arm}, scope={scope!r}: {n:,} sentences, "
                f"{sum(after.values()):,} occurrences")

        # two independent rankings
        _b = before.most_common(k)
        _a = after.most_common(k)
        _b += [("", 0)] * (k - len(_b))
        _a += [("", 0)] * (k - len(_a))
        return pd.DataFrame(
            {"before": [w for w, _ in _b], "n before": [c for _, c in _b],
                "after":  [w for w, _ in _a], "n after":  [c for _, c in _a]},
            index=range(1, k + 1),
        )

    return (neighbours,)


@app.cell
def _(neighbours):
    neighbours("heart", "guest/en")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 9: the acceptance test

    The test is whether each concept covers English and Danish equally well. Overall retention
    is reported but has no target.

    The checks are the per-concept table together with the `checked` record of which word form
    sweeps were done; retention per arm and per area, since an area that the filter keeps much
    better than another would make area differences partly an effect of the filter; the English
    to Danish ratio by sentence length, where a ratio that rises with length points to missing
    vocabulary and a flat one to a real difference in what people write (for v1 it rose from
    1.10 to 1.88); and what v1 kept that the new filter drops.

    v1 is the original hand-written keyword list, run through `make_matcher` so the RE2 bug does
    not affect it.
    """)
    return


@app.cell
def _():
    # STEP 9: the sweep checklist. An empty string means not done yet.
    #   en: US/UK spellings, plurals, a distinct American word
    #   da: definite forms, plurals, adjective inflections, compounds
    SWEEPS = {
        "en": ["us/uk spelling", "plurals", "american word"],
        "da": ["definite forms", "plurals", "adjective inflection", "compounds"],
    }

    def _blank():
        return {_lang: {_sweep: "" for _sweep in _sweeps}
                for _lang, _sweeps in SWEEPS.items()}

    checked = {
        "identity": _blank(),
        "character": _blank(),
        "location": _blank(),
        "amenities": _blank(),
        "connections": _blank(),
        "proximity": _blank(),
        "green and blue space": _blank(),
    }

    # Filled from one automated scan: every guest/da word that contains a Danish term,
    # is not in the lexicon, and occurs in >= 5 sentences. Forms spelled differently
    # ('cafeer' for 'caféer') were checked by hand.
    _EN = ("n/a: English sweeps not run — no English concept ratio was flagged, and "
           "English has few forms; the lists already carry US/UK and plural forms")
    _DA = {
        "identity": "scan: adopted nærområde(t), lokalområdet, villakvarter, "
                    "boligkvarter, boligområde (~73 new); rejected city-level "
                    "hovedstadsområdet, københavnsområdet",
        "character": "scan: no missing forms in >= 5 sentences",
        "location": "scan: adopted velbeliggende, centraltbeliggende, velplaceret, byens, "
                    "(stor/midt)byen, villavej, sidegade, sidevej, cykelafstand, "
                    "motorvej(en), gadestøj, supercentralt (~130 new)",
        "amenities": "scan: adopted kaffebar(er); by hand: cafeer (137) and "
                     "spisested(er)(ne) (286). Remaining gap is scope (shops absent "
                     "in both languages), not missing forms",
        "connections": "scan: adopted havnebus(sen), vandbussen, busstop(pested…), "
                       "busforbindelse(r), transport* compounds, togsstation (~42 "
                       "new); rejected buss",
        "proximity": "scan: adopted tættere, tætte, supertæt (~6 new)",
        "green and blue space": "scan: adopted badestrand (4 new); rejected every "
                                "parkering* (flat feature)",
    }
    for _concept in checked:
        for _sweep in SWEEPS["en"]:
            checked[_concept]["en"][_sweep] = _EN
        for _sweep in SWEEPS["da"]:
            checked[_concept]["da"][_sweep] = _DA[_concept]
    return SWEEPS, checked


@app.cell
def _(all_concepts):
    # one concept, both languages
    _c = all_concepts["identity"]
    print("en:", _c["en"])
    print("da:", _c["da"])
    return


@app.cell
def _(Counter, arms, make_matcher, pd, re, reference_words, toponyms):
    def compounds_of(term, arm="guest/da", k=30):
        """
        Words that CONTAIN `term` but that a word-bounded match cannot see
        (\\bbutik\\b misses 'dagligvarebutik'). For the compounds sweep.

          count        sentences containing that word
          in_lexicon   already a reference term or toponym
        """
        df = arms[arm]
        pattern = re.compile(rf"\w*{re.escape(term.lower())}\w*")

        counts = Counter()
        for text in df["text"].fillna(""):
            # once per sentence
            for word in set(pattern.findall(text.lower())):
                if word != term.lower():
                    counts[word] += 1

        known = {t.lower() for t in reference_words} | {t.lower() for t in toponyms}
        n_exact = int(make_matcher([term])(df["text"]).sum())
        print(f"{term!r} in {arm}: {n_exact:,} sentences match it as a word; "
              f"{len(counts):,} other words contain it")
        return pd.DataFrame(
            [(w, n, w in known) for w, n in counts.most_common(k)],
            columns=["word", "count", "in_lexicon"],
        )

    return (compounds_of,)


@app.cell
def _(compounds_of):
    compounds_of("station", "guest/da")
    return


@app.cell
def _(SWEEPS, all_concepts, checked, concept_table, pd):
    # step 9, part 1: per-concept table and checklist
    _missing = set(all_concepts) - set(checked)
    _extra = set(checked) - set(all_concepts)
    if _missing or _extra:
        raise KeyError(f"checked is out of step with all_concepts — "
                       f"missing: {sorted(_missing)}, not a concept: {sorted(_extra)}")

    _rows = []
    for _concept in all_concepts:
        _row = {"concept": _concept}
        _open = []
        for _lang, _sweeps in SWEEPS.items():
            _done = [s for s in _sweeps if checked[_concept][_lang][s].strip()]
            _row[f"{_lang} swept"] = f"{len(_done)}/{len(_sweeps)}"
            _open += [f"{_lang}: {s}" for s in _sweeps if s not in _done]
        _row["open sweeps"] = "; ".join(_open)
        _rows.append(_row)

    acceptance = concept_table[["en n", "da n", "en %", "da %", "en/da", "flags"]].join(
        pd.DataFrame(_rows).set_index("concept")
    )

    _n_open = int((acceptance["open sweeps"] != "").sum())
    print(f"{len(acceptance) - _n_open} of {len(acceptance)} concepts fully swept.")
    print("en/da is the triage ranking — far from 1.0 says LOOK HERE, not FIX TO 1.0.")
    acceptance
    return


@app.cell
def _(V1_PLACE_TERMS, arms, make_matcher, pd, reference_sets):
    # step 9, part 2: retention per arm, v1 and v2
    # v2 is reference_sets, the finished filter. v1 is the frozen list, without exclusions
    # or folding.
    _v1_match = make_matcher(V1_PLACE_TERMS)
    v1_keep = {_arm: _v1_match(_df["text"]) for _arm, _df in arms.items()}
    v2_keep = reference_sets

    _rows = []
    for _arm, _df in arms.items():
        _rows.append({
            "arm": _arm,
            "sentences": len(_df),
            "v1 kept": int(v1_keep[_arm].sum()),
            "v1 %": round(100 * v1_keep[_arm].mean(), 2),
            "v2 kept": int(v2_keep[_arm].sum()),
            "v2 %": round(100 * v2_keep[_arm].mean(), 2),
        })
    retention = pd.DataFrame(_rows).set_index("arm")

    # guest arms only: there is no Danish host arm
    for _v in ("v1", "v2"):
        _ratio = retention.loc["guest/en", f"{_v} %"] / retention.loc["guest/da", f"{_v} %"]
        print(f"{_v} guest en/da: {_ratio:.3f}")
    print("Reported, not scored. A level gap is reportable; a COMPOSITION gap is the defect.")
    retention
    return v1_keep, v2_keep


@app.cell
def _(arms, np, pd, v2_keep):
    # step 9, part 2b: retention per area
    # 'da ±' is the 95% margin for the Danish rate, 1.96 * sqrt(p(1-p)/n). Sentences are
    # not independent, so read it as a lower bound.
    _cols = {}
    for _arm, _df in arms.items():
        _g = v2_keep[_arm].groupby(_df["neighbourhood"])
        _cols[f"{_arm} %"] = (100 * _g.mean()).round(2)
        _cols[f"{_arm} n"] = _g.size()

    by_area = pd.DataFrame(_cols)
    _p = by_area["guest/da %"] / 100
    by_area["da ±"] = (100 * 1.96 * np.sqrt(_p * (1 - _p) / by_area["guest/da n"])).round(2)
    by_area["en/da"] = (by_area["guest/en %"] / by_area["guest/da %"]).round(3)
    by_area = by_area.sort_values("guest/en %", ascending=False)

    for _arm in arms:
        _r = by_area[f"{_arm} %"]
        print(f"{_arm:9} areas range {_r.min():5.2f}% – {_r.max():5.2f}%  "
              f"(spread {_r.max() - _r.min():.2f} points)")
    by_area[["host/en %", "guest/en %", "guest/da %", "da ±", "en/da", "guest/da n"]]
    return


@app.cell
def _(arms, np, pd, v1_keep, v2_keep):
    # step 9, part 3: English to Danish ratio by length
    #   a ratio that rises with length points to missing vocabulary,
    #   a flat ratio to a difference in what people write
    _SENT_BINS = [0, 5, 10, 20, np.inf]
    _SENT_LABELS = ["1–5", "6–10", "11–20", "21+"]
    _REV_BINS = [0, 1, 3, 6, np.inf]
    _REV_LABELS = ["1", "2–3", "4–6", "7+"]

    def _by_bin(bin_of, name):
        """en/da per bin, for v1 and v2. bin_of(df) returns each sentence's bin."""
        _rates = {}
        for _arm in ("guest/en", "guest/da"):
            _df = arms[_arm]
            _bins = bin_of(_df)
            for _v, _keep in (("v1", v1_keep), ("v2", v2_keep)):
                _rates[(_arm, _v)] = _keep[_arm].groupby(_bins, observed=True).mean()
            _rates[(_arm, "n")] = _bins.value_counts()

        _out = pd.DataFrame({
            "v1 en/da": _rates[("guest/en", "v1")] / _rates[("guest/da", "v1")],
            "v2 en/da": _rates[("guest/en", "v2")] / _rates[("guest/da", "v2")],
            "v2 en %": 100 * _rates[("guest/en", "v2")],
            "v2 da %": 100 * _rates[("guest/da", "v2")],
            "da n": _rates[("guest/da", "n")],
        })
        _out.index.name = name
        return _out.round(3)

    by_sentence_length = _by_bin(
        lambda df: pd.cut(df["n_tokens_approx"], _SENT_BINS, labels=_SENT_LABELS),
        "sentence tokens",
    )

    # review length is the number of sentences in the same review (in this arm)
    by_review_length = _by_bin(
        lambda df: pd.cut(df.groupby("doc_id")["sentence_id"].transform("size"),
                          _REV_BINS, labels=_REV_LABELS),
        "review sentences",
    )

    print("By SENTENCE length (tokens):")
    print(by_sentence_length.to_string())
    print("\nBy REVIEW length (sentences):")
    print(by_review_length.to_string())
    for _name, _t in (("sentence", by_sentence_length), ("review", by_review_length)):
        for _v in ("v1", "v2"):
            _r = _t[f"{_v} en/da"]
            print(f"{_v} {_name}-length slope: {_r.iloc[0]:.2f} -> {_r.iloc[-1]:.2f}")
    return


@app.cell
def _(Counter, V1_PLACE_TERMS, arms, make_pattern, pd, v1_keep, v2_keep):
    # step 9, part 4: overlap with v1
    _rows = []
    for _arm in arms:
        _v1, _v2 = v1_keep[_arm], v2_keep[_arm]
        _rows.append({
            "arm": _arm,
            "both": int((_v1 & _v2).sum()),
            "v1 only (dropped)": int((_v1 & ~_v2).sum()),
            "v2 only (added)": int((~_v1 & _v2).sum()),
            "neither": int((~_v1 & ~_v2).sum()),
            "% of v1 kept by v2": round(100 * (_v1 & _v2).sum() / max(int(_v1.sum()), 1), 1),
        })
    overlap = pd.DataFrame(_rows).set_index("arm")

    # Which v1 terms the dropped sentences relied on (counted once per sentence).
    _v1_pattern = make_pattern(V1_PLACE_TERMS)
    _cols = {}
    for _arm, _df in arms.items():
        _counts = Counter()
        for _text in _df.loc[v1_keep[_arm] & ~v2_keep[_arm], "text"].fillna(""):
            _counts.update({_m.lower() for _m in _v1_pattern.findall(_text)})
        _cols[_arm] = [f"{_t} ({_n:,})" for _t, _n in _counts.most_common(15)]
    dropped_terms = pd.DataFrame(
        {_a: _c + [""] * (15 - len(_c)) for _a, _c in _cols.items()},
        index=range(1, 16),
    )

    print(overlap.to_string())
    print("\nThe v1 terms behind the DROPPED sentences, most frequent first:")
    dropped_terms
    return


@app.cell
def _(V1_PLACE_TERMS, arms, make_matcher, make_pattern, v1_keep, v2_keep):
    def dropped_by_v2(arm="guest/da", n=20, term=None, seed=1):
        """
        n sentences v1 kept and v2 drops, with the v1 terms that had kept them.
        term= restricts the sample to one v1 term.
        """
        df = arms[arm]
        mask = v1_keep[arm] & ~v2_keep[arm]
        if term is not None:
            mask = mask & make_matcher([term])(df["text"])
        sub = df.loc[mask, ["text", "neighbourhood"]]
        print(f"{arm}: {int(mask.sum()):,} dropped sentences"
              + (f" containing {term!r}" if term else ""))
        sub = sub.sample(min(n, len(sub)), random_state=seed).copy()
        pattern = make_pattern(V1_PLACE_TERMS)
        sub["v1 terms"] = [
            ", ".join(sorted({m.lower() for m in pattern.findall(t)})) for t in sub["text"]
        ]
        return sub

    return (dropped_by_v2,)


@app.cell
def _(dropped_by_v2):
    dropped_by_v2("guest/da", n=20)
    return


@app.cell
def _(dropped_by_v2, exclusion_a, exclusion_b, make_pattern):
    def why_dropped(arm="host/en", term=None, n=20, seed=1):
        """
        dropped_by_v2's sample, plus the exclusions found in each sentence. If both
        columns are empty, the sentence was dropped because v2 no longer has the v1 word.
        """
        sub = dropped_by_v2(arm, n=n, term=term, seed=seed)
        a_pattern = make_pattern(exclusion_a)
        b_pattern = make_pattern(exclusion_b)
        sub["kind A (veto)"] = [
            ", ".join(sorted({m.lower() for m in a_pattern.findall(t)})) for t in sub["text"]
        ]
        sub["kind B (cancel)"] = [
            ", ".join(sorted({m.lower() for m in b_pattern.findall(t)})) for t in sub["text"]
        ]
        return sub

    return (why_dropped,)


@app.cell
def _(why_dropped):
    why_dropped("guest/en")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 10: ship the filter

    The word lists and the keep rule are saved in `scripts/place_filter.py`. `05_corpora.py`
    imports that file, so all three voices are filtered by exactly the same rules. The discovery
    classifier is not part of it, since it only suggested words.

    The last cell checks on every run that the script and this notebook have the same lists and
    keep exactly the same sentences.
    """)
    return


@app.cell
def _(
    all_concepts,
    arms,
    exclusion_a,
    exclusion_b,
    reference_sets,
    reference_words,
    toponyms,
    veto_overrides,
):
    # scripts/ is already on sys.path from the first cell
    import place_filter as pf

    # 1. the lists
    _list_checks = {
        "concepts": (pf.CONCEPTS, all_concepts),
        "reference words": (pf.REFERENCE_WORDS, reference_words),
        "Kind A": (pf.EXCLUSION_A, exclusion_a),
        "Kind B": (pf.EXCLUSION_B, exclusion_b),
        "toponyms": (pf.TOPONYMS, toponyms),
        "veto overrides": (pf.VETO_OVERRIDES, veto_overrides),
    }
    _list_diffs = [_name for _name, (_ship, _nb) in _list_checks.items() if _ship != _nb]
    for _name, (_ship, _nb) in _list_checks.items():
        print(f"  {_name:16} {len(_ship):>4} entries   "
              f"{'same' if _name not in _list_diffs else 'DIFFERENT'}")

    # 2. the behaviour: same sentences kept, row for row
    print(f"\n{'arm':10} {'script keeps':>13} {'notebook keeps':>15} {'rows that differ':>17}")
    _behaviour_diffs = {}
    for _arm, _df in arms.items():
        _shipped = pf.is_place(_df["text"])
        _n_diff = int((_shipped != reference_sets[_arm]).sum())
        _behaviour_diffs[_arm] = _n_diff
        print(f"{_arm:10} {int(_shipped.sum()):>13,} {int(reference_sets[_arm].sum()):>15,} "
              f"{_n_diff:>17,}")

    if _list_diffs or any(_behaviour_diffs.values()):
        raise AssertionError(
            f"scripts/place_filter.py and 03_place_filter.py disagree — lists: {_list_diffs}, "
            f"rows: {_behaviour_diffs}. Edit both, or neither."
        )
    print("\nshipped instrument == notebook instrument: ok")
    return


if __name__ == "__main__":
    app.run()
