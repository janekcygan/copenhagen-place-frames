"""
The place filter, built with King, Lam & Roberts' keyword discovery in
notebooks/03_place_filter.py.

All three voices are filtered with this one file, so the filter cannot differ
between notebooks. 03_place_filter.py checks on every run that this file and
the notebook give the same answers.

    import place_filter as pf
    df["is_place"] = pf.is_place(df["text"])

Only the word lists and the keep rule are here. The discovery classifier only
suggested candidate words, and the lists are what was chosen from them.

What counts as place: a sentence refers to somewhere in Copenhagen outside the
property itself, such as an area, a landmark, a route, or a facility a guest
would use. It does not have to say what the area is like. Places outside
Copenhagen, places inside the property, getting into and around the property,
and journeys to or from the city do not count.

The keep rule: a sentence is kept if it names a toponym (matched with the
Danish letters folded), or if a reference word is still there after the Kind B
phrases are crossed out and the sentence either has no Kind A word or still has
an area-naming word (VETO_OVERRIDES). In other words, a sentence is only removed
when the thing we object to is the reason it was matched.

Never match with pandas .str.contains. pandas stores strings with PyArrow,
which runs regular expressions in RE2, where \\b only knows ASCII letters:
\\bpå\\b finds 64 Danish sentences instead of 9,130. Everything here compiles
with Python's `re` and applies it with .map().

Matching never changes the stored text. The crossed-out and folded strings are
only used for the test.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# THE LEXICON: the same lists as 03_place_filter.py's seed cell (seeds,
# rounds 1-2 and the step-9 sweeps), by concept. Edit both or neither: 03's
# parity check stops on any difference.

CONCEPTS = {
    'identity': {
        "en": [
            'neighbourhood', 'neighbourhoods', 'neighborhood', 'neighborhoods',
            'district', 'districts', 'quarter', 'quarters', 'area', 'areas',
        ],
        "da": [
            'kvarter', 'kvarteret', 'kvarterer', 'kvartererne', 'bydel', 'bydelen',
            'område', 'området', 'områder', 'områderne', 'nabolag', 'nabolaget',
            'nabolagene', 'nærområdet', 'nærområde', 'lokalområdet', 'villakvarter',
            'boligkvarter', 'boligområde',
        ],
        "why": (
            'delimits an area as a specific place in the city'
        ),
    },
    'character': {
        "en": [
            'lively', 'green',
        ],
        "da": [
            'livlig', 'livligt', 'livlige', 'grøn', 'grønt', 'grønne',
        ],
        "why": (
            "describes the area's character, not a flat's features"
        ),
    },
    'location': {
        "en": [
            'location', 'locations', 'located', 'centre', 'centres', 'center',
            'centers', 'distance', 'distances', 'surroundings', 'street', 'streets',
            'road', 'roads', 'central', 'city', 'heart of',
        ],
        "da": [
            'beliggenhed', 'beliggenheden', 'beliggende', 'omgivelse', 'omgivelser',
            'omgivelserne', 'gade', 'gaden', 'gader', 'gaderne', 'vej', 'vejen',
            'veje', 'vejene', 'centrum', 'centret', 'afstand', 'afstanden', 'central',
            'centralt', 'centrale', 'byen', 'hjertet af', 'placering', 'placeringen',
            'placeret', 'velbeliggende', 'centraltbeliggende', 'velplaceret', 'byens',
            'midtbyen', 'storbyen', 'storbyens', 'villavej', 'sidegade', 'sidevej',
            'cykelafstand', 'supercentralt', 'motorvej', 'motorvejen', 'gadestøj',
        ],
        "why": (
            "describes the placement in relation to the urban space; 'central' is "
            "deliberately absent because it may refer to the flat's features rather "
            "than the area's character"
        ),
    },
    'amenities': {
        "en": [
            'bar', 'bars', 'cafe', 'cafes', 'café', 'cafés', 'restaurant',
            'restaurants', 'bakery', 'bakeries',
        ],
        "da": [
            'bar', 'baren', 'barer', 'café', 'caféen', 'caféer', 'caféerne',
            'restaurant', 'restauranten', 'restauranter', 'restauranterne', 'bageri',
            'bagerier', 'kaffebar', 'kaffebarer', 'cafeer', 'spisested', 'spisesteder',
            'spisestederne',
        ],
        "why": (
            'describes amenities of the area, not the flat'
        ),
    },
    'connections': {
        "en": [
            'metro', 'subway', 'tube', 'station', 'stations', 'bus', 'buses', 'stop',
            'stops', 'public', 'transport',
        ],
        "da": [
            'bus', 'bussen', 'busser', 'busserne', 'metro', 'metroen', 'metrostation',
            'metrostationen', 'metrostationer', 'station', 'stationen', 'stationer',
            'stationerne', 'stoppested', 'togstation', 'togstationen', 'togstationer',
            'offentlig', 'offentligt', 'offentlige', 'transport', 'havnebus',
            'havnebussen', 'vandbussen', 'busstop', 'busstoppested', 'busstoppestedet',
            'busstoppesteder', 'busforbindelse', 'busforbindelser',
            'transportmuligheder', 'transportmidler', 'transporten',
            'kollektivtransport', 'bustransport', 'togsstation',
        ],
        "why": (
            'describes how the flat connects to the city'
        ),
    },
    'proximity': {
        "en": [
            'close', 'near', 'nearby', 'around the corner', 'walkable', 'walkability',
        ],
        "da": [
            'tæt', 'i nærheden', 'om hjørnet', 'gåafstand', 'tættere', 'tætte',
            'supertæt',
        ],
        "why": (
            'states that somewhere is near or reachable on foot — the relation to a '
            'place rather than the place itself'
        ),
    },
    'green and blue space': {
        "en": [
            'beach', 'lake', 'lakes', 'parks',
        ],
        "da": [
            'strand', 'stranden', 'sø', 'søen', 'søer', 'søerne', 'parker',
            'badestrand',
        ],
        "why": (
            'names public green or water space in the city'
        ),
    },
}

# Kind A names a place that is out of scope (inside the property, outside Copenhagen). It vetoes.
EXCLUSION_A = [
    'attic', 'back yard', 'back yards', 'backyard', 'backyards', 'baggård', 'baggårde',
    'baggården', 'baggårdene', 'baghave', 'baghaven', 'baghaver', 'baghaverne',
    'basement', 'berlin', 'corridor', 'corridors', 'courtyard', 'courtyards',
    'elevator', 'elevatoren', 'elevatorer', 'elevatorerne', 'elevators', 'elsinore',
    'entrance', 'entrances', 'gård', 'gårde', 'gården', 'gårdene', 'gårdhave',
    'gårdhaven', 'gårdhaver', 'gårdhaverne', 'hallway', 'hallways', 'helsingor',
    'helsingør', 'indgang', 'indgange', 'indgangen', 'indgangene', 'key box',
    'key boxes', 'key code', 'keybox', 'korridor', 'korridoren', 'korridorer',
    'korridorerne', 'kælder', 'kælderen', 'kældre', 'kældrene', 'legeområde',
    'legeområder', 'legeområderne', 'legeområdet', 'lift', 'lifts', 'lock box',
    'lockbox', 'louisiana', 'malmo', 'malmö', 'malmø', 'nøgle', 'nøgleboks',
    'nøgleboksen', 'nøglen', 'nøgler', 'nøglerne', 'reception', 'receptionen',
    'receptions', 'roskilde', 'siddeområde', 'siddeområder', 'siddeområderne',
    'siddeområdet', 'siddeplads', 'siddepladsen', 'siddepladser', 'siddepladserne',
    'stair', 'staircase', 'staircases', 'stairs', 'sverige', 'sweden', 'trappe',
    'trappen', 'trapper', 'trapperne',
]

# Kind B is one of our own seeds used in a non-place sense. It only cancels its own match.
EXCLUSION_B = [
    'bar lidt præg af', 'bar over', 'bar præg af', 'bar tydeligt præg af',
    'bathroom area', 'bathroom areas', 'common area', 'common areas', 'dining area',
    'dining areas', 'garden area', 'garden areas', 'green garden', 'green plants',
    'grøn baggård', 'grøn gårdhave', 'grønne planter', 'hele vejen', 'kitchen area',
    'kitchen areas', 'living area', 'living areas', 'outdoor area', 'outdoor areas',
    'outside area', 'outside areas', 'play area', 'play areas', 'på en vej',
    'seating area', 'seating areas', 'stop by', 'stop me', 'storage area',
    'storage areas', 'central heating',
]

# Named places in Copenhagen. Matched FOLDED, so 'nørrebro' also matches 'Norrebro'.
TOPONYMS = [
    'indre by', 'inner city', 'nørrebro', 'vesterbro', 'østerbro', 'frederiksberg',
    'amager', 'amagerbro', 'valby', 'vanløse', 'brønshøj', 'husum', 'bispebjerg',
    'nordvest', 'enghave', 'sydhavn', 'sydhavnen', 'christianshavn', 'islands brygge',
    'island brygge', 'nordhavn', 'ørestad', 'refshaleøen', 'reffen', 'kødbyen',
    'meatpacking', 'meat packing', 'carlsberg', 'carlsbergbyen', 'carlsberg byen',
    'sluseholmen', 'frederiksstaden', 'latinerkvarteret', 'latin quarter', 'holmen',
    'christiania', 'brumleby', 'tivoli', 'nyhavn', 'strøget', 'rosenborg',
    'amalienborg', 'christiansborg', 'little mermaid', 'den lille havfrue',
    'round tower', 'rundetårn', 'marble church', 'marmorkirken', 'torvehallerne',
    'town hall', 'city hall', 'royal arena', 'bella center', 'bella centre',
    'fisketorvet', 'kongens nytorv', 'kgs nytorv', 'rådhuspladsen', 'enghave plads',
    'sankt hans torv', 'skjolds plads', 'nuuks plads', 'bopa plads', 'gammel strand',
    'trianglen', 'nørrebrogade', 'vesterbrogade', 'østerbrogade', 'amagerbrogade',
    'jægersborggade', 'værnedamsvej', 'istedgade', 'elmegade', 'gammel kongevej',
    'sønder boulevard', 'fasanvej', 'nørreport', 'østerport', 'hovedbanegården',
    'svanemøllen', 'dybbølsbro', 'flintholm', 'fælledparken', 'kongens have',
    "king's garden", 'kings garden', 'assistens', 'superkilen', 'utterslev mose',
    'amager strand', 'amager strandpark', 'amager fælled', 'søndermarken',
    'lergravsparken', 'valbyparken', 'nørrebroparken', 'nyhaven', 'christiana',
    'fredriksberg', 'fredericksberg', 'vestebro', 'christianhavn',
]

# DERIVED LISTS, computed from the lexicon rather than typed, so they cannot drift.

# The flat list the matcher needs. A set first, because some words serve both
# languages (restaurant, metro, central) and appear on both sides.
REFERENCE_WORDS = sorted({t for c in CONCEPTS.values() for t in c["en"] + c["da"]})


def _contains_word(term: str, phrase: str) -> bool:
    """Does `phrase` contain `term` as a whole word? ('play area' contains 'area')."""
    return bool(re.search(rf"\b{re.escape(term)}\b", phrase, flags=re.IGNORECASE))


# Which seeds can overrule a Kind A veto:
# every seed in `location` or `identity`, unless it has an in-property sense,
# which the lexicon records by giving it a Kind B entry ('area' has 'seating
# area'). Add a Kind B phrase for a seed and it leaves this list by itself.
_AREA_NAMING = sorted(set(
    CONCEPTS["location"]["en"] + CONCEPTS["location"]["da"]
    + CONCEPTS["identity"]["en"] + CONCEPTS["identity"]["da"]
))
VETO_OVERRIDES = [
    seed for seed in _AREA_NAMING
    if not any(_contains_word(seed, b) for b in EXCLUSION_B)
]


def _check_kind_a() -> None:
    """
    A Kind A term must never contain a seed. If it did, the term could be the
    very reason a sentence was retained, and vetoing on it would delete a
    sentence on the strength of a match we are declaring invalid. Such a term
    belongs in Kind B. Checked at import, so a bad edit fails loudly.
    """
    misfiled = [(a, s) for a in EXCLUSION_A for s in REFERENCE_WORDS if _contains_word(s, a)]
    if misfiled:
        raise AssertionError(f"Kind A term(s) contain a seed; move them to Kind B: {misfiled}")


_check_kind_a()


# THE MATCHER

def fold(s: str) -> str:
    """
    Turns 'Nørrebro' into 'norrebro', so one toponym entry matches the spellings
    guests actually type.

    ø, æ and å have no Unicode decomposition, so NFKD leaves them alone and they
    are replaced by hand first. NFKD then splits 'é' into 'e' plus an accent mark,
    and the accent marks are dropped.

    Only used for TOPONYMS: guests without a Danish keyboard write 'Norrebro', but
    ordinary Danish words are written by people who have the keys (folding
    'område' in Danish text adds 1 hit).
    """
    s = s.lower().replace("ø", "o").replace("æ", "ae").replace("å", "aa")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def make_pattern(terms, fold_accents: bool = False):
    """
    One compiled, case-insensitive, word-bounded pattern for a list of terms.

    Returns None for an empty list. The pattern an empty list would otherwise
    build, \\b(?:)\\b, matches at EVERY word boundary, so it would keep 100% of
    the corpus without complaint. None forces the caller to decide what
    "empty" means.
    """
    if not terms:
        return None
    prepared = [fold(t) if fold_accents else t for t in terms]
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in prepared) + r")\b",
        flags=re.IGNORECASE,
    )


def _hits(pattern, text: pd.Series, fold_accents: bool = False) -> pd.Series:
    """A clean boolean Series: does `pattern` occur in each sentence? Missing text counts as False."""
    if pattern is None:
        return pd.Series(False, index=text.index, dtype=bool)
    if fold_accents:
        return text.fillna("").map(lambda t: bool(pattern.search(fold(t)))).astype(bool)
    return text.fillna("").map(lambda t: bool(pattern.search(t))).astype(bool)


def place_filter(text: pd.Series, reference=REFERENCE_WORDS, kind_a=EXCLUSION_A,
                 kind_b=EXCLUSION_B, overrides=VETO_OVERRIDES) -> pd.Series:
    """
    The lexical half of the keep rule: design (c) with the R2′ overrides.

    Kind B phrases are crossed out first; then the reference words and the
    overrides are tested on that SAME crossed-out text. The order matters:
    'seating area' must be gone before we ask whether an area word is present.
    Crossing out can only remove matches, never create one, because every
    pattern is \\b-delimited.

    The arguments default to the shipped lists. They are parameters so that
    03_place_filter.py can call this with its own lists and compare.
    """
    ref_pattern = make_pattern(reference)
    b_pattern = make_pattern(kind_b)
    o_pattern = make_pattern(list(overrides))
    kind_a_hit = _hits(make_pattern(kind_a), text)

    if b_pattern is None:
        crossed = text.fillna("")
    else:
        crossed = text.fillna("").map(lambda s: b_pattern.sub(" ", s))

    still_a_place = _hits(ref_pattern, crossed)
    names_the_area = _hits(o_pattern, crossed)

    # Keep: a seed survives, AND (no Kind A word, OR an area word overrules it).
    return still_a_place & (~kind_a_hit | names_the_area)


def toponym_hit(text: pd.Series) -> pd.Series:
    """Does the sentence name a Copenhagen toponym? Folded, so spelling-tolerant."""
    return _hits(make_pattern(TOPONYMS, fold_accents=True), text, fold_accents=True)


def is_place(text: pd.Series) -> pd.Series:
    """
    THE FILTER. True where a sentence is place-describing (see THE CONSTRUCT).

    A toponym keeps a sentence outright (a toponym already beat the veto in
    the design); otherwise the lexical keep rule decides. ~15 s per million
    sentences.
    """
    return place_filter(text) | toponym_hit(text)
