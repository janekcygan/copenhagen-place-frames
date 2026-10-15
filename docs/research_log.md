# Research log

COS-R434, Statistical NLP for Social Science, autumn 2026

Each entry has a plan, written before running, and a report, written after. Data: Inside
Airbnb, Copenhagen, snapshot of 30 June 2026. The log contains counts and keywords only, no
review text.

---

## Entry 1 — Is the place filter language-biased?

**17 Sep 2026.** The first place filter was a hand-written keyword list (v1). It is kept in
`03_place_filter.py` for comparison.

**Plan.** Apply v1 to all English and Danish guest sentences (1,233,159). Compare the share
kept per language.

**Report.** A Danish probe for `tæt på` returned 0 hits in 67,285 sentences, which is
impossible. The cause: pandas 3 runs `.str.contains` in the RE2 engine, where `\b` is an ASCII
word boundary, so `ø`, `å`, `æ` break word matching without an error.

| pattern | `.str.contains` | Python `re` |
|---|---|---|
| `\bsø\b` (da) | 2,764 | 15 |
| `\bpå\b` (da) | 64 | 9,130 |
| `\bøsterbro\b` (en) | 0 | 419 |

With Python's `re`: English 26.19% kept, Danish 17.24%, ratio 1.52. The earlier figure of 1.27
came from the bug. Without the single word `location` the ratio drops to 1.17.

**Conclusions.** The filter favours English more than recorded, and one word carries most of
the gap. The remaining gap may be a missing Danish vocabulary, a real difference in what
guests write, or Airbnb's "Location" rating category. The filter was rebuilt (`03`) with a
Unicode-safe matcher, and the ratio is treated as something to decompose, not a target.

---

## Entry 2 — Do hosts and guests propose different place vocabulary?

**17 Sep 2026.** Planned as two KLR runs on host/en and guest/en with identical settings.
Not run. Replaced by Entry 3, which asks the same question by counting words.

---

## Entry 3 — Do the three voices describe place with the same words?

**29 Sep 2026** · `04_exp1_vocabulary.py`

**Plan.** Frequency counts of the final lexicon (188 words in seven concepts, 103 toponyms) in
the sentences the filter keeps: host/en 52,939, guest/en 346,692, guest/da 16,658. Every
figure is a share of the voice's kept sentences, since the voices differ in size by up to 21×.
Only host/en vs guest/en holds language constant.

**Prediction.** I expect more similarity between host/en and guest/en than between guest/en and guest/da. I expect more marketing-style concepts to appear in the host arm, such as 'central' or 'in the heart of'. I expect the guests to use more adjectives like 'nice' or 'cosy' and use place-related vocabulary in a way that reflects their navigating experinence, e.g. 'located' or 'walk'

**Report.** Ran as planned. Change: toponyms were 98 of the 164 words in Table 3, so Table 3 is
also shown without toponyms (3b).

Share of kept sentences with a word from each concept (%):

| concept | host/en | guest/en | guest/da |
|---|---|---|---|
| location | 60.0 | 67.7 | 59.1 |
| toponyms | 42.2 | 8.0 | 15.1 |
| connections | 30.4 | 27.0 | 25.3 |
| proximity | 22.6 | 25.5 | 25.9 |
| identity | 21.3 | 20.3 | 20.2 |
| amenities | 18.6 | 15.1 | 7.1 |
| green and blue space | 10.9 | 2.6 | 2.4 |
| character | 6.2 | 1.0 | 1.1 |

Most used word: host/en `city` (21.0%), guest/en `location` (32.7%), guest/da `tæt` and
`beliggenhed` (19.4% each).

Host vs guest, English, without toponyms. Host-leaning: `green` (16×), `lakes` (7×), `cafés`
(7×), `heart of` (5×), `beach` (4×). Guest-leaning: `walkable`, `location` (0.13×),
`neighbourhood`, `centre`, `near`. With toponyms, the host side is all small streets and
parks, and misspelled toponyms (`nyhaven`, `fredriksberg`) lean towards guests.

Table 3c (guest/da vs guest/en) was added later and is not reported yet.


---

## Entry 4 — Do BERTopic's default settings fit this corpus?

**29 Sep 2026** · `06_exp2_BERTopic_test.py`

**Plan.** Fit BERTopic three times on the same 24,000 sentences (8,000 per voice), UMAP seed
fixed:

| arm | encoder | `min_cluster_size` | topic words |
|---|---|---|---|
| A default | `all-MiniLM-L6-v2` | 10 | unigrams, no stopwords |
| B multilingual | `paraphrase-multilingual-MiniLM-L12-v2` | 10 | unigrams, no stopwords |
| C tuned | same as B | 70 | en+da stopwords, bigrams, `min_df=5` |

C is the first tuned version of `07`, before mpnet and name masking.

**Prediction.** I expect around 40 topics. I expect english-only encoder to generate some mostly-danish topics. Most topics will centre around specific place-describing categories, but almost all of them will be pragmatic, surface-level

**Report.** Ran as planned. C reproduced 07's 37 topics.

| | A | B | C |
|---|---|---|---|
| topics | 234 | 249 | 37 |
| outliers | 41.8% | 42.3% | 22.4% |
| guest/da in Danish-only topics | 99.6% | 22.0% | 10.1% |
| guest/en in English-only topics | 95.1% | 26.6% | 6.2% |
| stopwords among topic words | 13.0% | 9.1% | 0 (removed) |

- A → B: with the English encoder, Danish and English guests almost never share a topic. The
  multilingual encoder mixes them.
- B → C: topics fall from 249 to 37 and outliers halve. Guest/da goes from fewest outliers
  (41.2%) to most (28.9%, host/en 15.5%).
- Part of the drop in language isolation from B to C is arithmetic: a 10-sentence cluster is
  all-Danish by chance more easily than a 70-sentence one. A → B has no such problem.

---

## Entry 5 — Where do the kept discovery topics concentrate?

**8 Oct 2026** · `08_exp3_topics_distributions.py`

**Plan.** The 24 topics marked as not frame-less (9,450 sentences). For each grouping (voice,
voice × area, room type, price, hexagon map), each topic's share and lift (share ÷ expected).
Expected is built within voice, because the pool shrank each voice by a different factor.
Cells under 100 sentences get no ranking. Price: entire homes, quartiles over listings.

**Prediction.** Not written.

**Report.** Ran as planned, with three changes made before reading results: a lift needs at
least 2 expected sentences; a topic needs at least 5 sentences to win a hexagon; room type and
price pool the voices, with expected built from each group's voice mix.

- Largest topic: restaurants/cafés for host/en (32%) and guest/en (39%); apartment/location
  for guest/da (22%).
- Strongest area cells: host · Amager Vest · Islands Brygge 10.6×; host · Amager Øst · beach
  7.3×; guest/en · Indre By · noise 2.1×.
- Room type: the room topic is 7.9× in private rooms and 0.2× in entire homes. In entire
  homes, no other topic is more than ±10% from expected.
- Price separates no topic sharply (all lifts 0.49–1.52×).
- The map follows geography where a topic names a place (beach, Islands Brygge,
  Christianshavn, lakes). Toponym topics partly restate the listing's address; beach, lakes and
  noise are the non-toponym topics that concentrate.

---

## Candidate entries

- Is the remaining en/da gap the filter or the writers? Test: the en/da ratio by sentence
  length. A rising ratio points to missing vocabulary, a flat one to a real difference. v1
  rose from 1.10 to 1.88.
- Does the filter keep areas at different rates?
- What share of kept sentences say nothing about the area, by voice? (`05` draws the sample.)
