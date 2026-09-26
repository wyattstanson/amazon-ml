# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AryanshSinha  
**Team Members:** Aryansh Sinha  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

A two-stage pipeline — **IDF-weighted token meta-blocking** followed by a **two-stage
LightGBM matcher** — resolves every Source-1 business against ~10M Source-2/3 records on a
laptop CPU, fully offline. Blocking keeps **0.9782** of all true training pairs
in ~54.1 candidates per entity (oracle ceiling F0.5 0.9926); the matcher reaches a
cross-validated **macro F0.5 of 0.9820** on all 2.2M training entities (rule-based
baseline: 0.8525). The main technical contributions are cross-field conjunction tokens
that make generic Indian business names blockable, a built-in transliterator for all nine
Indic scripts, "support" features that expose planted sibling businesses, and a stage-2
model that reasons over the whole candidate graph (competition between S1s for a record,
probability-weighted cluster support, similarity to an S1's other likely matches) instead
of judging pairs in isolation.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the full training data (numbers are exact counts):

| observation | value | design consequence |
|---|---|---|
| train S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 | columnar memory-mapped store; vectorised numpy/scipy throughout |
| test S1 / S2 / S3 rows | 1,732,544 / 4,887,273 / 5,082,316 | same pipeline, same code path |
| S1 entities with no true match (singletons) | 5.6 % | abstention matters, but recall drives 94 % of the score |
| true matches per S1 | mode 3, max 11 (≤ 5 from S2, ≤ 6 from S3) | many-to-one; per-source candidate depth ≥ 6 |
| S2/S3 records owned by more than one S1 | 0 of 7,638,365 | exclusivity: a record is assigned to at most one S1 |
| S2/S3 records matching no S1 | 2,681,854 (27 %) | a large distractor pool; ~30 % share a house number + street word with some S1 |
| matches crossing countries | 0 | blocking partitioned by country (an open set of labels) |
| India S2 / S3 names in Indic scripts | 23 % / 12 %, all nine major Brahmic scripts | rule-based transliteration built into normalisation |
| native-script address text | 16 distinct strings (all state names) | exact alias table after transliteration |
| test countries | US 38 %, India 47 %, France 15 % (France absent from training) | no country feature in the model; French conventions only in normalisation |

Noise patterns observed in real clusters: typos and leetspeak (`Wils0n`, `5mart`), random
accents (`Córp`), junk prefixes (`***`, `--`, `<<`, `##`), legal suffixes
added/removed/bracketed/moved to the front (`LLC Moncada`, `[Llc]`), a spurious trailing
word (`… Center`), duplicated words, word reordering, domain-style names
(`schroederprinting.com`, `pediatricdentistryphysicianscom`), former-name aliases
(`Vantagedova fka Pioneer Redwood L.L.C.`), completely unrelated names linked only by
address; addresses reordered by component, `St`/`Street` and even `St`→`Saint`, state
code ↔ full name ↔ native script, house-number ranges and typos, fillers (`null`, `N/A`,
`<NULL>`), and empty addresses (3.4 % of S2/S3).

The distractors include deliberately planted **siblings**: records with the same or a
related name at a neighbouring house number on the same street (`2068` vs `2070
Northampton St`), which no pair-level similarity can separate from a true match with a
number typo.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + F0.5-tuned decision layer  
**Core Innovation:** cross-field conjunction blocking tokens for generic names; candidate-graph
("support" and "competition") features that model siblings and rival claimants; built-in
Indic transliteration and dictionary-free word segmentation — all without external data.

Pipeline (one command, `python -m ber all`):

1. **Normalise** every record once (multi-process): Unicode NFKC → Indic transliteration →
   accent stripping → case folding; names: legal-form canonicalisation and separation,
   leetspeak undo, alias (`fka`/`dba`) extraction, domain-name detection, word
   segmentation of concatenated names; addresses: component-aware tokenisation,
   street-type/directional/ordinal canonicalisation (country-specific overlays, e.g.
   French `R.`→`rue` only for France), state/region codes, house/unit code extraction.
2. **Block** with IDF-weighted token meta-blocking (section 3).
3. **Featurise** every candidate pair: 54 language-agnostic features (section 4).
4. **Match** with stage-1 LightGBM → stage-2 LightGBM over candidate-graph features
   (competition, cluster support, neighbour similarity), trained with S1-grouped 3-fold
   cross-fitting.
5. **Decide**: per-S1 selection tuned for macro F0.5 on out-of-fold probabilities, with
   exclusivity (each S2/S3 record to at most one S1).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** typed tokens hashed to 32-bit ids, scored with IDF weights over
  Sources 2+3 of the same country:
  name words, consonant-class phonetic keys (Soundex-like, robust to transliteration and
  Tamil's unmarked voicing), 5-character prefixes, name bigrams and phonetic bigrams, the
  whole space-free name, address words, address codes, house-number+street bigrams, and two
  **cross-field conjunctions**: phonetic-name-key × address-word and address-code ×
  address-word. Score(S1, r) = Σ IDF of shared tokens, computed as sparse matrix products
  per country; tokens held by more than 5,000 records are dropped; the top 20 records per
  S1 per source are kept, plus the top 8 among *empty-address* records per source (these
  can only score on their name, so they are ranked against each other).
- **Candidate pairs generated:** 119,463,846 for training (54.1 per S1);
  92,992,637 for test (53.7 per S1). Reduction ratio vs. the full
  cross product: 99.99946 % on test.
- **How true matches were not lost:** blocking recall was measured against the ground
  truth after every change and every miss pattern was diagnosed on real examples:

| step | pair recall | oracle F0.5 |
|---|---|---|
| strategy A — standard exact-key blocking (block cap 2000) | 0.885 | 0.953 (at 143 pairs/S1) |
| strategy B — unigram tokens only | 0.858 | 0.934 |
| + cross-field conjunction tokens | 0.932 | 0.977 |
| + name/phonetic bigrams, whole-name token, empty-address channel | 0.9647 | 0.9887 |
| + word segmentation of concatenated names, zero-run code normalisation | 0.9726 | 0.9910 |
| deeper lists: 16 per source / 6 empty-address (42.9 pairs/S1) | 0.9761 | 0.9922 |
| deeper lists: 20 per source / 8 empty-address (54.1 pairs/S1), adopted | 0.9782 | 0.9930 |
| **final, full training set (2.2M S1)** | **0.9782** | **0.9926** |

(Rows 1–7 on fixed random samples of 20k–50k training S1 against the full S2/S3 pool;
see `EXPERIMENTS.md`.) Final recall by country: US 0.9854, India
0.9673; every true match of an entity is found for 93.2 % of
non-singleton entities.

Key diagnosis: generic Indian names (`shivam builders`, `raj construction`: name words in
10–20k records) and city words (`kolkata` 80k, `howrah` 166k records) are individually
uninformative, so any frequency cap removed exactly the decisive evidence; the
*combination* "name near place" is rare, which is what the conjunction tokens encode.

---

## 4. Matching Model

**Features used (54 pair features + 10 stage-2 features):**
- Name features: rapidfuzz ratio, token-sort, token-set, partial ratio and Jaro-Winkler on
  the normalised core name; ratio / partial ratio on the space-free name (domains,
  concatenations); full-name ratio; best similarity to a former-name alias; ratio of
  phonetic-key strings (transliterated names); IDF-weighted overlap of name tokens —
  shared weight, share of each side, rarest shared token, rarest *missing* token (the
  signal that separates generic from distinctive names); phonetic-token overlap; legal-form
  relation.
- Address features: ratio / token-sort / token-set on the order-normalised address; IDF
  overlap of address words and codes; street-bigram overlap; state agreement; house/unit
  number agreement (first number contained, number-set Jaccard, missing/extra numbers,
  first-code equality and edit distance, code Jaccard); empty-address flag.
- Other: candidate flags (domain name, Indic script, alias, source); blocking score, rank
  and relative score; how many S1s compete for the record and this S1's rank among them;
  **support features** — how many of the S1's *other* candidates share this candidate's
  house code / name (a coherent sibling entity vs a one-off typo);
  **stage-2 competition features** from out-of-fold stage-1 probabilities — rank and share
  of the S1's best, gap to the S1's next candidate, expected number of matches of the S1,
  the best competing S1's probability for the same record and the gap to it;
  **probability-weighted cluster support** — the stage-1 mass of the S1's other candidates
  sharing this record's house code, name or address, the mass of the strongest *rival* code
  cluster (a planted sibling entity), the best probability among the S1's other-source
  candidates with the same code, per-source rank and mass, the S1's mass on empty vs
  non-empty candidates, and how many S1s claim the record with p > 0.1;
  **neighbour similarity** — name (token-set) similarity to the S1's two most probable other
  candidates, their probabilities, and address similarity to the first (true matches of one
  entity resemble each other; a namesake resembles only the S1).

**Model type:** two LightGBM binary classifiers (MIT licence; 303,600 leaf/split
parameters in total — far below the 8B limit). Stage 1 uses the pair features; stage 2
adds the candidate-graph features. Both are trained with S1-grouped 3-fold cross-fitting
(every entity's candidates are scored by a model that never saw that entity), each fold
model on a 40 % S1-entity sample of the other folds (~32M pairs, 600 rounds, 127
leaves, learning rate 0.05, fixed seed, deterministic mode). Final models use the same
recipe on all folds.

**Threshold selection method:** grid search of the decision layer on out-of-fold
probabilities with the exact macro-F0.5 metric (blocking misses count against recall):
global threshold 0.20–0.90 with and without exclusivity, and expected-F0.5 set selection
(per S1, choose the prefix of candidates that maximises the plug-in expected F0.5; the
empty set is scored by P(no true match) = Π(1−p), i.e. an explicit abstain option).
Selected: **expected-F0.5 set selection (floor 0.02, abstain weight 1.25) + exclusivity**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9820** — out-of-fold on all 2,206,821 training S1
  (singletons 0.9833, non-singletons 0.9819; US 0.9845, India
  0.9783; micro precision 0.9976, micro recall 0.9540).

| matcher (full training set, out-of-fold) | decision | macro F0.5 |
|---|---|---|
| rule-based baseline (hand-weighted similarity) | global threshold 0.850 + exclusivity | 0.8525 |
| LightGBM stage 1 (pair features) | global threshold 0.725 + exclusivity | 0.9771 |
| **LightGBM stage 2 (+ candidate-graph features)** | expected-F0.5 set selection (floor 0.02, abstain weight 1.25) + exclusivity | **0.9820** |
| oracle: perfect matcher on the candidates | – | 0.9926 |

Feature-set ablation (held-out fold of 735k S1, identical fast protocol for every variant,
run on the feature build before word segmentation was added):
base 0.9695 → + support features 0.9716 → + stage-2 competition features 0.9759; the
production stage-1 probabilities alone score 0.9707 on the same entities, so the stage-2
gain (+0.0052) is not inherited from a stronger stage-1 model.

Third iteration (same held-out fold, trained on S1s of folds 1–2, 12/4-deep candidates;
every idea kept only if it beat the reference):

| experiment | fold-0 macro F0.5 | verdict |
|---|---|---|
| reference: previous stage-2 recipe (15 % S1 sample) | 0.97880 | – |
| + probability-weighted cluster support (11 features) | 0.97998 | adopted |
| + neighbour similarity (6 features) | 0.98014 | adopted |
| stage 3 (competition features recomputed from stage-2 probabilities) | 0.97999 | rejected: no gain over cluster support |
| 40 % S1 training sample instead of 15 % | stage 2: 0.97915; stage 1: 0.97586 vs 0.97521 | adopted |
| 255 leaves, 1,200 rounds | 0.97873 | rejected: overfits |
| decision on claimant-normalised odds (full out-of-fold set) | 0.97872 vs 0.97870 | rejected: no gain |

Model history on the full training set (out-of-fold): first submission 0.9709, second
0.9787, **this submission 0.9820**.

- **Error budget (out-of-fold, all 7,638,365 true training pairs):**
  166,686 (2.18 %) never became candidates; 184,312
  (2.41 %) were candidates the matcher rejected; 17,385 predicted pairs are
  wrong merges (2,344 of them on singleton S1s). Compared with the first full run,
  wrong merges fell from 0.78 % to 0.24 % of predicted pairs.

Macro F0.5 by slice:

| slice | S1 entities | macro F0.5 |
|---|---|---|
| India | 883,188 | 0.9783 |
| US | 1,323,633 | 0.9845 |
| singletons | 123,247 | 0.9833 |
| 1 true match | 119,157 | 0.9405 |
| 2-4 true matches | 1,390,168 | 0.9832 |
| 5+ true matches | 574,249 | 0.9874 |

Share of each candidate trait among all true candidate pairs vs among the errors:

| trait | all true pairs | matcher misses (FN) | wrong merges (FP) |
|---|---|---|---|
| empty_address | 4.1% | 70.9% | 21.8% |
| indic_name | 6.8% | 1.1% | 3.3% |
| domain_name | 4.7% | 0.5% | 1.2% |
| fka_alias | 2.0% | 0.0% | 0.0% |
| name_sim<60 | 6.4% | 8.6% | 15.6% |
| first_code_differs | 20.2% | 19.6% | 49.3% |

All examples below are errors of the final model (`results/error_analysis_final.md`).

- **Common false positives (wrong merges):**
  1. *Planted siblings* — the same name at a neighbouring house number on the same street:
     `Regional Seafood Inc., 2085 Aldersgate Road` vs `Regional Seafood Inc, 2090A Aldersgate
     Rd` (a singleton S1, so the error costs a full 1.0); `Cassaundra's Bakery, 1902 Porter
     Street` vs `… 1905 Porter St`; `Chiropractic Gulf Care Associates, 712 Taft Road` vs
     `… 716-B Taft Road`. Records whose first house code differs from the S1's are
     20.2 % of true pairs but 49.3 % of the remaining
     wrong merges. True matches carry exactly the same kind of offsets, so the residue is
     largely irreducible from name + address alone.
  2. *Same address, (near-)same name, labelled as a different business*: `Corner Hypnosis`
     vs `Corner Hypnosis LLC` (both at 16900 Salmonberry Road), `Empire Alliance III` vs
     `Empire Alliance FIII` (555 Pride Avenue), `Perry, Gavra, O.D., M.D., P.C.` vs the same
     name plus `Inc` (3 Chloe Court). Nothing in the two fields separates these.
  3. *Empty-address records with a common name* claimed by many S1s: `Spears 6olden 6olden
     Orion Inc` (no address) is a candidate of 105 S1s, `Ravika Private Mining Limited` of 77.
     In the first full run these were 45.5 % of all wrong merges; with the candidate-graph
     features they are 21.8 %.
- **Common false negatives (missed matches):**
  1. *Empty-address candidates* (4.1 % of true pairs, 70.9 %
     of matcher misses): `Motech Inc.` (no address) competes with 203 S1s, `Shiva Consulting
     Pvt Ltd.` with 332. With only a name, the expected-F0.5 rule correctly prefers to
     abstain — under F0.5 a wrong merge costs twice a miss.
  2. *Unrelated generated names linked only by address*: `Anchor Freight Studios PLLC` ↔
     `Brixsynfaye` (Route 1, Steuben), `Laferriere, Hale & Vargas Era LLC` ↔ `Tavodova`
     (13122 Vista Station Boulevard) — indistinguishable from a different business at the
     same address.
  3. *The mirror image of the siblings*: true matches whose house number disagrees
     (`South Frontier Cbre Inc, 15602 Giese Lane` ↔ `15221 Giese Lane`; `Hatti Byfield Value
     Martin LLC, 8085 Bell Campground Road` ↔ `7937 Bell Campground Road`).
  4. *Blocking misses* (2.18 % of true pairs), concentrated in India (recall
     0.9673 vs US 0.9854): generic names combined with truncated
     addresses leave too little shared evidence to rank in the top 20.


---

## 6. Conclusion

Blocking recall — not the classifier — was the binding constraint for most of the work:
diagnosing *why* true pairs were missed (generic names, empty addresses, concatenated
names), then deepening the candidate lists once the matcher had caught up, moved the
ceiling from 0.934 to 0.9926. On the matching side, the largest gains came from treating
each decision in the context of the whole candidate graph (support, competition, cluster
mass, neighbour similarity, exclusivity) rather than pair by pair, and from training on
more entities. Everything is measured on the
full training set with the exact challenge metric, and the pipeline reproduces the
submission from raw files with one command.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (all source in `src/ber/`):

| module | role |
|---|---|
| `__main__.py` | CLI: `python -m ber {all, train, test, preprocess, analyze, experiment-*}` |
| `translit.py`, `normalize.py` | Indic transliteration; name/address normalisation, word segmentation |
| `store.py`, `preprocess.py` | memory-mapped columnar record store; multi-process normalisation |
| `tokens.py`, `blocking.py` | typed hashed blocking tokens; key blocking (A) and IDF meta-blocking (B) |
| `features.py`, `stage2.py` | pair, context and support features; stage-2 competition, cluster-support and neighbour features |
| `rules.py`, `model.py`, `decide.py` | rule baseline; LightGBM; exclusivity, threshold, expected-F0.5 selection |
| `evaluate.py` | exact macro-F0.5 metric, blocking recall and oracle ceiling |
| `pipeline.py`, `cache.py` | orchestration; code-and-settings fingerprinted caches |
| `analysis.py`, `experiments.py` | error analysis; reproducible experiment grids and ablations |
| `tools/matcher_experiments.py` | the held-out matcher experiments of the third iteration (`EXPERIMENTS.md` 7.2) |

Entry points: `python reproduce.py --data-dir <dataset> --team <team>` (tests → pipeline →
official validator → zip), or `python -m ber all --data-dir <dataset>`; `make all` on
Linux/macOS; a `Dockerfile` pins the runtime (run with `--network none`). Unit and
synthetic end-to-end tests: `pytest tests` (38 tests). Measured runtime of the full
reproduction: about 8 hours end to end on an 8-core / 32 GB Windows laptop (Intel Core Ultra 7 258V, CPU only; peak memory 24 GB). Per stage — train: preprocessing + tokens ~11 min, blocking 65 min, pair features ~40 min, rule baseline 9 min, stage-1 cross-fitting 78 min, stage-2 features 16 min, stage-2 cross-fitting 53 min, decision tuning ~15 min, final models 29 min; test: preprocessing + tokens ~34 min, blocking 54 min, pair features ~34 min, prediction + stage-2 features + writing 44 min. Every stage is cached by a content fingerprint, so an interrupted run resumes where it stopped.

### B. Additional Results

Test-set prediction profile compared with the training out-of-fold profile (no labels on
test; a sanity check that behaviour transfers, including to France, unseen in training):

| split | country | S1 entities | predicted no-match | matches / S1 | candidates / S1 |
|---|---|---|---|---|---|
| train (out-of-fold) | US | 1,323,633 | 5.80 % | 3.328 | 54.6 |
| train (out-of-fold) | India | 883,188 | 6.03 % | 3.283 | 53.4 |
| test | US | 663,106 | 5.29 % | 3.670 | 54.3 |
| test | France | 259,452 | 4.65 % | 3.601 | 54.0 |
| test | India | 809,986 | 5.59 % | 3.446 | 53.0 |

Full logs: `EXPERIMENTS.md` (every decision and its measurement),
`results/*.jsonl` (raw blocking and decision-tuning grids), `results/error_analysis*.md`.
