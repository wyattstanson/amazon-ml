# Experiment log

Every decision in the pipeline traces back to a measured number here. Raw per-run rows
are appended by the code to `work/experiments_blocking.jsonl` and
`work/decision_tuning.jsonl`; this file is the human-readable summary, in the order
the work was actually done.

Metric definitions (all on the labelled training set):
* **pair recall** – fraction of true (S1, S2/S3) pairs present in the candidate set.
* **oracle F0.5** – macro F0.5 of a *perfect* matcher restricted to the candidates
  (predicts exactly truth ∩ candidates). The hard ceiling blocking puts on the score.
* **macro F0.5** – the challenge metric (per-S1 F0.5, singletons 1/0, averaged).

## 0. EDA facts that shaped the design

| fact | number | consequence |
|---|---|---|
| train S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 | laptop-scale engineering: memory-mapped columnar store, vectorised everything |
| test S1 / S2 / S3 rows | 1,732,544 / 4,887,273 / 5,082,316 | same |
| singleton S1 (no match) | 5.6 % | recall matters for 94 % of entities, not just abstention |
| matches per S1 | mode 3, max 11 (≤5 from S2, ≤6 from S3) | many-to-one; top-k per source must be ≥ 6 |
| S2/S3 records owned by >1 S1 | **0** of 7,638,365 | exclusivity constraint usable at decision time |
| S2/S3 records matching nothing | 2,681,854 (27 %) | large distractor pool; ~30 % share a house-number+street word with some S1 ("sibling" hard negatives) |
| cross-country matches | **0** | blocking partitioned by country (open set of labels) |
| India S2 / S3 names in Indic scripts | 23 % / 12 %, all 9 major scripts | built-in transliterator (shared Brahmic block layout) |
| native-script address text | only 16 distinct values (state names) | exact alias table |
| test countries | US 38 %, India 47 %, **France 15 % (unseen)** | no country feature in the model; French conventions only in normalisation |
| test S2+S3 per S1 | 5.75 (train: 4.68) | test has more candidates per entity → precision pressure; noted as a risk |

## 1. Blocking

### 1.1 First meta-blocking run — recall far too low (diagnosed, not tuned around)
Strategy B with unigram tokens only (name, phonetic, prefix, address word/code,
street bigram), `max_df=2000`, `k=12`, 20k-S1 sample:
**pair recall 0.856, oracle F0.5 0.933, 23.5 pairs/S1.**

Diagnostic over 1,430 missed India pairs (`scratch/diag_blocking.py`):
* 100 % of misses *did* share tokens with their S1;
* 70 % ranked outside the top-k, 30 % had every shared token removed by the df cap.

Cause: generic Indian names (`shivam builders`, `raj construction`: name words in
10–20k records) and city words (`kolkata` 80k, `howrah` 166k). The decisive evidence is
the *combination* (name near city), which unigrams cannot express once capped.
Fix: cross-field conjunction tokens `phonetic-name-key × address-word` and
`address-code × address-word`.

Also found by profiling: 153 of 166 s per country were spent in random-order
`np.searchsorted` over a multi-million vocabulary (cache-miss bound); replaced by
`np.unique(return_inverse)` + sorted-key lookups.

### 1.2 Conjunction tokens + engine rewrite (50k-S1 sample unless noted)

| strategy | tokens | caps | k | pair recall (US / IN) | oracle F0.5 | pairs/S1 | time |
|---|---|---|---|---|---|---|---|
| A: key blocking | street bigram + 2 name keys | 500 | all pairs | 0.872 (0.914 / 0.809) | 0.946 | 65.0 | 55 s |
| A: key blocking | same | 2000 | all pairs | 0.885 (0.932 / 0.815) | 0.953 | 142.6 | 59 s |
| B: meta-blocking | unigrams only | 2000 | 12 | 0.858 (0.857 / 0.859) | 0.934 | 23.5 | 196 s |
| B: meta-blocking | + `pa`,`ca` conjunctions | 2000 | 12 | **0.932** (0.943 / 0.914) | **0.977** | 24.0 | 2336 s → 416 s* |
| B (20k sample) | + conjunctions | unigrams 300, conj 2000 | 12 | 0.924 (0.937 / 0.905) | 0.973 | 24.0 | 404 s |

\*same configuration after the engine rewrite: per-row partial selection for top-k
instead of a global lexsort of every score, flops-bounded query chunks, direct CSR
construction. Tightening unigram caps to 300 lowered recall and saved no time (cost is
dominated by one-time vocabulary setup), so caps stay at 2000.

Remaining India misses (diagnostic, 839 of 10,323 pairs) fell into three patterns, each
addressed by a new token type or channel:
1. candidate has an **empty address** → can only score on its name and loses to any
   address-sharing record → separate top-`k_empty` channel among empty-address records;
2. **generic words, distinctive pair** (`swastik it`, `baba brothers`) → name bigrams (`nb`)
   and phonetic bigrams (`pb`, typo-robust: `smart pavar` = `smart power`);
3. **concatenated names** (`elementalbrothers`) → whole compact core name token.

### 1.3 Bigram tokens + empty-address channel (20k-S1 sample)

| config | pair recall (US / IN) | entity-full recall | oracle F0.5 | pairs/S1 |
|---|---|---|---|---|
| previous best (conjunctions, k=12, cap 2000) | 0.930 (0.944 / 0.910) | – | 0.976 | 24.0 |
| + `nb`,`pb` bigrams + empty-address channel (k=12, k_empty=4) | 0.9631 (0.972 / 0.950) | 0.889 | 0.9879 | 31.3 |
| same, deeper k=16, k_empty=6 | 0.9669 (0.975 / 0.954) | 0.900 | 0.9891 | 42.4 |
| same as row 2, cap 5000 | 0.9647 (0.972 / 0.954) | 0.893 | 0.9887 | 31.6 |

Decision: k=12, k_empty=4, cap 5000. k=16 buys +0.0012 oracle F0.5 for +35 % pairs
(which scales feature time, memory and the test candidate file); cap 5000 buys +0.0008
for +1 % pairs.

### 1.4 Final blocking, full training set (all 2,206,821 S1)

| | pair recall | oracle F0.5 | pairs/S1 |
|---|---|---|---|
| all | **0.9650** | **0.9884** | 31.6 (69.7M pairs) |
| US | 0.9720 | 0.9913 | 31.7 |
| India | 0.9545 | 0.9840 | 31.4 |

Entity-level full recall (every true match of a non-singleton found): 0.892.
Wall time 37.5 min (US 20 min, India 16 min) on an 8-core laptop.
Reduction ratio vs. the full cross product: 1 − 69.7M / (2.2M × 10.3M) = 99.9997 %.
The 20k-sample estimate (0.9647 / 0.9887) matched the full-set number, validating the
sampled experiment protocol.

## 2. Matching — first end-to-end check (smoke sample)
3,000 S1 → 72k candidate pairs; LightGBM trained on 1,500 S1, evaluated on the other 1,500
(full ground truth, so blocking misses count against the score). Oracle on these
candidates: 0.976.

| matcher | best decision | macro F0.5 |
|---|---|---|
| rule-based (hand weights) | threshold 0.85 | 0.840 |
| LightGBM (51 features) | threshold 0.55 | **0.954** |

Exclusivity cannot be judged on a small sample (competing S1s are mostly absent); it is
evaluated on the full cross-fitted training set below.

## 3. Matching at full scale (all 2.2M training S1, S1-grouped 3-fold cross-fitting)

Every S1's candidates are scored by a model that never saw that S1 (folds by S1 entity);
the decision layer is tuned on these out-of-fold scores with the exact challenge metric,
counting blocking misses against recall.

| matcher | decision | macro F0.5 | singletons | non-singletons | micro P | micro R |
|---|---|---|---|---|---|---|
| rules (hand weights) | exclusive, t=0.85 | 0.8464 | 0.929 | 0.841 | 0.978 | 0.717 |
| LightGBM, 51 features (15 % S1 sample per fold, 600 rounds) | exclusive, t=0.70 | **0.9709** | 0.967 | 0.971 | 0.992 | 0.933 |
| oracle (perfect matcher on these candidates) | – | 0.9884 | 1.000 | – | 1.000 | 0.965 |

LightGBM by country: US 0.9732, India 0.9674.

Decision-layer comparison on the same LightGBM probabilities:

| decision | best setting | macro F0.5 |
|---|---|---|
| threshold, no exclusivity | t = 0.725 | 0.9707 |
| threshold + exclusivity | t = 0.70 | **0.9709** |
| expected-F0.5 set selection + exclusivity | floor 0.05, boost 1.5 | 0.9708 |
| threshold + exclusivity at the default 0.5 (untuned) | t = 0.5 | 0.9696 |

* Tuning the threshold is worth +0.0013 over 0.5; the optimum is flat over 0.65–0.75.
* Exclusivity helps most at loose thresholds (+0.008 at t = 0.2) and slightly at the optimum.
* Expected-F0.5 selection ties the tuned threshold (it trades singleton accuracy for
  precision); the simpler threshold rule is kept. On the rule-based *scores* it fails
  (0.749) because they are not calibrated probabilities.

## 4. Error analysis (out-of-fold, `python -m ber analyze` → `work/error_analysis.md`)

* True pairs 7,638,365: **3.50 % missed by blocking**, **3.16 % rejected by the matcher**;
  wrong merges are 0.78 % of predicted pairs (4,291 on singleton S1s).
* Hardest slice: S1s with exactly one true match (F0.5 0.922).
* Trait profile — share among all true candidate pairs vs among errors:

| trait | all true pairs | matcher misses | wrong merges |
|---|---|---|---|
| candidate address empty | 4.0 % | **56.6 %** | **45.5 %** |
| first house/unit code differs from S1's | 20.2 % | 34.7 % | 36.5 % |
| Indic-script name | 6.8 % | 1.8 % | 2.4 % |
| domain-style name | 4.1 % | 0.6 % | 0.5 % |
| fka alias | 2.0 % | 0.0 % | 0.0 % |

Reviewed examples (see `work/error_analysis.md`) fall into three patterns:
1. **Planted siblings**: identical name, neighbouring number on the same street
   (`2068`/`2070 Northampton St`, `704`/`709 Nash St`, `407`/`409 Dharma Block`). True
   matches show the same kind of offsets (`12425`/`12427 Brightwater Ln`), so the pair
   alone is ambiguous — but a sibling is its own entity whose several records agree on
   *its* number, while a typo is a one-off. → "support" features (how many other
   candidates of the S1 share this candidate's number / name).
2. **Empty-address records with a (near-)identical name** claimed by many S1s
   (`r_ncand` 22–117): over-accepted for the wrong S1, under-accepted (p ≈ 0.02–0.08) for
   the right one. → stage-2 competition features (is this S1 the clear best claimant?).
3. Heavy corruption / partial transliteration (`Efiasbg,fa Hunt`, `White पावर`): largely
   irreducible.

## 5. Second iteration, driven by the error analysis

### 5.1 First complete submission (baseline, kept in `submissions/v1_baseline/`)
The pipeline of sections 1–3 produced a full test submission: 54,399,986 candidate pairs
(31.4 per S1), 5,734,918 matches; the official validator passes in both default and
`--check-ids` mode. Test prediction profile per country matched the training out-of-fold
profile (predicted-empty rate 5.7–5.9 % US/India vs 5.9–6.1 % train; France 4.9 %).

### 5.2 Blocking misses at full scale → word segmentation + zero-run normalisation
Profile of the 267,519 true pairs blocking missed (3.5 %): concatenated / domain-style names
are 4.7 % of true pairs but 23 % of misses (17 % miss rate), e.g. `searsplatforms` vs S1
`camellia sears platforms`; codes with inner zeros (`l053` vs `l53`) also broke exact code
tokens. Fixes: a unigram word segmenter whose vocabulary is the same split's Source-1 name
words (applied only to single-token names ≥ 8 letters that are not themselves a known
word), and stripping leading zeros of every digit run inside codes.

| 20k-S1 sample, cap 5000, k=12 | pair recall (US / IN) | entity-full | oracle F0.5 | pairs/S1 |
|---|---|---|---|---|
| before | 0.9647 (0.9722 / 0.9537) | 0.893 | 0.9887 | 31.6 |
| **+ segmentation + zero runs** | **0.9726** (0.9814 / 0.9595) | 0.917 | **0.9910** | 31.5 |
| same, k_empty = 6 | 0.9735 (0.9823 / 0.9603) | 0.920 | 0.9913 | 34.9 |

Adopted segmentation + zero runs; k_empty stays 4 (+0.0003 oracle is not worth +11 % pairs).

### 5.3 Matcher features: support and stage-2 competition (ablation)
Fast like-for-like protocol: fit on a 5 % S1 sample of folds 1–2 (300 rounds), evaluate on
every S1 of fold 0 (735k entities, 23M pairs), full decision grid. (Run on the pre-
segmentation feature build; stage-2 inputs are the production out-of-fold stage-1 scores.)

| variant | macro F0.5 | singletons | micro P | micro R | best decision |
|---|---|---|---|---|---|
| base 51 features | 0.9695 | 0.962 | 0.9913 | 0.9313 | threshold 0.700 |
| + support features (3) | 0.9716 | 0.960 | 0.9941 | 0.9354 | threshold 0.725 |
| + support + stage-2 competition features (10) | **0.9759** | **0.978** | **0.9962** | **0.9405** | expected-F0.5, floor 0.05 |
| check: production stage-1 probabilities alone, same fold | 0.9707 | 0.969 | 0.9925 | 0.9317 | threshold 0.725 |

The check row rules out that stage 2 merely inherits the stronger production stage-1
model: stage 2 beats the best stage-1 scores on the same entities by +0.0052. Both
additions were adopted. With stage-2 probabilities the expected-F0.5 set selection (with an
explicit P(no match) abstain option) becomes the best decision rule.

### 5.4 Engineering fixes made along the way
* A feature-ablation run thrashed (16 GB RSS, 9.5 GB swap) by holding two 70M-row extra
  feature blocks in RAM next to a 14 GB memmap; redesigned to write blocks to disk and
  re-open memmaps per chunk (peak 2.9 GB).
* Output writer rewritten to stream by S1 chunk (the old one decoded all 54M candidate ids at
  once, ~4 GB); verified byte-identical to the v1 submission.
* Caches are now fingerprinted by the source code and settings that produced them, so a
  normalisation change can never silently reuse stale records.
* Competition features rewritten from four sorts + `ufunc.at` to two packed-key stable sorts
  (22 min → ~3 min at full scale), verified against a brute-force test.

## 6. Final pipeline, full training set (clean run from raw files)

All caches deleted, then `python -m ber all` (the run was interrupted once by a laptop
restart after the feature stage; it resumed from the fingerprinted caches, and the
recomputed rule-baseline score was identical to 16 digits — 0.8518712262399537 — confirming
determinism across the restart).

| stage | result |
|---|---|
| blocking pair recall (US / India) | **0.9728** (0.9813 / 0.9602), entity-full 0.916, 31.5 pairs/S1, 69,620,461 pairs |
| blocking oracle F0.5 (US / India) | **0.9909** (0.9942 / 0.9858) |
| rule baseline | 0.8519 (threshold 0.85 + exclusivity) |
| LightGBM stage 1 (54 pair features) | 0.9754 (threshold 0.725 + exclusivity; P 0.9949, R 0.9444) |
| **LightGBM stage 2 (+ 10 competition features)** | **0.9787** (expected-F0.5 selection, floor 0.05, abstain weight 1.5, + exclusivity; singletons 0.9803, P 0.9966, R 0.9478) |

Stage 2 reaches 98.8 % of the blocking ceiling. Versus the first full run (0.9709):
blocking misses 3.50 % → 2.72 % of true pairs, matcher misses 3.16 % → 2.50 %, wrong merges
0.78 % → 0.34 % of predicted pairs. Final error analysis: `results/error_analysis_final.md`.

## 7. Third iteration: pushing towards the ceiling

### 7.1 Where the remaining F0.5 is lost (out-of-fold, all 2.2M training S1)
Counterfactual scores obtained by editing the production decisions (`scratch`-style analysis,
numbers exact):

| counterfactual | macro F0.5 | gain |
|---|---|---|
| production (stage 2) | 0.97870 | – |
| remove every wrong merge | 0.98212 | +0.0034 |
| recover every in-candidate miss | 0.98746 | +0.0088 |
| … only misses on empty-address records (122k pairs) | 0.98392 | +0.0052 |
| … only misses on records with an address (69k pairs) | 0.98235 | +0.0037 |
| perfect matcher on the candidates (oracle) | 0.99088 | +0.0122 |

Empty-address misses are the largest bucket but mostly irreducible: for 64 % of them the
record is claimed by ≥ 10 S1s with near-identical names, and the true owner is the top
claimant in only 31 % of those misses. Checked and ruled out: record-id order carries no
information about matches (Spearman ρ between S1 id number and matched id number −0.002).

### 7.2 Fast held-out protocol
Train on a seeded 15 % S1 sample of folds 1–2, evaluate on every S1 of fold 0 (735,513
entities, 23.2M pairs) with the exact macro F0.5 after a decision grid; stage-2 inputs are the
production out-of-fold stage-1 probabilities (`scratch/exp.py`, results in
`results/experiments_matcher_v3.jsonl`).

| variant | fold-0 macro F0.5 | Δ |
|---|---|---|
| E0 production stage-2 recipe | 0.97880 | – |
| **E2 + probability-weighted cluster support (11 features)** | **0.97998** | **+0.0012** |

The new block weights the support counts by stage-1 probability (`stage2.mass_features`):
mass of the S1's other candidates sharing the record's house code / name / address, the mass
of the strongest *rival* code cluster (a planted sibling entity), best probability among the
S1's other-source candidates with the same code, per-source rank and mass, the S1's mass on
empty vs non-empty candidates, and how many S1s claim the record with p > 0.1. Five of them
rank among the model's top 15 features by gain. Adopted.

Decision layer, on the full production OOF probabilities: normalising each record's
claimants' odds (a record belongs to at most one S1) before the expected-F0.5 rule gives
0.97872 vs 0.97870 — no gain; stage 2 already models the competition. Not adopted.

### 7.3 Blocking depth re-measured after segmentation (20k-S1 sample, full pool)

| k / k_empty | pair recall (US / India) | oracle F0.5 | pairs/S1 |
|---|---|---|---|
| 12 / 4 (previous) | 0.9726 (0.9814 / 0.9595) | 0.9910 | 31.5 |
| 16 / 6 | 0.9761 (0.9841 / 0.9643) | 0.9922 | 42.9 |
| **20 / 8** | **0.9782** (0.9857 / 0.9672) | **0.9930** | 54.1 |

With the matcher now close to its ceiling, the ceiling itself is worth raising: adopted
k = 20, k_empty = 8 (+0.0020 oracle for +72 % pairs; the feature and model stages scale
linearly and fit the same laptop).

### 7.4 Final pipeline with the third-iteration changes (full training set)
Changes: blocking k = 20 / k_empty = 8; stage-2 block of 27 features (10 competition + 11
cluster-support + 6 neighbour); every fold / final model fit on a 40 % S1 sample (~32M
pairs); decision grid extended to floors {0.02 … 0.20} and abstain weights {0.8 … 2.5}.
Blocking and pair features were precomputed through the pipeline's own stage functions
(same cache stamps), then `python -m ber train` / `test` ran from those caches.

| stage | previous (section 6) | now |
|---|---|---|
| blocking pair recall (US / India) | 0.9728 (0.9813 / 0.9602) | **0.9782** (0.9854 / 0.9673) |
| entity-full recall | 0.916 | **0.932** |
| blocking oracle F0.5 (US / India) | 0.9909 (0.9942 / 0.9858) | **0.9926** (0.9955 / 0.9884) |
| candidate pairs (train) | 69.6M (31.5/S1) | 119.5M (54.1/S1) |
| rule baseline | 0.8519 | 0.8525 |
| LightGBM stage 1 | 0.9754 | **0.9771** |
| **LightGBM stage 2** | 0.9787 | **0.9820** (US 0.9845 / India 0.9783) |
| stage-2 micro precision / recall | 0.9966 / 0.9478 | **0.9976 / 0.9540** |
| decision | expected-F0.5, floor 0.05, abstain 1.5, exclusive | expected-F0.5, floor 0.02, abstain 1.25, exclusive |

Stage 2 now reaches 98.9 % of the (higher) blocking ceiling, with both fewer wrong merges
and fewer misses than before. Training wall time on the laptop: stage-1 folds 23–31 min,
stage-2 features 16 min, stage-2 folds 18 min, final models 2 × 13–18 min; peak RSS of the
training process 24 GB.
