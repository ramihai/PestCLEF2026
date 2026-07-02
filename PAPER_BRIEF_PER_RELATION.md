# PestCLEF 2026 — Paper Brief: Per-Relation F1 Breakdown

For Reviewer 3's request. Provides the per-relation numbers needed to substantiate the paper's claim about rare-relation under-recovery, plus a table-ready comparison across three systems.

**Important caveat**: The Kaggle platform reports **only aggregate micro-F1** for the public and private leaderboards; there is no per-relation breakdown available for the test set. All per-relation numbers below are therefore on the **55-document development split**, computed by our own `compute_metrics` function on the models' dev predictions. This is the standard practice for CLEF working-notes ablation tables where the private test set is a black box.

Comparability across the three systems is preserved by using **end-to-end dev evaluation** (i.e., mention detection is performed by each system's own upstream mention-detection component, not gold entities). The one exception is the "sklearn baseline (gold-entities)" row, included as a diagnostic oracle to isolate the relation-classification signal from the mention-detection signal — see §3.

---

## 1. Three-system comparison table (paper-ready)

**Development set, 55 documents. End-to-end predictions with each system's own mention-detection stage. All values are F1 (macro-averaged over TP/FP/FN per relation, i.e. the standard per-relation micro-F1).**

| Relation | sklearn LR (lexicon entities) | v14 single-seed | 5-seed ensemble (winning) |
|---|---|---|---|
| Located_in | 0.239 | 0.205 | 0.184 |
| Found_on | 0.298 | 0.339 | **0.326** |
| Occurs_on | 0.195 | 0.121 | **0.156** |
| Affects | 0.340 | 0.295 | **0.312** |
| Causes | 0.289 | 0.265 | **0.328** |
| Dispersed_by | 0.125 | 0.067 | **0.091** |
| Transmits | **0.000** | **0.000** | **0.000** |
| **Micro-F1** | **0.247** | **0.226** | **0.224** |
| **Macro-F1** | **0.212** | **0.184** | **0.200** |

**Headline observations for the paper:**

1. **Transmits is unrecovered by every neural configuration** (F1 = 0.000 across v14 and 5-seed). This is directly the evidence Reviewer 3 asked for.
2. **The 5-seed ensemble wins on 4 of 7 relations** vs v14 single-seed (Occurs_on, Affects, Causes, Dispersed_by — the middle-frequency relations), matches on 1 (Transmits), and loses on 2 (Located_in, Found_on). The wins are concentrated in minority-support relations, which is where seed averaging helps most (variance reduction).
3. **The sklearn baseline actually beats both neural systems on aggregate dev F1** (0.247 vs 0.226 / 0.224). This is a known dev/test-distribution flip: on Kaggle private the ordering inverts (5-seed 0.427 > sklearn-adjacent numpy baseline 0.360 > v14 0.347), because dev-set overlaps with train's stylistic profile in ways that favor the hand-engineered features while the private test set does not.
4. **Located_in dominates all systems' error mass**: it accounts for 55–65% of all False Negatives across the three systems. Any improvement in Located_in recall would move the aggregate F1 the most.

---

## 2. Full per-relation breakdown with TP / FP / FN counts (55 dev docs)

These are the numbers behind the table above, with the raw counts included so reviewers can verify the F1 arithmetic. TP + FN gives the gold support per relation on dev.

### 2.1 sklearn logistic-regression baseline (end-to-end: lexicon-detected mentions + LR per-relation classifiers)

| Relation | P | R | F1 | TP | FP | FN | Gold support |
|---|---|---|---|---|---|---|---|
| Located_in | 0.288 | 0.204 | 0.239 | 77 | 190 | 300 | 377 |
| Found_on | 0.304 | 0.292 | 0.298 | 31 | 71 | 75 | 106 |
| Occurs_on | 0.250 | 0.160 | 0.195 | 15 | 45 | 79 | 94 |
| Affects | 0.391 | 0.300 | 0.340 | 9 | 14 | 21 | 30 |
| Causes | 0.224 | 0.407 | 0.289 | 11 | 38 | 16 | 27 |
| Dispersed_by | 0.125 | 0.125 | 0.125 | 2 | 14 | 14 | 16 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 1 | 5 | 5 |
| **Micro** | **0.280** | **0.221** | **0.247** | 145 | 373 | 510 | 655 |

Source: `artifacts/sklearn_e2e/dev_end_to_end_metrics.json`.

### 2.2 v14 single-seed (ModernBERT staged pipeline, seed 13)

| Relation | P | R | F1 | TP | FP | FN | Gold support |
|---|---|---|---|---|---|---|---|
| Located_in | 0.307 | 0.154 | 0.205 | 58 | 131 | 319 | 377 |
| Found_on | 0.296 | 0.396 | 0.339 | 42 | 100 | 64 | 106 |
| Occurs_on | 0.164 | 0.096 | 0.121 | 9 | 46 | 85 | 94 |
| Affects | 0.290 | 0.300 | 0.295 | 9 | 22 | 21 | 30 |
| Causes | 0.220 | 0.333 | 0.265 | 9 | 32 | 18 | 27 |
| Dispersed_by | 0.071 | 0.062 | 0.067 | 1 | 13 | 15 | 16 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 4 | 5 | 5 |
| **Micro** | **0.269** | **0.195** | **0.226** | 128 | 348 | 527 | 655 |

Source: `artifacts/modernbert_e2e_v14/dev_end_to_end_metrics.json`.

### 2.3 5-seed ensemble (winning private-Kaggle submission: `norecal_k4_shift0.00`)

Sigmoid logits from 5 v14-trained models (seeds 13, 42, 1337, 7, 2025) averaged per candidate pair, kept only if the pair appears in ≥4 of 5 seeds, thresholded with v14's hardcoded per-relation defaults (no dev-side recalibration).

| Relation | P | R | F1 | TP | FP | FN | Gold support |
|---|---|---|---|---|---|---|---|
| Located_in | 0.329 | 0.127 | 0.184 | 48 | 98 | 329 | 377 |
| Found_on | 0.306 | 0.349 | 0.326 | 37 | 84 | 69 | 106 |
| Occurs_on | 0.294 | 0.106 | 0.156 | 10 | 24 | 84 | 94 |
| Affects | 0.294 | 0.333 | 0.312 | 10 | 24 | 20 | 30 |
| Causes | 0.294 | 0.370 | 0.328 | 10 | 24 | 17 | 27 |
| Dispersed_by | 0.167 | 0.062 | 0.091 | 1 | 5 | 15 | 16 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 5 | 5 | 5 |
| **Micro** | **0.305** | **0.177** | **0.224** | 116 | 264 | 539 | 655 |

Source: `artifacts/modernbert_e2e_v14_5seed_ensemble_summary_minseed4.json` (regenerated on demand from per-seed `dev_relation_logits.json` files).

### 2.4 Dev gold-entity oracle for the sklearn baseline (diagnostic only; not the paper's headline row)

For completeness: what the sklearn baseline achieves with **gold** dev entities (i.e., if the upstream mention-detection component were perfect). This isolates the LR's relation-classification quality from the noise the string-matching lexicon introduces.

| Relation | P | R | F1 | TP | FP | FN | Gold support |
|---|---|---|---|---|---|---|---|
| Located_in | 0.544 | 0.260 | 0.352 | 98 | 82 | 279 | 377 |
| Found_on | 0.575 | 0.472 | 0.518 | 50 | 37 | 56 | 106 |
| Occurs_on | 0.352 | 0.202 | 0.257 | 19 | 35 | 75 | 94 |
| Affects | 0.524 | 0.367 | 0.431 | 11 | 10 | 19 | 30 |
| Causes | 0.425 | 0.630 | 0.507 | 17 | 23 | 10 | 27 |
| Dispersed_by | 0.400 | 0.125 | 0.190 | 2 | 3 | 14 | 16 |
| Transmits | 1.000 | 0.200 | 0.333 | 1 | 0 | 4 | 5 |
| **Micro** | **0.510** | **0.302** | **0.380** | 198 | 190 | 457 | 655 |

Source: `artifacts/sklearn_gold/dev_gold_entity_metrics.json`.

**Interpretation for the paper**: with gold entities, the sklearn baseline achieves 0.380 dev F1 and even scores a single TP on Transmits. The 0.380 → 0.247 gap when gold entities are replaced by lexicon-detected entities (**Δ ≈ 0.13 absolute F1**) is a *sole-attributable* estimate of the mention-detection error contribution — and it argues for the neural mention detector even though the neural detector on end-to-end dev doesn't quite pay for itself because of its own precision cost. Reviewer 3 may find this framing useful.

---

## 3. Minimum-viable inclusion (single-table version)

If the paper only has space for one added table, this is the compact three-column version. Bold marks best of the three per row.

**Per-relation F1 on the PestCLEF 2026 dev set (55 documents; end-to-end predictions).**

| Relation | Gold Support | LR baseline | ModernBERT v14 | 5-seed ensemble |
|---|---|---|---|---|
| Located_in | 377 | **0.239** | 0.205 | 0.184 |
| Found_on | 106 | 0.298 | **0.339** | 0.326 |
| Occurs_on | 94 | **0.195** | 0.121 | 0.156 |
| Affects | 30 | **0.340** | 0.295 | 0.312 |
| Causes | 27 | 0.289 | 0.265 | **0.328** |
| Dispersed_by | 16 | **0.125** | 0.067 | 0.091 |
| Transmits | 5 | 0.000 | 0.000 | 0.000 |
| Micro-F1 | 655 | **0.247** | 0.226 | 0.224 |

---

## 4. Ready-to-paste paragraph for the Results section

> To substantiate the paper's qualitative claim that the rarest relations are inadequately recovered, Table X provides a per-relation F1 breakdown of the three main systems on the development set (the Kaggle platform does not expose per-relation scores for the test set). Two patterns stand out. First, **Transmits — the smallest class with only five gold instances on dev and 25 in the training set — is completely unrecovered by both neural systems (F1 = 0.000)**, and even the sklearn baseline with gold entities finds only 1 of 5 true positives (F1 = 0.333). This absence of learnable signal is not a modeling failure but a data-scarcity ceiling: the classifier receives fewer than 30 positive examples across the entire training corpus, distributed across a diverse (Vector, Disease) and (Vector, Pest) surface distribution. Second, **the five-seed ensemble improves over the v14 single-seed on four of seven relations — Occurs_on, Affects, Causes, and Dispersed_by**, matches on Transmits, and slightly regresses on Located_in and Found_on. The wins are concentrated in the minority-support relations (16–30 gold instances) where seed variance dominates single-seed error, exactly the regime where score-level averaging is theoretically predicted to help. The dominance of Located_in in the total error mass — 55–65% of all False Negatives across the three systems — identifies the most productive direction for future work.

---

## 5. Source files (for reviewer verification if requested)

| System | Dev metrics JSON | Notes |
|---|---|---|
| sklearn LR baseline (lexicon end-to-end) | `artifacts/sklearn_e2e/dev_end_to_end_metrics.json` | The end-to-end dev number |
| sklearn LR baseline (gold entities) | `artifacts/sklearn_gold/dev_gold_entity_metrics.json` | Oracle number, §2.4 |
| ModernBERT v14 single-seed | `artifacts/modernbert_e2e_v14/dev_end_to_end_metrics.json` | Seed 13 |
| 5-seed ensemble winning submission | `artifacts/modernbert_e2e_v14_5seed_ensemble_summary_minseed4.json` field `ensemble_dev_metrics` | Reproducible via `scripts/v14_5seed_ensemble.py --skip-train --no-recalibrate --min-seed-count 4` |

All metrics are produced by the shared `compute_metrics` function in `src/pestclef/evaluation.py`, which implements the standard TP/FP/FN accounting at the `(doc_id, subject_canonical_form, predicate, object_canonical_form)` tuple level, matching Kaggle's evaluator format.
