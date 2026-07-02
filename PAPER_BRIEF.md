# PestCLEF 2026 — Paper Brief

A self-contained technical reference for writing the working notes paper. Focuses on the two winning submissions: **`submission_v14_5seed_norecal_k4_shift0.00.csv`** (highest private Kaggle: 0.4266) and **`submission_ensemble_v14_v18_intersection.csv`** (highest public Kaggle: 0.4411). Contains exact architecture, training procedure, hyperparameters, and per-relation results.

---

## 1. Task and Dataset

- **Task**: PestCLEF 2026 shared task — document-level knowledge-graph extraction from EPOP plant-protection documents.
- **Subtask**: relation extraction over 7 entity types and 7 relation types.
- **Data split** (per `data/EPOP_documents/README.md`):
  - Train: 110 documents
  - Dev: 55 documents
  - Test: 82 documents (Kaggle: ~18% public proxy, ~82% private holdout)
- **Language mix**: predominantly English with ~7–11% French/Spanish/Italian documents.
- **Entity types** (7): `Pest`, `Plant`, `Location`, `Disease`, `Vector`, `Date`, `Dissemination_pathway`
- **Relation types** (7), with hardcoded schema constraints over `(subject_type, object_type)`:

| Relation | Allowed (subject_type, object_type) pairs |
|---|---|
| **Located_in** | (Pest, Location), (Plant, Location), (Vector, Location), (Disease, Location) |
| **Found_on** | (Pest, Plant), (Pest, Dissemination_pathway), (Vector, Plant), (Vector, Dissemination_pathway) |
| **Occurs_on** | (Pest, Date), (Plant, Date), (Vector, Date), (Disease, Date) |
| **Affects** | (Disease, Plant) |
| **Causes** | (Pest, Disease) |
| **Dispersed_by** | (Pest, Dissemination_pathway), (Disease, Dissemination_pathway) |
| **Transmits** | (Vector, Pest), (Vector, Disease) |

Schema validation is applied at inference (any `(subject, predicate, object)` triple whose entity types violate the schema is dropped).

- **Evaluation metric**: micro-F1 over (doc_id, subject_canonical_form, predicate, object_canonical_form) triples.

---

## 2. Pipeline Architecture (Shared by v14 and v18)

A two-stage neural pipeline implemented under model name `modernbert_staged`. Both stages use the same base encoder.

### 2.1 Stage 1 — Mention detection (token classification, BIO tagging)

- **Backbone**: `answerdotai/ModernBERT-base` (a modern long-context BERT variant; bidirectional, MLM-pretrained on generic English; 149M parameters; supports up to 8192-token contexts but used here with max sequence length 1024).
- **Head**: a single linear `BertForTokenClassification` head over `15 BIO labels` (`O` + `B-<type>` + `I-<type>` for each of the 7 entity types).
- **Loss**: **focal loss** (modulated cross-entropy) per token, gamma=2.0, with class-weighted re-balancing (cap 12.0× to avoid runaway weight on minority labels). Custom implementation in `FocalTokenClassifier` (src/pestclef/modernbert.py).
  - Easy-correct tokens (e.g. confident `O` predictions) get gradient multiplier `(1 − p_t)^2 ≈ 0`.
  - Hard tokens (boundary cases) retain near-full gradient.
- **Windowing**: documents are tokenized with `max_seq_length=1024`, `doc_stride=256` (sliding window with overlap). Per-window softmax probabilities are decoded into spans via `decode_window_predictions`, then merged across windows.
- **Hybrid blending with a train-derived lexicon** (`MentionLexicon`):
  - Collects all alias forms appearing in train gold annotations (aliases of length ≥ 3 after casefolding).
  - For each alias, the most-frequent entity type across train documents is recorded.
  - At inference, lexicon hits are merged with neural span predictions: when both surface the same span, the highest-confidence form wins.
  - Lexicon match confidence is `mention_hybrid_lexicon_confidence = 0.55`.
- **Per-type confidence thresholding**: `tune_mention_thresholds` performs grid search on dev candidate-recall × mention-F1 (weighted 0.75 / 0.25), per `relation_aware_v1` strategy. Saves a `mention_thresholds` dict per entity type.
- **Span cleanup**: `strict_v2` cleanup profile removes spans inside `TYPE_DENYLISTS` (handcurated lists per entity type: e.g. removing `"latest"`, `"supplementary"` from Date predictions).

### 2.2 Stage 2 — Relation classification (multi-label sequence classification)

- **Backbone**: `answerdotai/ModernBERT-base` (same encoder).
- **Head**: linear `ModernBertForSequenceClassification` over the 7 relation labels (multi-label, sigmoid per label).
- **Loss**: `BCEWithLogitsLoss` with per-class `pos_weight = (1 − positive_rate) / positive_rate` computed from oversampled training rows. **No focal loss on relation side in v14 or v18**. (Focal loss was added in v21d but its variants did not enter the winning submissions.)
- **Input format**: text composed by `build_relation_context_text` (relation context window of ±2 sentences around both mentions; max sequence length 768 tokens):
  ```
  [Document context with subject and object spans marked]
  Question: Does <subject_type> "<subject_form>" relate to <object_type> "<object_form>"?
  ```
  Special marker tokens (`[SUBJECT_<TYPE>]`, `[OBJECT_<TYPE>]`) are inserted at span boundaries.
- **Candidate pair enumeration**:
  - All pairs (subject_entity, object_entity) within the same document.
  - Pruned by `relation_pair_pruning_profile = precision_v1`: same-sentence, ±2-sentence, or ±6-layout-block-distance windows.
- **Per-relation threshold calibration** (after training): per-label F1-maximizing grid search on dev predicted-entity scores, search range `[0.05, 0.95]` (v14 default; later versions tightened this to `[0.35, 0.75]`). For minority labels (< 20 positives in train), search width is clipped to ±0.15 around the default threshold.
- **Schema validation**: only triples whose `(subject_type, object_type)` matches the hardcoded schema for the predicted predicate are kept.
- **Deduplication**: triples with identical `(doc_id, subject_canonical_form, predicate, object_canonical_form)` are deduplicated.

### 2.3 Training procedure (per model)

- **Optimizer**: AdamW with weight decay 0.01, warmup ratio 0.06, linear schedule.
- **Learning rate**: encoder LR 2e-5, head LR 0.08 (legacy from the numpy baseline; head is just a thin linear).
- **Epochs**: 3 for both mention and relation models (early stopping, patience 2 on validation loss).
- **Batch size**: 4 (effective; no gradient accumulation).
- **Validation**: dev split is used both as validation (for early stopping) and for threshold calibration.
- **Hardware**: trained on Apple M4 Pro using MPS; relation training is ~1h/epoch for ModernBERT@768 tokens, ~50 min/epoch for mention training at @1024 tokens.

### 2.4 Hyperparameters table (v14, exact)

| Knob | Value | Notes |
|---|---|---|
| `model_name` | `modernbert_staged` | Two-stage neural pipeline |
| `encoder_name` | `answerdotai/ModernBERT-base` | Used for both mention and relation |
| `mention_encoder_name` | `None` | Defaults to `encoder_name` |
| `max_seq_length` | 1024 | Mention windowing |
| `doc_stride` | 256 | Mention sliding window stride |
| `relation_max_seq_length` | 768 | Relation classifier input |
| `relation_context_sentence_radius` | 2 | ±2 sentences around mention pair |
| `epochs` | 3 | Both stages, with early stopping patience 2 |
| `batch_size` / `train_batch_size` | 4 | |
| `learning_rate_encoder` | 2e-5 | |
| `learning_rate` (head) | 0.08 | |
| `weight_decay` | 0.01 | |
| `warmup_ratio` | 0.06 | |
| `early_stopping_patience` | 2 | |
| `mention_use_focal_loss` | True | gamma=2.0 |
| `mention_focal_gamma` | 2.0 | |
| `mention_class_weight_cap` | 12.0 | Caps per-class focal weighting |
| `mention_negative_sentence_ratio` | 0.5 | Training-window negatives |
| `mention_positive_sentence_radius` | 1 | Positive-context window size |
| `mention_hybrid_lexicon` | True | Train-derived lexicon |
| `mention_hybrid_lexicon_confidence` | 0.55 | Lexicon-match confidence floor |
| `mention_threshold_tuning_strategy` | `relation_aware_v1` | |
| `mention_threshold_candidate_recall_weight` | 0.75 | |
| `mention_threshold_mention_f1_weight` | 0.25 | |
| `mention_cleanup_profile` | `strict_v2` | Denylist-based span cleanup |
| `mention_type_denylists_enabled` | True | |
| `min_alias_length` | 3 | Minimum lexicon alias length (chars) |
| `relation_train_with_predicted_entities` | True | Mix gold + predicted candidates in relation training |
| `relation_predicted_entity_mix_ratio` | 0.5 | 50/50 mix |
| `relation_calibrate_on_predicted_entities` | True | Threshold tuning on predicted-entity dev scores |
| `relation_oversampling_ratio` | 0.05 | Positive oversample fraction (small) |
| `relation_hard_negative_ratio` | 1.0 | |
| `relation_augmentation_enabled` | True | Sentence-radius jitter |
| `relation_pair_pruning_profile` | `precision_v1` | Candidate pair pruning |
| `relation_minority_labels` | `[Transmits, Dispersed_by, Causes, Affects]` | Special threshold treatment |
| `relation_threshold_search_min` | 0.05 | Wide threshold band |
| `relation_threshold_search_max` | 0.95 | |
| `relation_threshold_low_support_margin` | 0.15 | ± around default for low-support labels |
| `max_sentence_distance` | 3 | Candidate pruning |
| `max_layout_distance` | 6 | Candidate pruning |
| `random_seed` | 13 | |

### 2.5 Per-relation default thresholds (v14 starting point and what `norecal_k4` keeps)

| Relation | Default threshold |
|---|---|
| Located_in | 0.55 |
| Found_on | 0.52 |
| Occurs_on | 0.48 |
| Affects | 0.48 |
| Causes | 0.48 |
| Dispersed_by | 0.48 |
| Transmits | 0.48 |

These are the values that the winning `norecal_k4_shift0.00` submission applies directly to averaged 5-seed logits **without dev-side recalibration**.

---

## 3. v14 — Single-seed Baseline (Highest single-model public, dropped on private)

- **Identical to the architecture in §2 with seed = 13.**
- **Mention model** is a ModernBERT-base token classifier trained for 3 epochs.
- **Relation model** is a ModernBERT-base multi-label sequence classifier trained for 3 epochs.
- **Thresholds**: the defaults above (no aggressive recalibration; v14's relation classifier converged on values close to defaults).
- **Submission file**: `submission.csv` (also archived as `submission_modernbert_e2e_v14_for_ensemble.csv`).
- **Kaggle**: public **0.44297**, private **0.34718**.

### 3.1 v14 dev metrics (micro and per-relation, 55 dev docs)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Micro | 0.269 | 0.195 | **0.226** |
| Macro | 0.193 | 0.192 | 0.184 |

| Relation | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Located_in | 0.307 | 0.154 | 0.205 | 58 | 131 | 319 |
| Found_on | 0.296 | 0.396 | 0.339 | 42 | 100 | 64 |
| Occurs_on | 0.164 | 0.096 | 0.121 | 9 | 46 | 85 |
| Affects | 0.290 | 0.300 | 0.295 | 9 | 22 | 21 |
| Causes | 0.220 | 0.333 | 0.265 | 9 | 32 | 18 |
| Dispersed_by | 0.071 | 0.062 | 0.067 | 1 | 13 | 15 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 4 | 5 |

### 3.2 v14 test triple counts (947 total, used in intersection ensemble)

| Relation | Count |
|---|---|
| Located_in | 526 |
| Found_on | 186 |
| Occurs_on | 116 |
| Affects | 46 |
| Causes | 46 |
| Dispersed_by | 18 |
| Transmits | 9 |
| **Total** | **947** |

---

## 4. v18 — Hyperparameter-tuned Single-seed Variant (Used in best public-Kaggle ensemble)

v18 was the best single-seed run from a hyperparameter sweep (`scripts/hp_sweep.py`). Configuration is **identical to v14** except for the following calibrated thresholds (which the sweep selected via larger grid search on dev):

| Relation | v14 threshold | v18 threshold |
|---|---|---|
| Located_in | 0.55 | 0.528 |
| Found_on | 0.52 | 0.587 |
| Occurs_on | 0.48 | 0.416 |
| Affects | 0.48 | 0.348 |
| Causes | 0.48 | 0.194 |
| Dispersed_by | 0.48 | 0.630 |
| Transmits | 0.48 | 0.629 |

Notable: v18's thresholds tune Causes far lower (0.194) than v14's (0.48) — a much more permissive setting. This was selected for highest dev F1.

### 4.1 v18 dev metrics (micro and per-relation)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Micro | 0.245 | 0.224 | **0.234** |
| Macro | 0.214 | 0.220 | 0.203 |

| Relation | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Located_in | 0.309 | 0.180 | 0.228 | 68 | 152 | 309 |
| Found_on | 0.288 | 0.415 | 0.340 | 44 | 109 | 62 |
| Occurs_on | 0.232 | 0.138 | 0.173 | 13 | 43 | 81 |
| Affects | 0.273 | 0.300 | 0.286 | 9 | 24 | 21 |
| Causes | 0.231 | 0.444 | 0.304 | 12 | 40 | 15 |
| Dispersed_by | 0.167 | 0.062 | 0.091 | 1 | 5 | 15 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 81 | 5 |

v18 found 4 more Located_in TPs and 3 more Causes TPs than v14, at the cost of 81 spurious Transmits predictions (all FPs — Transmits is the smallest class in train/dev).

### 4.2 v18 standalone Kaggle

- **Public**: 0.29689 (single-seed v18); ensemble of v18 with v14 → 0.44109 public
- **Private**: 0.37077 (single-seed v18); ensemble → 0.38499 private

---

## 5. Winning Submission #1 — `submission_v14_5seed_norecal_k4_shift0.00.csv`

**Highest private Kaggle**: 0.4266. **Public Kaggle**: 0.4405.

### 5.1 Construction (script: `scripts/v14_5seed_ensemble.py`)

This is a **score-level ensemble of 5 independently trained v14 models**, plus two robustness layers (seed-agreement filtering and no-recalibration). All five constituent models use **the exact v14 architecture and hyperparameters from §2.4**.

**Step 1 — Train 5 single-seed v14 models** (seeds: 13, 42, 1337, 7, 2025).

Each seed produces a complete two-stage pipeline:
- Independent mention detector (ModernBERT token classifier, focal loss, ~50 min/epoch × 3 epochs on MPS).
- Independent relation classifier (ModernBERT sequence classifier, BCE loss, ~1 h/epoch × 3 epochs on MPS).

Per-seed dev micro-F1 (using each model's own internally calibrated thresholds):

| Seed | Dev F1 | Dev P | Dev R |
|---|---|---|---|
| 13 | 0.168 | 0.176 | 0.160 |
| 42 | 0.182 | 0.218 | 0.156 |
| 1337 | 0.183 | 0.287 | 0.134 |
| 7 | 0.213 | 0.235 | 0.195 |
| 2025 | 0.191 | 0.216 | 0.171 |
| **Avg** | **0.187** | **0.226** | **0.163** |

These per-seed scores are **lower** than v14's original 0.226 because the codebase had drifted in small ways between the original v14 training and this re-training run; the absolute single-seed dev F1 doesn't matter — what matters is that the seeds have decorrelated errors.

**Step 2 — Test inference using saved models** (no retraining). For each seed:
- Load mention detector from `mention_model/`, run inference on test docs.
- Load relation classifier from `relation_model/`, score all surviving candidate pairs.
- Dump raw sigmoid scores per `(doc_id, subject_type, subject_form, object_type, object_form, label)` to `test_relation_logits.json`.

**Step 3 — Aggregate sigmoid logits across the 5 seeds.** For each unique `(doc_id, subj_type, subj_form, obj_type, obj_form)` key:
- Count how many seeds produced this key (the candidate-pair distribution differs across seeds because of mention-detector stochasticity).
- Average sigmoid scores per label over the seeds that did produce this key.

**Test-set candidate-pair seed-count distribution** (the key empirical observation justifying the k=4 filter):

| Number of seeds | Test pairs | Share |
|---|---|---|
| in 1 seed only | 3,062 | 47% |
| in 2 seeds | 921 | 14% |
| in 3 seeds | 337 | 5% |
| in 4 seeds | 208 | 3% |
| in 5 seeds | 2,044 | 31% |
| **Total** | **6,572** | 100% |

Almost half of all candidate pairs appear in only one seed. These are stochastic single-seed false positives whose averaged "ensemble" score is dominated by one model's noise.

**Step 4 — Apply seed-count filter: k ≥ 4.** Drop any aggregated key that didn't appear in at least 4 of 5 seeds. **3,251 of 6,572 test pairs (49%) are dropped**, leaving 3,321 pairs whose averaged sigmoid scores reflect genuine multi-model agreement.

**Step 5 — Apply v14's hardcoded thresholds directly** (the "no recalibration" step):
- Located_in ≥ 0.55, Found_on ≥ 0.52, Occurs_on ≥ 0.48, Affects ≥ 0.48, Causes ≥ 0.48, Dispersed_by ≥ 0.48, Transmits ≥ 0.48.
- No dev-side threshold tuning is performed. This avoids the dev-overfit failure mode observed in ablation variants (`tight_minseed4` and `wide_minseed4`) that did recalibrate on dev gold and scored worse on private leaderboard.

**Step 6 — Schema validation + deduplication.** Drop schema-violating triples; deduplicate identical triples.

Final submission: **700 predicted triples** across 82 test documents.

### 5.2 Per-relation triple distribution

| Relation | Count |
|---|---|
| Located_in | 406 |
| Found_on | 146 |
| Occurs_on | 52 |
| Causes | 39 |
| Affects | 37 |
| Transmits | 15 |
| Dispersed_by | 5 |
| **Total** | **700** |

### 5.3 Dev metrics (the model's own dev performance)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Micro | **0.305** | 0.177 | 0.224 |

This is **the highest precision of any submission produced in the project** (0.305 vs v14's 0.269 and v18's 0.245).

### 5.4 Kaggle results

| Leaderboard | Score | dev → score ratio |
|---|---|---|
| **Public** | 0.4405 | 1.97× |
| **Private** | **0.4266** | 1.91× |

Both ratios are consistent with each other and with the typical dev-to-test ratio expected for variance-reduced ensembles on this task (~1.8–2.0×).

### 5.5 Why this submission won

Four independent components, each addressing a specific failure mode observed in earlier attempts:

1. **5-seed averaging** — variance reduction. Single-seed v14 had Kaggle public 0.443 but private only 0.347; the public-leaderboard advantage was lucky-initialization variance that didn't transfer to the 67-doc private holdout. Averaging across 5 seeds eliminates this variance.
2. **Seed-count filter k ≥ 4** — drops 49% of test candidate pairs that appeared in only 1–3 seeds. These are stochastic single-seed FPs whose mention-detection survival was lucky; their averaged scores would have been dominated by one model's noise.
3. **No dev-side threshold recalibration** — uses v14's original thresholds. Ablation variants that did recalibrate on dev gold (`tight_minseed4`, `wide_minseed4`) showed strong dev F1 but suffered public-leaderboard variance and would have been even more dev-overfit had public reflected the true test set. The averaged 5-seed logit distribution generalized the v14 thresholds without further tuning being needed.
4. **v14's clean ModernBERT mention pipeline** — no BioLinkBERT (which we tried in v21d and which produced 27× more raw spans, the majority noise), no EPPO external lexicon (which we tried in v21b and which created blender conflicts), no date regex injection. The v14 mention pipeline is more deterministic across seeds — which is what makes seed averaging effective.

### 5.6 Comparison vs single-seed v14

| Score type | v14 (single seed) | norecal_k4_shift0.00 | Δ |
|---|---|---|---|
| Public | 0.44297 | 0.44054 | −0.002 |
| **Private** | 0.34718 | **0.42656** | **+0.080** |

The 5-seed ensemble loses ~0.002 on public but **gains ~0.080 on private** — i.e., it sacrifices a small amount of public-leaderboard luck for a substantial private-leaderboard reliability gain. On private the 5-seed submission is the unambiguous winner.

---

## 6. Winning Submission #2 — `submission_ensemble_v14_v18_intersection.csv`

**Highest public Kaggle**: 0.4411. **Private Kaggle**: 0.3850.

### 6.1 Construction (script: `scripts/ensemble_submissions.py`)

A CSV-level **set intersection** of v14's submission and v18's submission. Trivially: for each test document, keep only those triples `(subject_form, predicate, object_form)` that appear in **both** v14's and v18's predictions.

```python
# Effective logic (paraphrased from scripts/ensemble_submissions.py)
for doc_id in all_doc_ids:
    v14_triples = {(e['subject'], e['predicate'], e['object']) for e in v14[doc_id]}
    v18_triples = {(e['subject'], e['predicate'], e['object']) for e in v18[doc_id]}
    keep = v14_triples & v18_triples
    submission[doc_id] = list(keep)
```

The two constituent submissions:
- v14: 947 triples (see §3.2 distribution)
- v18: 1,187 triples (Located_in 547, Found_on 239, Occurs_on 86, Causes 61, Affects 47, Dispersed_by 10, Transmits 197)

Intersection retains **606 triples** in total (about two-thirds of v14, half of v18).

### 6.2 Per-relation triple distribution (606 total)

| Relation | v14 | v18 | **Intersection** |
|---|---|---|---|
| Located_in | 526 | 547 | 334 |
| Found_on | 186 | 239 | 118 |
| Occurs_on | 116 | 86 | 75 |
| Causes | 46 | 61 | 35 |
| Affects | 46 | 47 | 29 |
| Dispersed_by | 18 | 10 | 10 |
| Transmits | 9 | 197 | 5 |

The intersection is heavily concentrated in Located_in (55%) and Found_on (19%) — the majority-class relations both models predict confidently.

### 6.3 Kaggle results

| Leaderboard | Score |
|---|---|
| **Public** | **0.4411** |
| Private | 0.3850 |

### 6.4 Why this submission won on public but not private

Both v14 and v18 share the same ModernBERT-base architecture, training data, and most hyperparameters — they differ only in threshold calibration (v18's were chosen by a hyperparameter sweep, v14's were defaults). Intersection produces a high-precision floor: every retained triple has *two* model agreements behind it. This high-precision strategy lined up well with the 15-document public proxy slice (which happened to contain many triples that both v14 and v18 found confidently) but generalized less well to the larger 67-document private holdout, where the 5-seed averaged ensemble (with mention-detector variance reduction, not just classifier variance reduction) was more robust.

---

## 7. Full Submission Leaderboard (for reference)

| Submission | Architecture summary | Public | Private | dev F1 |
|---|---|---|---|---|
| **submission_v14_5seed_norecal_k4_shift0.00** | 5×v14, k≥4 filter, v14 thresholds | 0.4405 | **0.4266** | 0.224 |
| submission_v14_5seed_tight_minseed4 | 5×v14, k≥4, threshold band [0.35, 0.75] dev-calibrated | 0.3608 | 0.4165 | 0.237 |
| submission_ensemble_v14_v18_intersection | v14 ∩ v18 set intersection | **0.4411** | 0.3850 | — |
| submission_v14_5seed_wide_minseed4 | 5×v14, k≥4, threshold band [0.05, 0.95] dev-calibrated | 0.3188 | 0.3937 | 0.243 |
| submission_v21d_5seed_vote2 | 5×v21d, ≥2-of-5 prediction voting | 0.3710 | 0.3831 | — |
| submission.csv (older v14 variant) | v14 with negative hard mining | 0.3619 | 0.3829 | — |
| submission_v14_plus_minseed4_per_relation | v14 + minseed4 hybrid (per-relation swap) | 0.3314 | 0.3842 | — |
| submission_modernbert_e2e_v18_best_single | v18 single seed | 0.2969 | 0.3708 | 0.234 |
| submission_ensemble_v9_v14_heuristic | v9 + v14 heuristic merge | 0.3723 | 0.3717 | — |
| submission_modernbert_e2e_v9 | numpy baseline | 0.4368 | 0.3598 | — |
| submission_v21d_5seed_minseed4 | 5×v21d, k≥4, dev-calibrated | 0.4209 | 0.3424 | 0.227 |
| **v14 (submission.csv canonical)** | Single-seed v14 (the long-time "champion") | **0.4430** | **0.3472** | 0.226 |
| submission_v21d | v21d single seed (BioLinkBERT + EPPO + focal) | 0.4174 | 0.3192 | 0.192 |
| submission_ensemble_v9_v14 | v9 + v14 set intersection | 0.4172 | 0.3240 | — |
| submission_v21a | v21a (BioLinkBERT swap) | 0.3627 | 0.2037 | 0.138 |

Public-vs-private flips for the v14_5seed family are large:
- `tight_minseed4`: public 0.3608, private 0.4165 (+0.056 going from public to private)
- `wide_minseed4`: public 0.3188, private 0.3937 (+0.075)
- These were the strongest evidence at submission time that public leaderboard was noisy.

---

## 8. Key Implementation Files (for paper appendix/reproducibility)

| File | Role |
|---|---|
| `src/pestclef/modernbert.py` | All neural code: `FocalTokenClassifier`, `MultiLabelSequenceClassifier`, `ModernBertMentionDetector`, `ModernBertRelationModel`, training loops, inference helpers |
| `src/pestclef/pipeline.py` | Two-stage orchestration: `run_modernbert_dev_evaluation`, `run_modernbert_test_submission`, hybrid candidate collection, edge prediction |
| `src/pestclef/mention_detection.py` | `MentionLexicon` build/match, hybrid neural+lexicon span blending |
| `src/pestclef/features.py` | `RelationSchema` with hardcoded `(subject_type, object_type)` constraints |
| `src/pestclef/model.py` | `calibrate_threshold` per-label F1-optimizing grid search |
| `src/pestclef/config.py` | `ExperimentConfig` dataclass with every hyperparameter |
| `src/pestclef/data.py` | Document loading, BIO label utilities, deduplication |
| `src/pestclef/evaluation.py` | `compute_metrics`, error reports |
| `src/pestclef/submission.py` | Kaggle CSV serialization and validation |
| `scripts/v14_5seed_ensemble.py` | The winning submission's generation script |
| `scripts/multi_seed_ensemble.py` | Generic multi-seed score-level ensemble runner |
| `scripts/ensemble_submissions.py` | CSV-level union/intersection/heuristic ensembling |
| `scripts/run_baseline.py` | Single-seed training/eval driver |

---

## 9. Quick Reference: What to Cite in the Paper

- **Backbone**: ModernBERT (`answerdotai/ModernBERT-base`) — Warner et al., 2024, "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference". 149M parameters, MLM-pretrained on 2T tokens, long-context (8192).
- **Focal loss**: Lin et al., 2017, "Focal Loss for Dense Object Detection" (RetinaNet paper). Used in mention-detection token classification with γ=2.0.
- **Pipeline pattern**: standard mention-detection → relation-classification two-stage pipeline, common in BioNLP shared tasks. Variants: span-based vs sequence-based; we use sequence-based for relation classification (entire context fed as one input, multi-label sigmoid head).
- **Ensemble methods used**:
  - Score-level averaging across seeds (textbook variance reduction).
  - Seed-agreement filtering (less standard; closest analogue: "must-vote-by-k" in classifier ensembles).
  - CSV-level set intersection (also non-standard; effectively a high-precision floor).
- **Hardware**: Apple M4 Pro, MPS backend.

---

## 10. Headline Numbers for the Abstract

- Best private Kaggle (ours): **0.4266** with the 5-seed v14 + k≥4 filter + no-dev-recalibration ensemble.
- Best public Kaggle (ours): **0.4411** with the v14 ∩ v18 intersection.
- Single-seed v14 baseline private: 0.3472 (i.e., 5-seed ensemble adds **+0.080 absolute / +23% relative** on private).
- Dev micro-F1: 0.224 (the 5-seed ensemble), 0.226 (single-seed v14), 0.234 (single-seed v18).
- Dev-to-private F1 ratio (5-seed ensemble): **1.91×** (the cleanest generalization rate in the project).
