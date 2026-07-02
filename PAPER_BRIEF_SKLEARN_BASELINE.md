# PestCLEF 2026 — Paper Brief: scikit-learn Logistic Regression Baseline

A self-contained reference for the working notes paper's baseline section. Covers exactly what the sklearn baseline does, how it differs from the ModernBERT system, what features it uses, the scikit-learn settings, and its dev/Kaggle results.

---

## 1. Classification framing

The sklearn baseline frames relation extraction as **7 independent binary logistic-regression classifiers — one per relation label — applied per candidate entity pair**. It does **not** classify candidate triples directly; instead, for each `(subject_entity, object_entity)` pair, every binary classifier produces a probability for "does this pair stand in relation R?", and the final triple set is the union of all (pair × relation) decisions that pass thresholding and schema validation.

Concretely:

- **Unit of classification**: ordered candidate pair `(subject_canonical_entity, object_canonical_entity)`. The dataset is built by `generate_relation_examples` (in `src/pestclef/features.py`), which produces one feature dictionary per pair.
- **Pair enumeration**: for every document, all distinct pairs of canonical entities are considered, subject to a sentence/layout-distance pruning filter (`max_sentence_distance=3`, `max_layout_distance=6`) and to the relation-schema's type compatibility (a pair whose `(subject_type, object_type)` doesn't appear in any relation's allowed set is discarded). This pruning is shared with the ModernBERT model and is not specific to the baseline.
- **Multi-label decoding**: each binary classifier is queried independently; a pair can predict 0, 1, or more relations. Identical triples are deduplicated; schema-invalid triples (e.g. predicting `Affects` between a `Pest` and a `Date`) are filtered post hoc.
- **Per-label schema masking at training time**: when training the classifier for relation R, only pairs whose `(subject_type, object_type)` is in R's allowed set are used as training data. Pairs that *could* be R but happen to be negatives are kept; pairs that can never be R (e.g. `Plant → Date` pairs cannot be Causes) are excluded entirely. This avoids forcing the classifier to learn the schema constraints that we already encode explicitly.

### Entity source

- **Training**: uses **gold canonical entities** from the train documents' annotations. The candidate pairs come from `document.canonical_entities`, which on train/dev is populated from gold mentions during the data loading step.
- **Development evaluation (`--mode gold`)**: uses gold dev entities. This is the **"oracle-entity" baseline number**: `dev_gold_entity_metrics.json` in `artifacts/sklearn_gold/`. It measures relation-classification quality in isolation from mention-detection errors.
- **Development evaluation (`--mode end_to_end`)**: uses **predicted entities** obtained from a lightweight train-derived `MentionLexicon` (string-matching mention detection — alias forms collected from gold train mentions, used to surface mentions in dev documents). Results in `artifacts/sklearn_e2e/dev_end_to_end_metrics.json`.
- **Kaggle test submission**: uses the same lexicon-based mention detection (no ModernBERT mention detector for the sklearn baseline). The test pipeline lives in `run_test_submission` (the non-modernbert branch) in `src/pestclef/pipeline.py`.

The two dev numbers (gold vs end-to-end) bracket the baseline's contribution: the gap between 0.380 (gold entities) and 0.247 (lexicon-detected entities) quantifies how much error the upstream mention-detection step contributes when a string-matching lexicon is used instead of a neural detector.

---

## 2. Feature representation

The baseline does **not** use TF-IDF and does not use any continuous text representation. Features are a **hand-engineered sparse binary feature dictionary** built per candidate pair by `extract_pair_features` (in `src/pestclef/features.py`), then vectorized with `sklearn.feature_extraction.DictVectorizer(sparse=True)`. Every feature value is `1.0` (presence indicator); no IDF weighting, no term-frequency counts, no learned dense embeddings.

The feature dictionary contains seven groups (all are sparse binary indicators, joined by the DictVectorizer into a single sparse CSR matrix):

### 2.1 Entity-type features

| Feature key template | Example | Purpose |
|---|---|---|
| `subject_type=<TYPE>` | `subject_type=Pest` | One-hot subject type |
| `object_type=<TYPE>` | `object_type=Location` | One-hot object type |
| `type_pair=<S>-><O>` | `type_pair=Pest->Location` | One-hot ordered type pair (gives the classifier direct access to schema-aligned signal) |

### 2.2 Distance / layout features (all bucketized to keep features categorical)

| Feature | Buckets | Notes |
|---|---|---|
| `sentence_gap=<k>` | k ∈ {0, 1, 2, 3, 4, 5} (capped at 5) | Min sentence-index distance between any mention of subject and any mention of object |
| `layout_gap=<k>` | k ∈ {0, …, 8} (capped at 8) | Min layout-block distance (a layout block ≈ a paragraph or labelled span from the PDF) |
| `surface_bucket=<bucket>` | `<64`, `<256`, `<1024`, `>=1024` | Bucketized absolute character offset between subject's and object's earliest start positions |
| `between_bucket=<bucket>` | Same buckets | Bucketized character span between the closest mention pair (right_mention.start − left_mention.end) |
| `same_sentence=<0/1>` | binary | 1 if sentence_gap == 0 |
| `same_layout=<0/1>` | binary | 1 if layout_gap == 0 |
| `subject_first=<0/1>` | binary | 1 if subject's earliest_start ≤ object's earliest_start (textual order) |

### 2.3 Mention-count features

| Feature | Buckets | Notes |
|---|---|---|
| `subject_mentions=<k>` | k ∈ {0, 1, 2, 3, 4} (capped at 4) | Number of surface mentions of the subject entity |
| `object_mentions=<k>` | k ∈ {0, 1, 2, 3, 4} | Same for object |
| `pair_mentions=<bucket>` | `1`, `2`, `3_4`, `5+` | Bucketized product (subject_mentions × object_mentions) |
| `subject_name_len=<bucket>` | Same buckets | Bucketized token count of subject's canonical form |
| `object_name_len=<bucket>` | Same buckets | Same for object |

### 2.4 Canonical-name features (binary)

| Feature | Example |
|---|---|
| `subject_name=<casefolded canonical>` | `subject_name=tobrfv virus` |
| `object_name=<casefolded canonical>` | `object_name=europe` |

These two features make the model effectively memorize specific subject/object identities that occur frequently in training; they are not normalized or weighted by frequency.

### 2.5 Bag-of-words "context" features (binary, top-k)

Two text windows around the closest mention pair are tokenized using a simple regex tokenizer (`[a-z][a-z\-]+`, casefolded), and the **top-k most frequent tokens** in each window are emitted as binary features. There is no TF-IDF re-weighting and no global vocabulary — each pair gets its own top-k tokens. Stopword removal is not applied (so tokens like `the`, `of`, `in` are common features, which the classifier can use or ignore).

| Feature group | Window | Top-k | Token feature key |
|---|---|---|---|
| `context=<token>` | ±100 characters around `[min(first_start, second_end), max(first_start, second_end)]` | 8 | `context=virus` |
| `between=<token>` | Just the span between the closest mention pair's `left.end` and `right.start` | 10 | `between=spread` |
| `subject_token=<token>` | The first 4 tokens of the subject's canonical form | n/a | `subject_token=tobrfv` |
| `object_token=<token>` | The first 4 tokens of the object's canonical form | n/a | `object_token=europe` |

So the "TF-IDF"-style component is really a sparse bag-of-binary-tokens per window, no IDF, no n-grams, no character n-grams. The top-k cap (per pair, per window) keeps the feature space tractable.

### 2.6 Hand-curated relation-trigger features (binary)

For each of the 7 relations, a small handcurated list of trigger keywords:

```python
RELATION_TRIGGER_KEYWORDS = {
    "Affects":      ("affect", "attack", "damage", "impact", "symptom"),
    "Causes":       ("cause", "caused", "causing", "induces", "results"),
    "Dispersed_by": ("dispersed", "spread", "spreads", "via", "through"),
    "Found_on":     ("found", "detected", "observed", "present", "collected"),
    "Located_in":   ("in", "inside", "within", "from", "region"),
    "Occurs_on":    ("during", "in", "on", "season", "year"),
    "Transmits":    ("transmit", "transmitted", "vector", "carry", "carried"),
}
```

If *any* keyword from R's list appears in either the `context=` token set or the `between=` token set, the binary feature `trigger_hint=<R>` is added to the pair's feature dict. This gives each per-relation classifier a strong distillation feature for its target relation.

### 2.7 Feature vectorization summary

- All feature values are `1.0` (binary indicator).
- The `DictVectorizer(sparse=True)` is fit on the training set; unseen features at inference are silently dropped.
- No `TfidfTransformer` and no `CountVectorizer` are used. The reason TF-IDF isn't applied: the bag-of-words component is per-pair top-k, not global, so it isn't a fixed-vocabulary BoW vector that TF-IDF could re-weight.
- Typical feature-vector dimensionality after fitting on train: a few thousand active features per training matrix.

---

## 3. scikit-learn settings

For each of the 7 relations, a separate `sklearn.linear_model.LogisticRegression` is fit on schema-valid pairs (`src/pestclef/model.py:243–250`):

```python
estimator = LogisticRegression(
    C=config.sklearn_c,                # 1.0
    class_weight="balanced",
    max_iter=config.sklearn_max_iter,  # 400
    random_state=config.random_seed,   # 13
    solver="liblinear",
)
estimator.fit(x_label, y)
```

### 3.1 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `solver` | `liblinear` | Coordinate-descent solver, used because (a) it handles sparse inputs efficiently, (b) it supports L1 and L2 directly, (c) it's well-suited to binary classification per label. |
| Penalty | L2 (sklearn `liblinear` default) | Equivalent to ridge regularization. |
| `C` | 1.0 | Inverse regularization strength. Default value; not tuned. Equivalent to a moderate L2 penalty. |
| `class_weight` | `"balanced"` | Sets `class_weight[i] = n_samples / (n_classes × bincount(y)[i])` — the standard inverse-frequency reweighting. Important because positive rates for each relation in training are low (e.g. Transmits has ~5 positives in dev). |
| `max_iter` | 400 | Increased from sklearn's default 100 to ensure convergence on the sparser, class-weighted problem. |
| `random_state` | 13 | Reproducibility seed (only affects internal `liblinear` tie-breaking, not data ordering). |
| `dual` | False (default) | Primal form preferred for sparse-feature high-instance problems. |
| `fit_intercept` | True (default) | |
| `tol` | 1e-4 (default) | |

### 3.2 Degenerate-label handling

If, after schema masking, a binary label has no positives or no negatives, the per-label classifier falls back to a constant-score `BinaryLabelModel` (with score equal to the single observed class) rather than throwing an error. This handles rare relations on small train splits.

### 3.3 Per-relation threshold calibration

After fitting each per-relation classifier, the decision threshold is **re-tuned to maximize F1 on the training set's own scores** (this is technically training-set calibration, not dev-set calibration — `train_sklearn_relation_model` does not see dev when picking thresholds):

```python
score_column = estimator.predict_proba(x_label)[:, 1]
thresholds[label] = calibrate_threshold(
    score_column, y, default_threshold=config.relation_thresholds.get(label, 0.5)
)
```

`calibrate_threshold` (in `src/pestclef/model.py:267–295`) does a simple grid search over ~30 thresholds (`np.quantile(scores, np.linspace(0.6, 0.995, 30))`) plus the configured default, picking the threshold that maximizes F1 on `(scores, y)`. This is the same calibrator used by the ModernBERT model.

### 3.4 Inference

`SklearnRelationModel.predict_scores` produces a `(n_pairs × 7)` probability matrix by running each binary classifier independently on the shared DictVectorizer output. Triples are emitted where `score ≥ threshold` for that label, then the schema validity check filters out type-mismatched triples, then duplicates are removed.

---

## 4. Results

### 4.1 Configuration used for the reported sklearn baseline

| Knob | Value |
|---|---|
| `model_name` | `sklearn` |
| `sklearn_c` | 1.0 |
| `sklearn_max_iter` | 400 |
| `feature_cap` | 25 000 (a `FeatureVectorizer` cap that the sklearn path doesn't actually use — it has its own DictVectorizer; this is leftover from the numpy baseline) |
| `max_sentence_distance` | 3 |
| `max_layout_distance` | 6 |
| `random_seed` | 13 |
| `relation_thresholds` (defaults before calibration) | All 0.48 |

### 4.2 Dev micro-F1

| Setup | Entity source | Micro P | Micro R | Micro F1 |
|---|---|---|---|---|
| **sklearn_gold** | Gold dev entities | **0.510** | 0.302 | **0.380** |
| **sklearn_e2e** | Lexicon-predicted entities | 0.280 | 0.221 | 0.247 |

(Macro-F1 for `sklearn_gold`: 0.370; for `sklearn_e2e`: 0.212.)

The **0.380 → 0.247 gap (Δ ≈ 0.13 F1)** is what the upstream lexicon-based mention detector loses relative to gold-perfect entities. This is the key motivation for the ModernBERT mention detector in the main system.

### 4.3 Per-relation dev breakdown — `sklearn_gold` (gold entities)

| Relation | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Located_in | 0.544 | 0.260 | 0.352 | 98 | 82 | 279 |
| Found_on | 0.575 | 0.472 | 0.518 | 50 | 37 | 56 |
| Occurs_on | 0.352 | 0.202 | 0.257 | 19 | 35 | 75 |
| Affects | 0.524 | 0.367 | 0.431 | 11 | 10 | 19 |
| Causes | 0.425 | 0.630 | 0.507 | 17 | 23 | 10 |
| Dispersed_by | 0.400 | 0.125 | 0.190 | 2 | 3 | 14 |
| Transmits | 1.000 | 0.200 | 0.333 | 1 | 0 | 4 |

### 4.4 Per-relation dev breakdown — `sklearn_e2e` (predicted entities via lexicon)

| Relation | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Located_in | 0.288 | 0.204 | 0.239 | 77 | 190 | 300 |
| Found_on | 0.304 | 0.292 | 0.298 | 31 | 71 | 75 |
| Occurs_on | 0.250 | 0.160 | 0.195 | 15 | 45 | 79 |
| Affects | 0.391 | 0.300 | 0.340 | 9 | 14 | 21 |
| Causes | 0.224 | 0.407 | 0.289 | 11 | 38 | 16 |
| Dispersed_by | 0.125 | 0.125 | 0.125 | 2 | 14 | 14 |
| Transmits | 0.000 | 0.000 | 0.000 | 0 | 1 | 5 |

### 4.5 Triple counts on test split (sklearn_submit, predicted via lexicon)

The sklearn baseline produces a test-split prediction file in `artifacts/sklearn_submit/test_predictions.json` (predicted entities via the train-derived lexicon, then per-relation classifiers as above). Triple totals:

| Relation | Count |
|---|---|
| Located_in | 786 |
| Found_on | 173 |
| Occurs_on | 106 |
| Causes | 51 |
| Affects | 42 |
| Transmits | 13 |
| Dispersed_by | 4 |
| **Total** | **1 175** |

### 4.6 Kaggle scores

The sklearn baseline reached **0.40341 public / 0.26020 private**.

---

## 5. Key implementation files (for the appendix)

| File | Role |
|---|---|
| `src/pestclef/model.py` | `train_sklearn_relation_model`, `SklearnRelationModel`, `BinaryLabelModel`, threshold `calibrate_threshold` |
| `src/pestclef/features.py` | `extract_pair_features`, `generate_relation_examples`, `enumerate_candidate_entity_pairs`, `RELATION_TRIGGER_KEYWORDS`, `RelationSchema` |
| `src/pestclef/pipeline.py` | `train_gold_entity_baseline`, `run_dev_evaluation`, `run_test_submission` (non-modernbert branches) |
| `src/pestclef/mention_detection.py` | `MentionLexicon` (used by the sklearn end-to-end pipeline to surface candidate entities at inference) |
| `scripts/run_baseline.py` | CLI entrypoint with `--model sklearn` |

CLI command used:

```bash
python scripts/run_baseline.py --model sklearn --mode gold --artifacts-dir artifacts/sklearn_gold
python scripts/run_baseline.py --model sklearn --mode end_to_end --artifacts-dir artifacts/sklearn_e2e
python scripts/run_baseline.py --model sklearn --mode test_submission --artifacts-dir artifacts/sklearn_submit
```

---

## 6. One-paragraph summary suitable for paste-into-paper

> The scikit-learn baseline frames relation extraction as seven independent binary logistic-regression classifiers (one per relation label) over ordered candidate entity pairs `(subject, object)`. Each pair receives a sparse hand-engineered feature vector containing entity-type indicators, ordered type-pair, sentence/layout-distance buckets, mention-count buckets, casefolded canonical-name indicators, per-pair top-k bag-of-words features over a ±100-character context window and the inter-mention span, the first few tokens of each canonical name, and a handcurated keyword-trigger indicator per relation. All features are binary; no TF-IDF or learned representations are used. Pairs are vectorized via `sklearn.feature_extraction.DictVectorizer` and each per-relation classifier is trained only on pairs whose `(subject_type, object_type)` is schema-compatible with that relation. The estimator is `LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', max_iter=400, random_state=13)` with L2 regularization. Per-relation decision thresholds are post-hoc calibrated by F1-maximizing grid search on training-set scores. On dev with gold canonical entities the baseline achieves micro-F1 0.380 (P=0.510, R=0.302); when paired with a string-matching MentionLexicon for entity prediction in the end-to-end setting it drops to 0.247 (P=0.280, R=0.221). The baseline was submitted to Kaggle; the model scored 0.40341 public / 0.26020 private on the leaderboard.
