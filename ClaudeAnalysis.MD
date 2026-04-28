# Claude Project Analysis: PestCLEF 2026

*Companion to `GeminiAnalysis.MD` and `HANDOFF.md`. Authored by Claude (Opus 4.7) on 2026-04-28 to ground the scientific paper that follows the competition.*

---

## 1. Task and dataset

**PestCLEF 2026** is a document-level relation extraction shared task on the EPOP corpus (European and Mediterranean Plant Protection Organization). Given a heterogeneous document (news brief, scientific abstract, regulatory bulletin), we must extract a knowledge graph of (subject, predicate, object) triples over 7 entity types and 7 relations.

| Split | Docs | Entity mentions | Relations |
|---|---|---|---|
| Train | 110 | 3,482 | 1,374 |
| Dev   |  55 | 1,615 |   656 |
| Test  |  82 | — (no gold) | — |

Entity-type frequency (train + dev): Location 1,527, Pest 1,246, Plant 1,010, Date 636, Disease 382, Dissemination_pathway 186, Vector 110. Relation frequency (train + dev): Located_in 1,158 (57%), Found_on 339, Occurs_on 304, Causes/Affects/Dispersed_by/Transmits combined < 200. Document length is moderate — median ~400 whitespace tokens, ~17 sentences. Roughly 7–11% of documents per split are non-English (French, Spanish, Italian) and present as either originals or machine-translated text per `data/EPOP_documents/README.md`.

Submission format is a Kaggle CSV with two columns: `doc_id`, `knowledge_graph` (a JSON-serialized list of triples). Scoring is a custom token-level F1 over knowledge graphs.

## 2. Architecture summary

The pipeline is a **staged transformer cascade** organised in `src/pestclef/`:

1. **Mention detection** ([src/pestclef/modernbert.py:521](src/pestclef/modernbert.py:521)) — token classification on ModernBERT-base with BIO tags over 7 entity types. Loss is focal (γ=2.0, [src/pestclef/config.py:42](src/pestclef/config.py:42)) with class weights capped at 12. Decoded via sliding window (max_seq_length=1024, doc_stride=256). Predicted spans are blended with a **lexicon stream** built from training entities + a configurable confidence floor ([src/pestclef/modernbert.py:749-774](src/pestclef/modernbert.py:749-774)). Per-type denylists at [src/pestclef/modernbert.py:110](src/pestclef/modernbert.py:110) suppress recurrent false positives.
2. **Coreference / canonical entity construction** ([src/pestclef/data.py:85](src/pestclef/data.py:85), [src/pestclef/modernbert.py:937](src/pestclef/modernbert.py:937)) — union-find over identity-coreference blocks at training time; at inference time a heuristic merges by normalized form, singularization, species abbreviation pattern (`B. xylophilus` ↔ `Bursaphelenchus xylophilus`), and acronyms.
3. **Candidate pruning** ([src/pestclef/features.py:177](src/pestclef/features.py:177)) — type-pair filter from a hardcoded relation schema, plus distance gates (`max_sentence_distance=3`, `max_layout_distance=6`).
4. **Relation classification** ([src/pestclef/modernbert.py:1225](src/pestclef/modernbert.py:1225)) — multi-label sequence classification with **entity markers** (`<SUBJ_Pest> ... </SUBJ_Pest>` / `<OBJ_Plant> ... </OBJ_Plant>`) over a 2-sentence radius (`relation_context_sentence_radius=2`). Backbone: ModernBERT-base again, max length 768. Sigmoid + BCE-with-logits with `pos_weight` rebalancing.
5. **Calibration** ([src/pestclef/modernbert.py:990](src/pestclef/modernbert.py:990), [src/pestclef/model.py](src/pestclef/model.py)) — per-relation thresholds tuned on dev via grid search (range 0.05–0.95, low-support margin 0.15 for relations with <20 positives).
6. **Submission writing** ([src/pestclef/submission.py](src/pestclef/submission.py), [scripts/generate_submission.py](scripts/generate_submission.py)) — schema masking + JSON serialisation.

Auxiliary baselines remain for sanity (NumPy linear model, sklearn logistic regression — see `src/pestclef/model.py`); they are 0.10–0.18 dev F1 and serve as floors.

## 3. Version-by-version journey

The table below normalises Gemini's prior log with corrected dev metrics from the saved artifacts (`dev_end_to_end_metrics.json` per version). v15 was a heuristic CSV ensemble of v9+v14, not a trained run.

| Ver | Description | Dev micro F1 | Dev micro P | Dev micro R | Kaggle |
|---:|---|---:|---:|---:|---:|
| baseline (NumPy, gold) | features.py only | 0.155 | — | — | — |
| sklearn (e2e) | logistic regression | ~0.13 | — | — | 0.260 |
| v5  | ModernBERT staged, loose cleanup | 0.162 | — | — | 0.333 |
| v7  | strict_v1 cleanup + entity merge | 0.212 | — | — | 0.388 |
| v8  | strict_v2 cleanup + relation-aware threshold tuning | 0.208 | — | — | 0.372 |
| v9  | per-relation threshold search 0.05–0.95 | **0.215** | 0.271 | 0.179 | **0.437** |
| v10 | relaxed pruning (sent_dist 5, layout 10), hybrid_conf 0.65 | 0.196 | 0.20 | — | 0.217 (regression) |
| v11 | revert distances, ctx_radius 2, rule-NER | 0.200 | 0.26 | — | n/a |
| v12 | per-class thresholds (Located_in 0.55) | 0.184 | — | — | n/a |
| v13 | oversample 0.10 + jitter, 3 epochs | 0.163 | — | — | n/a |
| v14 | oversample 0.05, hard_neg 1.0, ctx_radius 2 | **0.226** | 0.269 | 0.195 | **0.443** |
| v16 | dynamic hard negatives epoch 1 | 0.213 | — | — | n/a |
| v17 | distant supervision on (~18 KB injections) | 0.193 | — | — | n/a |
| v18 | distant supervision off, recall-leaning thresholds | **0.234** | 0.245 | 0.224 | 0.297 (overfit) |
| v19 | re-enabled distant supervision | 0.186 | — | — | n/a |
| v20 | mention candidate-recall weight 0.85 | 0.187 | — | — | n/a |
| ensemble v9+v14 (intersection) | hand-merged CSVs | — | — | — | 0.417 |
| ensemble v14+v18 (intersection) | hand-merged CSVs | — | — | — | **0.441** |

**Headline.** v14 is still the best leaderboard model at **0.44297**. v18 reached the highest dev F1 (0.234) but Kaggle was only **0.297** — a clean dev/test divergence that justifies multi-seed ensembling and explicit regularisation in v21.

## 4. Where v14 fails — quantitative bottleneck analysis

From `artifacts/modernbert_e2e_v14/dev_candidate_recall.json` (655 missed gold relations):

| Failure bucket | Count | Share |
|---|---:|---:|
| **bad_span** — mention detector failed to surface either argument | **427** | **65.2%** |
| valid candidate, classifier missed it (`valid_candidate − predicted = 49`) | 49 | 7.5% |
| candidate_missed_by_pruning (distance / type filter) | 29 | 4.4% |
| bad_clustering_or_canonicalization | 16 | 2.4% |
| other / multi-cause | balance | balance |

**Mention recall** (`dev_entity_metrics.json`, v14):

| Type | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|
| Disease | 0.61 | 0.29 | 0.39 | 36 / 23 / 89 |
| Date | 0.60 | 0.12 | 0.20 | 26 / 17 / 191 |
| Diss. path. | 0.18 | 0.17 | 0.17 | 8 / 36 / 40 |
| Location | 0.69 | 0.29 | 0.40 | 135 / 61 / 338 |
| Pest | 0.69 | 0.24 | 0.36 | 72 / 32 / 224 |
| Plant | 0.63 | 0.33 | 0.43 | 110 / 65 / 224 |
| **Vector** | **0.02** | **0.08** | **0.04** | 2 / 83 / 24 |

The mention layer is **precision-rich, recall-starved**. The Vector classifier is essentially broken (F1 0.04), which explains the persistent **0.0 F1 on Transmits** since Transmits requires Vector → Disease/Pest pairs.

**Per-relation v14 metrics** (TP / FP / FN):
- Located_in 58 / 131 / 319 — the recall sink (319 missed)
- Found_on 42 / 100 / 64 — most balanced
- Affects 9 / 22 / 21 — classifier improved sharply vs v9 (+87% F1)
- Causes 9 / 32 / 18 — still imprecise
- Occurs_on 9 / 46 / 85 — Date recall drags this down
- Dispersed_by 1 / 13 / 15
- Transmits 0 / 4 / 5 — irrecoverable without external supervision

## 5. The relation classifier itself is **not** the bottleneck — the entity layer is

A subtle but important finding emerges when we line up `prediction_recall_after_candidates` (the share of *valid* candidates the classifier successfully promotes to a prediction):

| Version | candidate_recall | prediction_recall_after_candidates |
|---|---:|---:|
| v9 | 0.296 | 0.649 |
| v14 | 0.279 | 0.732 |
| v18 | 0.296 | **0.809** |

v18's relation classifier was the *strongest yet* — once a candidate pair reaches it, it gets the right label 81% of the time. The reason v18 still flopped on Kaggle is that the gain came from **lower thresholds**, which in turn injected enough false positives (Transmits FP = 81 — see `dev_end_to_end_metrics.json`) to wreck precision in the wild. The implication for v21 is sharp: lifting the **27.9% candidate-recall ceiling** is worth more than any further classifier tuning.

## 6. Variance study (HP sweep, `modernbert_sweep_summary.json`)

A 3×3 grid over `learning_rate_encoder` ∈ {1e-5, 2e-5, 3e-5} and `warmup_ratio` ∈ {0.0, 0.06, 0.1} with the rest of v14's config produced this dev micro F1 spread (best to worst):

```
lr3e-05_w0.1   0.2032
lr3e-05_w0.0   0.2029
lr1e-05_w0.0   0.2012
lr2e-05_w0.1   0.1999
lr1e-05_w0.06  0.1979
lr1e-05_w0.1   0.1965
lr2e-05_w0.06  0.1929   ← v14's actual config in this run
lr3e-05_w0.06  0.1801
lr2e-05_w0.0   0.1781
```

Two takeaways:

1. **The same v14 config produced 0.226 in the original run, 0.193 in the sweep** — a 3.3-point std purely from initialisation/data-shuffle randomness on the small classification head. This is the single biggest argument for multi-seed ensembling in v21.
2. The HP grid is **flat within ±0.025** — encoder learning rate and warmup are second-order knobs at this scale. Spending more cycles on them is wasted effort.

## 7. What we know doesn't work

From the existing 20 versions:

- **Aggressive context expansion** (v10's distance 5/10): drops F1 by 2 points and Kaggle by 22 points. Stay tight.
- **Heavy oversampling alone** (v13's ratio 0.10): hallucinates minority relations, especially Transmits (75 FP from nothing). Pair oversampling with hard negatives or skip.
- **Rule-based NER replacing the neural span** (v11's regex Date/Vector): breaks Date F1 in half. Rules can supplement, not replace.
- **Distant supervision with a tiny KB** (v17/v19, ~18 injections): adds noise without enough signal. Either go big or go home.
- **Threshold tuning past v14's per-class settings** (v12, v18, v20): hits diminishing returns; trades dev F1 for Kaggle stability poorly.

## 8. v21 plan in one paragraph

v21 attacks the bad-span bottleneck (item-1 in §4) and the variance ceiling (§6) at the same time, with three independent contributions ablated as separate runs: (a) swap the **mention** encoder to BioLinkBERT-base while keeping ModernBERT-base for relation classification, plus a high-precision Date regex stream; (b) ingest the **EPPO Global Database** Pest/Plant/Disease lists as an external lexicon merged into the existing hybrid blender; (c) train 5 seeds of the best v21 config and average sigmoid logits before recalibrating per-relation thresholds. The full plan with file/line edits, verification gates, and target metrics lives in [.claude/plans/we-are-working-on-zany-micali.md](.claude/plans/we-are-working-on-zany-micali.md). Predicted lift: dev candidate recall 0.279 → ≥0.36, dev micro F1 0.226 → ≥0.24, Kaggle 0.443 → ≥0.46.

## 9. Open problems carried forward (post-v21)

1. **Multilingual** — neither ModernBERT nor BioLinkBERT is multilingual; an estimated 7–11% of test docs are French/Spanish/Italian. v22 candidate: language-detect non-English docs and route them to xlm-roberta-base for mention detection.
2. **Long-range relations** — `Located_in` has 319/377 false negatives in v14. Many gold pairs span >2 sentences. v23 candidate: full-doc relation classifier exploiting ModernBERT's native long-context (8192 tokens).
3. **Cross-doc + intra-doc coreference** — `the vector` / `the insect` / `the disease` never resolve to the specific named entity. Even v14's species-abbreviation matcher misses pronoun chains. A focused coreference upgrade likely converts 20–30 missed Transmits/Causes/Affects relations.
4. **Minority class floor** — Transmits has only 5 dev / 25 train examples; expecting non-zero F1 without LLM-based synthesis or external KB augmentation is unrealistic. We accept this floor and report it transparently in the paper.
5. **Dev/test distribution shift** — v18 made this concrete: dev 0.234 → Kaggle 0.297 is a 14-point gap. Worth analysing test docs vs dev docs along surface features (length, biomedical jargon density, original language) to identify what dev under-represents.
6. **LLM verifier cascade** — explicitly parked for v21 (user choice). Reasonable v22+ candidate: Claude Haiku as a sentence-grounded yes/no verifier on relations with logit confidence in [0.40, 0.55].

## 10. Reproducibility note

All runs reported above are saved verbatim in `artifacts/modernbert_e2e_v*/` with the exact `experiment_config.json`, dev metrics, candidate-recall breakdown, error report, predicted mentions, predicted entities, and per-relation thresholds. The HP sweep is in `artifacts/modernbert_sweep_*` plus `modernbert_sweep_summary.json`. Submitted CSVs are in the repo root (`submission_modernbert_e2e_v*.csv`). Kaggle scores are tracked manually in `kaggle_scores.MD`.

Hardware used to date: M4 Pro Mac (MPS device fallback path is wired in [src/pestclef/modernbert.py:154](src/pestclef/modernbert.py:154)). v21's 5-seed ensemble is planned to run on Colab T4/L4 to keep wall-clock reasonable.

Author: Mihai Radulescu (team **BioNLP412**). Coding-assistant collaboration log: Codex (early scaffolding through v17-ish), Gemini CLI (v18 sweep + analysis on 2026-04-20), Claude Opus 4.7 (this analysis + v21 plan, 2026-04-28).
