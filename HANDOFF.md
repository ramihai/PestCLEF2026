# PestCLEF 2026 Handoff

## Current State
- Implemented a runnable baseline pipeline for document-level binary relation extraction.
- Added reusable code under `src/pestclef/` for:
  - data loading and schema normalization
  - canonical entity construction and alias alignment
  - candidate generation and feature extraction
  - multi-label relation classification
  - mention detection for test-time inference
  - evaluation, error reporting, and submission validation
- Added runnable scripts:
  - `python scripts/run_baseline.py --mode gold`
  - `python scripts/run_baseline.py --mode end_to_end`
  - `python scripts/generate_submission.py`
- Added regression tests in `tests/test_pipeline.py`

## Verified Results
- `python -m unittest discover -s tests -v`
  - passed
- `python scripts/run_baseline.py --mode gold --artifacts-dir artifacts/smoke_gold`
  - dev micro F1: `0.1548`
  - dev macro F1: `0.1481`
- `python scripts/run_baseline.py --mode end_to_end --artifacts-dir artifacts/smoke_e2e`
  - dev micro F1: `0.1553`
  - dev macro F1: `0.1146`
- `python scripts/generate_submission.py --artifacts-dir artifacts/smoke_submit`
  - generated and validated `submission.csv`

## Important Files
- `src/pestclef/data.py`: unified document loading, mention parsing, canonicalization, KG alias alignment
- `src/pestclef/features.py`: relation schema, candidate pruning, pair features
- `src/pestclef/model.py`: pure-NumPy multi-label classifier with threshold calibration
- `src/pestclef/mention_detection.py`: dictionary-based mention detection and canonical entity prediction
- `src/pestclef/pipeline.py`: train/eval/submission orchestration
- `src/pestclef/submission.py`: CSV and JSON submission validation
- `tests/test_pipeline.py`: smoke/regression coverage

## Environment Notes
- This implementation was kept lightweight because the working environment did not have a usable `torch` / `transformers` stack.
- `sklearn` was also unusable due to a NumPy compatibility issue.
- Because of that, the current baseline is pure Python + NumPy and is suitable as a local notebook-first starting point.
- The code is structured so a future transformer-based model can replace the current classifier without rewriting the whole data or evaluation pipeline.

## Recommended Next Steps
1. Recreate the environment on the new machine and verify `python -m unittest discover -s tests -v`.
2. Install a working `torch` + `transformers` stack with `mps` support if you want to move from the NumPy baseline to an encoder-based model.
3. Keep the current data layer and submission tooling, and swap the relation model in `src/pestclef/model.py` / `src/pestclef/pipeline.py`.
4. Improve end-to-end extraction by replacing dictionary mention detection with a trained entity detector.
5. Re-run dev evaluation and compare against the saved smoke artifacts before changing decoding or thresholding policy.

## Artifacts
- `artifacts/smoke_gold/`: gold-entity baseline metrics, predictions, schema, saved model
- `artifacts/smoke_e2e/`: end-to-end dev metrics, predictions, error report, mention lexicon
- `artifacts/smoke_submit/`: test predictions and predicted entities used to generate `submission.csv`

## Notes For The Next Codex Session
- Do not assume conversation memory carries over.
- Start by reading `HANDOFF.md`, `implementation_plan.MD`, and `src/pestclef/pipeline.py`.
- If moving to a transformer model, preserve:
  - canonical entity naming policy
  - submission validation rules
  - train/dev evaluation outputs and saved error buckets
