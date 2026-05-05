#!/usr/bin/env python
"""5-seed v14 ensemble — no double retraining.

This is the v14 equivalent of multi_seed_ensemble.py but with two important
differences:

1. **No test_submission retraining.** After per-seed dev_evaluation trains and
   saves both the mention detector and relation model, we *load* those saved
   weights and run test inference once. The original multi_seed flow called
   run_modernbert_test_submission which retrained from scratch, which (a) doubled
   per-seed compute and (b) produced a different best-checkpoint epoch because
   the val split differs between dev_evaluation and test_submission. v21d-seed42
   dropped from 0.417 (original) to 0.391 (5-seed run) for exactly this reason.

2. **v14's exact config**, including the wide threshold band [0.05, 0.95] and
   epochs=3. v14 used ModernBERT for both mention and relation (no BioLinkBERT,
   no EPPO, no date regex, no focal loss on relations).

Hypothesis: v14's mention pipeline is more deterministic than v21d's BioLinkBERT
+ EPPO blend, so the candidate-set divergence problem (only 18% of test pairs
in all 5 seeds for v21d) should be much smaller. A v14 5-seed should ensemble
cleanly and add 1-3 F1 points on top of v14's 0.443 Kaggle baseline.

Compute estimate: ~3h/seed on MPS × 5 seeds = ~15h total (vs v21d's ~50h).

Usage:
  # Full run (trains 5 seeds, then aggregates)
  python scripts/v14_5seed_ensemble.py \\
    --seeds 13,42,1337,7,2025 \\
    --artifacts-root artifacts/modernbert_e2e_v14_5seed \\
    --output submission_v14_5seed.csv

  # Aggregation only (after seeds are trained)
  python scripts/v14_5seed_ensemble.py --skip-train \\
    --seeds 13,42,1337,7,2025 \\
    --artifacts-root artifacts/modernbert_e2e_v14_5seed \\
    --min-seed-count 3 \\
    --output submission_v14_5seed_minseed3.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from pestclef.config import ExperimentConfig
from pestclef.data import deduplicate_edges, load_documents
from pestclef.evaluation import compute_metrics
from pestclef.features import RelationSchema
from pestclef.mention_detection import MentionLexicon
from pestclef.modernbert import (
    ModernBertMentionDetector,
    ModernBertRelationModel,
    build_relation_inference_examples,
    resolve_torch_device,
)
from pestclef.pipeline import (
    _collect_predicted_entity_state,
    _predict_edges_with_optional_logits,
    run_modernbert_dev_evaluation,
)
from pestclef.submission import write_submission

# Reuse aggregator + threshold-calibrator from multi_seed_ensemble.py
from multi_seed_ensemble import _aggregate_logits, _calibrate_thresholds, _edges_from_aggregated


@dataclass
class V14SeedConfig:
    """v14's exact training config — verified against the saved
    artifacts/modernbert_e2e_v14/experiment_config.json on the parent branch."""
    seeds: List[int] = field(default_factory=lambda: [13, 42, 1337, 7, 2025])
    artifacts_root: Path = Path("artifacts/modernbert_e2e_v14_5seed")
    encoder_name: str = "answerdotai/ModernBERT-base"
    # v14 used the SAME encoder for mention and relation (no BioLinkBERT split)
    mention_encoder_name: str = ""  # empty = use encoder_name
    max_seq_length: int = 1024
    doc_stride: int = 256
    relation_max_seq_length: int = 768
    relation_context_sentence_radius: int = 2
    epochs: int = 3
    batch_size: int = 4
    relation_oversampling_ratio: float = 0.05
    relation_predicted_entity_mix_ratio: float = 0.5
    relation_hard_negative_ratio: float = 1.0
    # v14's wide threshold band — wider than v21d's [0.35, 0.75] because v14's
    # candidate distribution is cleaner so calibration doesn't go to extremes
    relation_threshold_search_min: float = 0.05
    relation_threshold_search_max: float = 0.95
    # NOT enabled in v14
    relation_use_focal_loss: bool = False
    mention_date_regex_enabled: bool = False
    lexicon_external_path: str = ""  # empty = no EPPO
    output_csv: Path = Path("submission_v14_5seed.csv")


def _materialise_v14_seed_config(base: V14SeedConfig, seed: int) -> ExperimentConfig:
    """Build an ExperimentConfig for one seed, exactly matching v14."""
    config = ExperimentConfig()
    config.model_name = "modernbert_staged"
    config.encoder_name = base.encoder_name
    # v14 had no separate mention encoder — leave mention_encoder_name as default
    # (ExperimentConfig defaults to None which means: use encoder_name)
    config.max_seq_length = base.max_seq_length
    config.doc_stride = base.doc_stride
    config.relation_max_seq_length = base.relation_max_seq_length
    config.relation_context_sentence_radius = base.relation_context_sentence_radius
    config.epochs = base.epochs
    config.batch_size = base.batch_size
    config.train_batch_size = max(1, min(base.batch_size, 8))
    config.eval_batch_size = max(1, min(base.batch_size, 16))
    config.relation_oversampling_ratio = base.relation_oversampling_ratio
    config.relation_predicted_entity_mix_ratio = base.relation_predicted_entity_mix_ratio
    config.relation_hard_negative_ratio = base.relation_hard_negative_ratio
    config.relation_threshold_search_min = base.relation_threshold_search_min
    config.relation_threshold_search_max = base.relation_threshold_search_max
    config.relation_use_focal_loss = base.relation_use_focal_loss
    config.mention_date_regex_enabled = base.mention_date_regex_enabled
    if base.lexicon_external_path:
        config.lexicon_external_path = base.lexicon_external_path
    # v14 default thresholds — calibration will tune from these
    config.relation_thresholds = {
        "Located_in": 0.55, "Found_on": 0.52, "Occurs_on": 0.48,
        "Affects": 0.48, "Causes": 0.48, "Dispersed_by": 0.48, "Transmits": 0.48,
    }
    config.random_seed = seed
    config.save_relation_logits = True  # critical for the aggregator
    config.artifacts_dir = base.artifacts_root.parent / f"{base.artifacts_root.name}_seed{seed}"
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return config


def _load_saved_mention_detector(mention_dir: Path, config: ExperimentConfig) -> ModernBertMentionDetector:
    """Reconstruct mention detector from saved HF artifacts + metadata.json."""
    tokenizer = AutoTokenizer.from_pretrained(mention_dir, use_fast=True)
    model_config = AutoConfig.from_pretrained(mention_dir)
    base_model = AutoModelForTokenClassification.from_pretrained(mention_dir, config=model_config)
    metadata = json.loads((mention_dir / "metadata.json").read_text(encoding="utf-8"))
    labels: list[str] = metadata["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    mention_thresholds: dict[str, float] = {
        k: float(v) for k, v in metadata["mention_thresholds"].items()
    }
    return ModernBertMentionDetector(
        tokenizer=tokenizer,
        model=base_model,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        device=resolve_torch_device(config.device),
        config=config,
        mention_thresholds=mention_thresholds,
    )


def _run_test_inference_from_saved(
    config: ExperimentConfig,
    schema: RelationSchema,
) -> None:
    """Load saved mention + relation models for this seed, run test inference,
    save test_relation_logits.json. NO retraining."""
    artifacts_dir = config.artifacts_dir
    mention_dir = artifacts_dir / "mention_model"
    relation_dir = artifacts_dir / "relation_model"
    if not (mention_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"No mention model weights at {mention_dir}")
    if not (relation_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"No relation model weights at {relation_dir}")

    print(f"  loading saved mention detector from {mention_dir}", flush=True)
    mention_detector = _load_saved_mention_detector(mention_dir, config)
    print(f"  loading saved relation model from {relation_dir}", flush=True)
    relation_model = ModernBertRelationModel.load(relation_dir, config)

    train_documents = load_documents("train", config)
    mention_lexicon = (
        MentionLexicon.from_documents(train_documents, config)
        if config.mention_hybrid_lexicon else None
    )
    if mention_lexicon is not None:
        print(f"  lexicon: {len(mention_lexicon.aliases)} aliases", flush=True)

    test_documents = load_documents("test", config)
    n = len(test_documents)
    print(f"  running test inference on {n} docs", flush=True)
    score_records: List[Dict[str, object]] = []
    t0 = time.time()
    for i, document in enumerate(test_documents, start=1):
        if i == 1 or i % 10 == 0 or i == n:
            print(f"    [{i}/{n}] doc_id={document.doc_id} ({time.time()-t0:.1f}s elapsed)", flush=True)
        (_n, _l, _b, _m, entities_by_doc, _c) = _collect_predicted_entity_state(
            [document], mention_detector, mention_lexicon
        )
        predicted_entities = entities_by_doc[document.doc_id]
        examples, entity_pairs = build_relation_inference_examples(
            document, predicted_entities, schema, config
        )
        # _predict_edges_with_optional_logits appends to score_records
        _predict_edges_with_optional_logits(
            relation_model, examples, entity_pairs, schema, document.doc_id, score_records,
        )
    out_path = artifacts_dir / "test_relation_logits.json"
    out_path.write_text(
        json.dumps({"records": score_records, "labels": list(relation_model.labels)},
                   ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  saved {len(score_records)} test logit records → {out_path}", flush=True)


def _load_logits(path: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    if not path.exists():
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("records", [])), list(raw.get("labels", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=str, default="13,42,1337,7,2025")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/modernbert_e2e_v14_5seed"))
    parser.add_argument("--epochs", type=int, default=3, help="v14 used 3; default keeps that")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold-min", type=float, default=0.05, help="v14's wide band default")
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip per-seed training; only aggregate existing logits")
    parser.add_argument(
        "--min-seed-count", type=int, default=1,
        help="Drop pairs not in at least this many seeds before averaging logits (default 1 = no filter)"
    )
    parser.add_argument("--output", type=Path, default=Path("submission_v14_5seed.csv"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    base = V14SeedConfig(
        seeds=seeds,
        artifacts_root=args.artifacts_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        relation_threshold_search_min=args.threshold_min,
        relation_threshold_search_max=args.threshold_max,
        output_csv=args.output,
    )

    schema = RelationSchema.hardcoded()
    seed_configs: Dict[int, ExperimentConfig] = {}
    per_seed_metrics: Dict[int, Dict[str, object]] = {}

    for seed in seeds:
        seed_config = _materialise_v14_seed_config(base, seed)
        seed_config.device = args.device
        seed_configs[seed] = seed_config

        if not args.skip_train:
            print(f"\n[seed={seed}] === training (dev_evaluation) → {seed_config.artifacts_dir}", flush=True)
            t0 = time.time()
            dev_result = run_modernbert_dev_evaluation(seed_config)
            per_seed_metrics[seed] = {"dev": dev_result.get("metrics")}
            print(f"[seed={seed}] dev_evaluation finished in {time.time()-t0:.1f}s", flush=True)

            print(f"\n[seed={seed}] === test inference (NO retrain — using saved models)", flush=True)
            t0 = time.time()
            _run_test_inference_from_saved(seed_config, schema)
            print(f"[seed={seed}] test inference finished in {time.time()-t0:.1f}s", flush=True)
        else:
            print(f"[seed={seed}] --skip-train: assuming logits already exist at {seed_config.artifacts_dir}", flush=True)

    # Aggregate
    dev_records_per_seed: List[List[Dict[str, object]]] = []
    test_records_per_seed: List[List[Dict[str, object]]] = []
    label_set: List[str] = []
    for seed in seeds:
        dev_records, dev_labels = _load_logits(seed_configs[seed].artifacts_dir / "dev_relation_logits.json")
        test_records, test_labels = _load_logits(seed_configs[seed].artifacts_dir / "test_relation_logits.json")
        if dev_labels and not label_set:
            label_set = list(dev_labels)
        elif test_labels and not label_set:
            label_set = list(test_labels)
        dev_records_per_seed.append(dev_records)
        test_records_per_seed.append(test_records)
        print(f"[aggregate] seed{seed}: dev={len(dev_records)}, test={len(test_records)} records", flush=True)

    if not label_set:
        raise SystemExit("No relation logits found — run without --skip-train at least once.")

    print(f"\n[aggregate] min_seed_count={args.min_seed_count}", flush=True)
    aggregated_dev, dev_stats = _aggregate_logits(dev_records_per_seed, label_set, args.min_seed_count)
    aggregated_test, test_stats = _aggregate_logits(test_records_per_seed, label_set, args.min_seed_count)
    print(f"[aggregate]   dev: kept {dev_stats['kept']}/{dev_stats['total_keys']}, dropped {dev_stats['dropped']}")
    print(f"[aggregate]   test: kept {test_stats['kept']}/{test_stats['total_keys']}, dropped {test_stats['dropped']}")
    dev_hist = {k: v for k, v in dev_stats.items() if k.startswith("in_")}
    test_hist = {k: v for k, v in test_stats.items() if k.startswith("in_")}
    print(f"[aggregate]   dev seed-count histogram: {dev_hist}")
    print(f"[aggregate]   test seed-count histogram: {test_hist}")

    # Calibrate thresholds on dev
    print(f"\n[calibrate] grid search in [{args.threshold_min:.2f}, {args.threshold_max:.2f}]", flush=True)
    dev_documents = load_documents("dev", ExperimentConfig())
    seed_thresholds = ExperimentConfig().relation_thresholds
    thresholds, threshold_diag = _calibrate_thresholds(
        aggregated_dev, dev_documents, label_set, seed_thresholds=seed_thresholds,
        grid_min=args.threshold_min, grid_max=args.threshold_max,
    )
    print("[calibrate] thresholds:")
    for label, t in thresholds.items():
        print(f"  {label:<22} {t:.3f}")

    # Apply to dev (sanity) and test (the submission)
    ensemble_dev_edges = _edges_from_aggregated(aggregated_dev, thresholds)
    ensemble_dev_metrics = compute_metrics(dev_documents, ensemble_dev_edges, list(label_set))
    print(f"\n[ensemble dev] micro F1={ensemble_dev_metrics['micro']['f1']:.3f} "
          f"P={ensemble_dev_metrics['micro']['precision']:.3f} "
          f"R={ensemble_dev_metrics['micro']['recall']:.3f}")
    print("[ensemble dev] per-relation:")
    for r, m in ensemble_dev_metrics["per_relation"].items():
        print(f"  {r:<22} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | tp={m['tp']} fp={m['fp']} fn={m['fn']}")

    ensemble_test_edges = _edges_from_aggregated(aggregated_test, thresholds)
    test_documents = load_documents("test", ExperimentConfig())
    write_submission(
        ensemble_test_edges,
        [document.doc_id for document in test_documents],
        base.output_csv,
    )

    summary_suffix = f"_minseed{args.min_seed_count}" if args.min_seed_count > 1 else ""
    summary_path = base.artifacts_root.parent / f"{base.artifacts_root.name}_ensemble_summary{summary_suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({
            "seeds": seeds,
            "label_set": label_set,
            "min_seed_count": args.min_seed_count,
            "per_seed_metrics": per_seed_metrics,
            "ensemble_dev_metrics": ensemble_dev_metrics,
            "ensemble_thresholds": thresholds,
            "threshold_diagnostics": threshold_diag,
            "submission_path": str(base.output_csv),
            "ensemble_dev_record_count": len(aggregated_dev),
            "ensemble_test_record_count": len(aggregated_test),
            "dev_aggregation_stats": dev_stats,
            "test_aggregation_stats": test_stats,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[done] Submission → {base.output_csv}")
    print(f"[done] Summary    → {summary_path}")


if __name__ == "__main__":
    main()
