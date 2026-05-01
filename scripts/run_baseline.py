#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.config import ExperimentConfig
from pestclef.pipeline import (
    run_dev_evaluation,
    run_modernbert_dev_evaluation,
    run_modernbert_test_submission,
    run_test_submission,
    train_gold_entity_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PestCLEF baseline.")
    parser.add_argument("--mode", choices=["gold", "end_to_end", "test_submission"], default="gold")
    parser.add_argument("--model", choices=["numpy", "sklearn", "modernbert_staged"], default="numpy")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # v21a: encoder split + date regex
    parser.add_argument(
        "--mention-encoder",
        type=str,
        default=None,
        help="HF model ID for mention detection (default: same as --encoder)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="HF model ID for relation classification",
    )
    parser.add_argument(
        "--mention-max-seq-length",
        type=int,
        default=None,
        help="Max token length for mention encoder (default: config max_seq_length)",
    )
    parser.add_argument(
        "--mention-doc-stride",
        type=int,
        default=None,
        help="Doc stride for mention encoder sliding window",
    )
    parser.add_argument(
        "--date-regex",
        action="store_true",
        help="Enable high-precision date regex candidate injection",
    )

    # v21b: EPPO lexicon
    parser.add_argument(
        "--lexicon-path",
        type=str,
        default=None,
        help="Path to external lexicon JSON (e.g. data/lexicons/eppo_pest_plant_disease.json)",
    )
    parser.add_argument(
        "--lexicon-confidence",
        type=float,
        default=0.70,
        help="Confidence assigned to external lexicon matches (default: 0.70)",
    )
    parser.add_argument(
        "--lexicon-disable-types",
        type=str,
        default="",
        help="Comma-separated entity types to exclude from external lexicon (e.g. Plant,Disease)",
    )

    # Threshold calibration band (v21a showed the [0.05, 0.95] default is too wide
    # when mention noise increases — calibrator swings to 0.20 or 0.95 and destroys F1)
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.35,
        help="Floor for per-relation threshold search (default: 0.35)",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.75,
        help="Ceiling for per-relation threshold search (default: 0.75)",
    )

    # Path 2 / Fix C: align relation training distribution with inference
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=None,
        help=(
            "relation_predicted_entity_mix_ratio override (default: config 0.5; "
            "use 1.0 to train fully on predicted-entity examples — matches inference)"
        ),
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=None,
        help=(
            "relation_oversampling_ratio override (default: config 0.05; "
            "use 0.20+ to compensate for larger negative pool from BioLinkBERT mentions)"
        ),
    )

    # v21c: logit saving
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Dump relation sigmoid logits to artifacts dir (needed for multi-seed ensemble)",
    )

    args = parser.parse_args()

    disabled_types = [t.strip() for t in args.lexicon_disable_types.split(",") if t.strip()]

    config = ExperimentConfig(
        artifacts_dir=Path(args.artifacts_dir),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_batch_size=max(1, min(args.batch_size, 8)),
        eval_batch_size=max(1, min(args.batch_size, 16)),
        random_seed=args.seed,
        encoder_name=args.encoder,
        mention_encoder_name=args.mention_encoder,
        mention_max_seq_length=args.mention_max_seq_length,
        mention_doc_stride=args.mention_doc_stride,
        mention_date_regex_enabled=args.date_regex,
        lexicon_external_path=args.lexicon_path,
        lexicon_external_confidence=args.lexicon_confidence,
        lexicon_external_disabled_types=disabled_types,
        relation_threshold_search_min=args.threshold_min,
        relation_threshold_search_max=args.threshold_max,
        save_relation_logits=args.save_logits,
        **(
            {"relation_predicted_entity_mix_ratio": args.mix_ratio}
            if args.mix_ratio is not None else {}
        ),
        **(
            {"relation_oversampling_ratio": args.oversample}
            if args.oversample is not None else {}
        ),
    )

    if args.model == "modernbert_staged":
        if args.mode == "gold":
            from pestclef.pipeline import train_gold_entity_modernbert_baseline
            result = train_gold_entity_modernbert_baseline(config)
        elif args.mode == "end_to_end":
            result = run_modernbert_dev_evaluation(config)
        else:
            result = run_modernbert_test_submission(config)
    else:
        if args.mode == "gold":
            result = train_gold_entity_baseline(config)
        elif args.mode == "end_to_end":
            result = run_dev_evaluation(config)
        else:
            result = run_test_submission(config)

    if isinstance(result, dict) and "metrics" in result:
        print(json.dumps(result["metrics"], indent=2))
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
