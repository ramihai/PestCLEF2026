#!/usr/bin/env python
"""Generate a Kaggle submission CSV from already-trained model artifacts.

This skips all training — it loads a saved mention model and relation model
from an artifacts directory and runs inference on the test split.

Usage:
    .venv/bin/python scripts/submit_from_artifacts.py \
        --artifacts-dir artifacts/modernbert_e2e_v21a \
        --output submission_v21a.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from pestclef.config import ExperimentConfig
from pestclef.data import deduplicate_edges, load_documents
from pestclef.evaluation import save_predictions
from pestclef.features import RelationSchema
from pestclef.mention_detection import MentionLexicon
from pestclef.modernbert import (
    MultiLabelSequenceClassifier,
    ModernBertMentionDetector,
    ModernBertRelationModel,
    build_relation_inference_examples,
    resolve_torch_device,
)
from pestclef.pipeline import (
    _collect_predicted_entity_state,
    _predict_edges_with_optional_logits,
)
from pestclef.submission import write_submission


def load_mention_detector(
    mention_model_dir: Path, config: ExperimentConfig
) -> ModernBertMentionDetector:
    """Reconstruct a ModernBertMentionDetector from a saved artifacts directory."""
    tokenizer = AutoTokenizer.from_pretrained(mention_model_dir, use_fast=True)
    model_config = AutoConfig.from_pretrained(mention_model_dir)
    base_model = AutoModelForTokenClassification.from_pretrained(
        mention_model_dir, config=model_config
    )
    metadata = json.loads((mention_model_dir / "metadata.json").read_text(encoding="utf-8"))
    labels: list[str] = metadata["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    mention_thresholds: dict[str, float] = {
        k: float(v) for k, v in metadata["mention_thresholds"].items()
    }
    device = resolve_torch_device(config.device)
    detector = ModernBertMentionDetector(
        tokenizer=tokenizer,
        model=base_model,  # type: ignore[arg-type]
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        device=device,
        config=config,
        mention_thresholds=mention_thresholds,
    )
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a submission CSV from pre-trained v21 artifacts (no retraining)."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to the artifacts directory containing mention_model/ and relation_model/",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: submission_<artifacts_dir_name>.csv)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto | mps | cuda | cpu (default: auto)",
    )
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Optional external lexicon JSON (e.g. data/lexicons/eppo_pest_plant_disease.json)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    mention_model_dir = artifacts_dir / "mention_model"
    relation_model_dir = artifacts_dir / "relation_model"

    for d in (mention_model_dir, relation_model_dir):
        if not d.exists():
            print(f"ERROR: missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output) if args.output else Path(f"submission_{artifacts_dir.name}.csv")

    # Load saved experiment config so inference uses the same hyperparameters.
    # We start from a fresh default (so Path fields are proper Path objects) and
    # then overlay the scalar inference-relevant fields from the saved JSON.
    _INFERENCE_FIELDS = {
        "mention_encoder_name", "encoder_name", "mention_max_seq_length",
        "mention_doc_stride", "max_seq_length", "doc_stride",
        "relation_max_seq_length", "relation_context_sentence_radius",
        "mention_hybrid_lexicon", "mention_hybrid_lexicon_confidence",
        "lexicon_external_path", "lexicon_external_confidence",
        "lexicon_external_disabled_types", "mention_date_regex_enabled",
        "mention_positive_sentence_radius", "min_alias_length",
        "max_sentence_distance", "max_layout_distance",
    }
    config = ExperimentConfig()
    config_path = artifacts_dir / "experiment_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8")).get("config", {})
        for field, value in saved.items():
            if field in _INFERENCE_FIELDS and value is not None:
                setattr(config, field, value)
        print(f"[submit] Inference config loaded from {config_path}", flush=True)
    else:
        print("[submit] WARNING: experiment_config.json not found — using defaults", flush=True)

    config.device = args.device
    config.artifacts_dir = artifacts_dir
    if args.lexicon_path:
        config.lexicon_external_path = args.lexicon_path

    print(f"\n[submit] artifacts_dir = {artifacts_dir}", flush=True)
    print(f"[submit] output         = {output_path}", flush=True)
    print(f"[submit] device         = {args.device}", flush=True)
    print(f"[submit] lexicon        = {config.lexicon_external_path}", flush=True)

    # Relation schema is always hardcoded (from_documents delegates to hardcoded())
    schema = RelationSchema.hardcoded()
    print(f"[submit] relation schema: {len(schema.allowed_pairs)} relation types", flush=True)

    # Load models
    print("\n[submit] Loading mention detector...", flush=True)
    t0 = time.time()
    mention_detector = load_mention_detector(mention_model_dir, config)
    print(f"[submit] Mention detector loaded in {time.time()-t0:.1f}s", flush=True)

    print("[submit] Loading relation model...", flush=True)
    t0 = time.time()
    relation_model = ModernBertRelationModel.load(relation_model_dir, config)
    print(f"[submit] Relation model loaded in {time.time()-t0:.1f}s", flush=True)

    # Optional lexicon for hybrid span blending
    if config.mention_hybrid_lexicon:
        train_documents = load_documents("train", config)
        mention_lexicon: MentionLexicon | None = MentionLexicon.from_documents(train_documents, config)
        print(f"[submit] Lexicon built: {len(mention_lexicon.aliases)} aliases", flush=True)
    else:
        mention_lexicon = None

    # Inference on test documents
    print("\n[submit] Loading test documents...", flush=True)
    test_documents = load_documents("test", config)
    n_test = len(test_documents)
    print(f"[submit] {n_test} test documents", flush=True)

    predictions = {}
    predicted_entities_by_doc = {}
    t_infer_start = time.time()

    for i, document in enumerate(test_documents, start=1):
        if i == 1 or i % 10 == 0 or i == n_test:
            elapsed = time.time() - t_infer_start
            print(f"  [{i}/{n_test}] doc_id={document.doc_id} ({elapsed:.1f}s elapsed)", flush=True)

        (
            _neural_spans,
            _lex_spans,
            _blended_spans,
            _mentions,
            entities_by_doc,
            _counts,
        ) = _collect_predicted_entity_state([document], mention_detector, mention_lexicon)

        predicted_entities = entities_by_doc[document.doc_id]
        predicted_entities_by_doc[document.doc_id] = predicted_entities

        examples, entity_pairs = build_relation_inference_examples(
            document, predicted_entities, schema, config
        )
        edges = _predict_edges_with_optional_logits(
            relation_model, examples, entity_pairs, schema, document.doc_id, None
        )
        predictions[document.doc_id] = deduplicate_edges(edges)

    total_relations = sum(len(v) for v in predictions.values())
    print(f"\n[submit] Inference complete in {time.time()-t_infer_start:.1f}s", flush=True)
    print(f"[submit] Total predicted relations: {total_relations}", flush=True)

    # Write submission
    write_submission(predictions, [doc.doc_id for doc in test_documents], output_path)
    print(f"[submit] Submission written → {output_path}", flush=True)

    # Save test predictions for inspection
    test_preds_path = artifacts_dir / "test_predictions.json"
    save_predictions(predictions, test_preds_path)
    print(f"[submit] Test predictions saved → {test_preds_path}", flush=True)

    # Summary
    per_relation: dict[str, int] = {}
    for edges in predictions.values():
        for edge in edges:
            per_relation[edge.predicate] = per_relation.get(edge.predicate, 0) + 1
    print("\n[submit] Per-relation counts:")
    for rel, count in sorted(per_relation.items()):
        print(f"  {rel:22s} {count}")


if __name__ == "__main__":
    main()
