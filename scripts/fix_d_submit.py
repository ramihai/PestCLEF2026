#!/usr/bin/env python
"""Fix D — generate a submission using v21b's mention model + v14's relation classifier.

Hypothesis: v21b has substantially better mention recall (Vector 0.594 vs 0.031,
Pest 0.778 vs 0.654, candidate_recall 0.347 vs 0.279) but its relation classifier
converts only 46% of valid candidates to TPs vs v14's 73%. Pairing v21b's mentions
with v14's classifier should keep the recall gains while restoring conversion.

The first dev run revealed v14's thresholds (calibrated for v14's small/clean
candidate pool) are too lax for v21b's 2x larger pool — recall improved (0.235
vs 0.195) but precision crashed (0.136 vs 0.269). The fix is to *recalibrate*
v14's thresholds against the new candidate distribution. This script supports:

  --calibrate-thresholds : (dev mode) tune thresholds in [0.35, 0.75] band against
                           dev gold relations and use them; saves them for reuse.
  --thresholds-from PATH : (test mode) load thresholds calibrated on dev.

Usage:
  # Step 1: calibrate on dev
  .venv/bin/python scripts/fix_d_submit.py \\
    --mention-artifacts artifacts/modernbert_e2e_v21b \\
    --relation-artifacts /path/to/parent/artifacts/modernbert_e2e_v14 \\
    --mode dev --calibrate-thresholds --device mps

  # Step 2: apply to test for Kaggle submission
  .venv/bin/python scripts/fix_d_submit.py \\
    --mention-artifacts artifacts/modernbert_e2e_v21b \\
    --relation-artifacts /path/to/parent/artifacts/modernbert_e2e_v14 \\
    --mode test --thresholds-from fix_d_thresholds.json \\
    --output submission_fix_d.csv --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from pestclef.config import ExperimentConfig
from pestclef.data import deduplicate_edges, load_documents
from pestclef.evaluation import compute_metrics, save_predictions
from pestclef.features import RelationSchema
from pestclef.mention_detection import MentionLexicon
from pestclef.model import calibrate_threshold
from pestclef.modernbert import (
    ModernBertMentionDetector,
    ModernBertRelationModel,
    build_relation_inference_examples,
    resolve_torch_device,
)
from pestclef.pipeline import _collect_predicted_entity_state
from pestclef.schema import CanonicalEntity, Document, RelationEdge
from pestclef.submission import write_submission


_MENTION_INFERENCE_FIELDS = {
    "mention_encoder_name", "mention_max_seq_length", "mention_doc_stride",
    "mention_hybrid_lexicon", "mention_hybrid_lexicon_confidence",
    "lexicon_external_path", "lexicon_external_confidence",
    "lexicon_external_disabled_types", "mention_date_regex_enabled",
    "mention_positive_sentence_radius", "min_alias_length",
}
_RELATION_INFERENCE_FIELDS = {
    "encoder_name", "max_seq_length", "doc_stride",
    "relation_max_seq_length", "relation_context_sentence_radius",
    "max_sentence_distance", "max_layout_distance",
}


def _overlay_saved_config(config: ExperimentConfig, artifacts_dir: Path, fields: set[str], label: str) -> None:
    config_path = artifacts_dir / "experiment_config.json"
    if not config_path.exists():
        print(f"[fix_d] WARNING: {label} experiment_config.json not found at {config_path}", flush=True)
        return
    saved = json.loads(config_path.read_text(encoding="utf-8")).get("config", {})
    overlaid = []
    for field, value in saved.items():
        if field in fields and value is not None:
            setattr(config, field, value)
            overlaid.append(field)
    print(f"[fix_d] {label} config overlay: {len(overlaid)} fields from {config_path.name}", flush=True)


def load_mention_detector(mention_dir: Path, config: ExperimentConfig) -> ModernBertMentionDetector:
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
    device = resolve_torch_device(config.device)
    return ModernBertMentionDetector(
        tokenizer=tokenizer,
        model=base_model,  # type: ignore[arg-type]
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        device=device,
        config=config,
        mention_thresholds=mention_thresholds,
    )


# ---- Score collection (split out from _predict_edges_with_optional_logits) ----

def collect_doc_scores(
    document: Document,
    mention_detector: ModernBertMentionDetector,
    mention_lexicon: MentionLexicon | None,
    relation_model: ModernBertRelationModel,
    schema: RelationSchema,
    config: ExperimentConfig,
) -> Tuple[List[Tuple[CanonicalEntity, CanonicalEntity]], np.ndarray]:
    """Run inference for one doc and return (entity_pairs, scores [n_pairs, n_labels])."""
    (_n, _l, _b, _m, entities_by_doc, _c) = _collect_predicted_entity_state(
        [document], mention_detector, mention_lexicon
    )
    predicted_entities = entities_by_doc[document.doc_id]
    examples, entity_pairs = build_relation_inference_examples(
        document, predicted_entities, schema, config
    )
    if not examples:
        return [], np.zeros((0, len(relation_model.labels)), dtype=np.float32)
    scores = relation_model.predict_scores(examples)
    return list(entity_pairs), scores


def edges_from_scores(
    entity_pairs: Sequence[Tuple[CanonicalEntity, CanonicalEntity]],
    scores: np.ndarray,
    labels: Sequence[str],
    thresholds: Dict[str, float],
    schema: RelationSchema,
) -> List[RelationEdge]:
    edges: List[RelationEdge] = []
    for row_index, (subject, obj) in enumerate(entity_pairs):
        row = scores[row_index]
        for col, label in enumerate(labels):
            score = float(row[col])
            if score < float(thresholds.get(label, 0.5)):
                continue
            if not schema.is_valid_pair(label, subject.entity_type, obj.entity_type):
                continue
            edges.append(RelationEdge(subject=subject.canonical_form, predicate=label, object=obj.canonical_form))
    return edges


# ---- Threshold calibration on dev candidates ----

def calibrate_dev_thresholds(
    doc_data: Dict[str, Tuple[List[Tuple[CanonicalEntity, CanonicalEntity]], np.ndarray]],
    documents: Sequence[Document],
    labels: Sequence[str],
    schema: RelationSchema,
    default_thresholds: Dict[str, float],
    band: Tuple[float, float] = (0.35, 0.75),
) -> Dict[str, float]:
    """For each label, find the threshold in `band` that maximizes F1 on dev.

    Aligns each candidate (doc, subj, obj) against dev gold by canonical-form
    string match, which is how `compute_metrics` evaluates relations.
    """
    docs_by_id = {doc.doc_id: doc for doc in documents}

    # Build (score_array, gold_array) per label by walking all candidate pairs.
    # We treat schema-invalid pairs as "not a candidate" — they get score=0 / gold=0
    # so they don't perturb tuning.
    per_label_scores: Dict[str, List[float]] = {label: [] for label in labels}
    per_label_gold: Dict[str, List[float]] = {label: [] for label in labels}

    for doc_id, (entity_pairs, scores) in doc_data.items():
        document = docs_by_id.get(doc_id)
        if document is None or scores.shape[0] == 0:
            continue
        gold_keys: set[Tuple[str, str, str]] = {
            (edge.subject, edge.predicate, edge.object)
            for edge in document.gold_relation_edges
        }
        for row_index, (subject, obj) in enumerate(entity_pairs):
            for col, label in enumerate(labels):
                if not schema.is_valid_pair(label, subject.entity_type, obj.entity_type):
                    # Don't include schema-invalid pairs in calibration
                    continue
                per_label_scores[label].append(float(scores[row_index, col]))
                gold_match = (subject.canonical_form, label, obj.canonical_form) in gold_keys
                per_label_gold[label].append(1.0 if gold_match else 0.0)

    calibrated: Dict[str, float] = {}
    print(f"\n[fix_d] Calibrating thresholds in band [{band[0]:.2f}, {band[1]:.2f}]:", flush=True)
    for label in labels:
        s = np.asarray(per_label_scores[label], dtype=np.float32)
        g = np.asarray(per_label_gold[label], dtype=np.float32)
        positives = int(g.sum())
        n_candidates = int(s.size)
        default = float(default_thresholds.get(label, 0.5))
        if n_candidates == 0:
            calibrated[label] = default
            print(f"  {label:<22} no candidates — keep default {default:.3f}", flush=True)
            continue
        new_t = calibrate_threshold(
            s, g, default, min_threshold=band[0], max_threshold=band[1]
        )
        # Compute f1 at old vs new threshold for diagnostic
        def _f1_at(t: float) -> Tuple[float, int, int, int]:
            pred = (s >= t).astype(np.float32)
            tp = int(((pred == 1) & (g == 1)).sum())
            fp = int(((pred == 1) & (g == 0)).sum())
            fn = int(((pred == 0) & (g == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            return f1, tp, fp, fn
        old_f1, old_tp, old_fp, _ = _f1_at(default)
        new_f1, new_tp, new_fp, _ = _f1_at(new_t)
        calibrated[label] = float(new_t)
        print(
            f"  {label:<22} "
            f"n={n_candidates:<5} pos={positives:<3} | "
            f"default {default:.3f} F1={old_f1:.3f} (tp={old_tp} fp={old_fp}) → "
            f"calibrated {new_t:.3f} F1={new_f1:.3f} (tp={new_tp} fp={new_fp})",
            flush=True,
        )
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix D: v21b mention encoder + v14 relation classifier (with optional threshold calibration).")
    parser.add_argument("--mention-artifacts", required=True)
    parser.add_argument("--relation-artifacts", required=True)
    parser.add_argument("--mode", choices=["test", "dev"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="(dev mode only) Tune thresholds against dev gold and use them. Saves to fix_d_thresholds.json.",
    )
    parser.add_argument(
        "--thresholds-from",
        default=None,
        help="Load per-label thresholds from a JSON file produced by --calibrate-thresholds.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.35,
        help="Lower bound for threshold calibration (default: 0.35)",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.75,
        help="Upper bound for threshold calibration (default: 0.75)",
    )
    parser.add_argument(
        "--save-thresholds",
        default="fix_d_thresholds.json",
        help="Where to write calibrated thresholds (default: fix_d_thresholds.json)",
    )
    args = parser.parse_args()

    if args.calibrate_thresholds and args.mode != "dev":
        print("ERROR: --calibrate-thresholds is only valid with --mode dev", file=sys.stderr)
        sys.exit(1)

    mention_dir = Path(args.mention_artifacts).resolve() / "mention_model"
    relation_dir = Path(args.relation_artifacts).resolve() / "relation_model"
    for d in (mention_dir, relation_dir):
        if not d.exists():
            print(f"ERROR: missing directory: {d}", file=sys.stderr); sys.exit(1)
        if not ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
            print(f"ERROR: no model weights in {d}", file=sys.stderr); sys.exit(1)

    # Build config: relation fields from v14 (relation-artifacts), mention from v21b
    config = ExperimentConfig()
    _overlay_saved_config(config, Path(args.relation_artifacts).resolve(), _RELATION_INFERENCE_FIELDS, "relation")
    _overlay_saved_config(config, Path(args.mention_artifacts).resolve(), _MENTION_INFERENCE_FIELDS, "mention")
    config.device = args.device

    print(f"\n[fix_d] mention_dir       = {mention_dir}", flush=True)
    print(f"[fix_d] relation_dir      = {relation_dir}", flush=True)
    print(f"[fix_d] mode              = {args.mode}", flush=True)
    print(f"[fix_d] calibrate         = {args.calibrate_thresholds}", flush=True)
    print(f"[fix_d] thresholds_from   = {args.thresholds_from}", flush=True)

    schema = RelationSchema.hardcoded()

    print("\n[fix_d] Loading mention detector (v21b)...", flush=True)
    t0 = time.time()
    mention_detector = load_mention_detector(mention_dir, config)
    print(f"[fix_d] Mention detector loaded in {time.time()-t0:.1f}s", flush=True)

    print("[fix_d] Loading relation model (v14)...", flush=True)
    t0 = time.time()
    relation_model = ModernBertRelationModel.load(relation_dir, config)
    print(f"[fix_d] Relation model loaded in {time.time()-t0:.1f}s", flush=True)
    default_thresholds = dict(relation_model.thresholds)
    print(f"[fix_d] v14 default thresholds: {default_thresholds}", flush=True)

    # Lexicon from v21b's training-document set (with EPPO via config.lexicon_external_path)
    if config.mention_hybrid_lexicon:
        train_documents = load_documents("train", config)
        mention_lexicon: MentionLexicon | None = MentionLexicon.from_documents(train_documents, config)
        print(f"[fix_d] Lexicon: {len(mention_lexicon.aliases)} aliases", flush=True)
    else:
        mention_lexicon = None

    print(f"\n[fix_d] Loading {args.mode} documents...", flush=True)
    documents = load_documents(args.mode, config)
    n_docs = len(documents)
    print(f"[fix_d] {n_docs} {args.mode} documents", flush=True)

    # Pass 1: collect scores per doc (no thresholding yet)
    doc_data: Dict[str, Tuple[List[Tuple[CanonicalEntity, CanonicalEntity]], np.ndarray]] = {}
    t_start = time.time()
    for i, document in enumerate(documents, start=1):
        if i == 1 or i % 10 == 0 or i == n_docs:
            print(f"  [{i}/{n_docs}] doc_id={document.doc_id} ({time.time()-t_start:.1f}s elapsed)", flush=True)
        entity_pairs, scores = collect_doc_scores(
            document, mention_detector, mention_lexicon, relation_model, schema, config
        )
        doc_data[document.doc_id] = (entity_pairs, scores)
    print(f"[fix_d] Score collection complete in {time.time()-t_start:.1f}s", flush=True)

    # Decide which thresholds to use
    labels = list(relation_model.labels)
    if args.calibrate_thresholds:
        thresholds = calibrate_dev_thresholds(
            doc_data, documents, labels, schema, default_thresholds,
            band=(args.threshold_min, args.threshold_max),
        )
        save_path = Path(args.save_thresholds)
        save_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"\n[fix_d] Calibrated thresholds saved → {save_path}", flush=True)
    elif args.thresholds_from:
        loaded = json.loads(Path(args.thresholds_from).read_text(encoding="utf-8"))
        thresholds = {label: float(loaded.get(label, default_thresholds.get(label, 0.5))) for label in labels}
        print(f"\n[fix_d] Loaded thresholds from {args.thresholds_from}: {thresholds}", flush=True)
    else:
        thresholds = default_thresholds
        print(f"\n[fix_d] Using v14 default thresholds (no calibration).", flush=True)

    # Pass 2: emit edges with chosen thresholds
    predictions: Dict[str, List[RelationEdge]] = {}
    for doc_id, (entity_pairs, scores) in doc_data.items():
        edges = edges_from_scores(entity_pairs, scores, labels, thresholds, schema)
        predictions[doc_id] = deduplicate_edges(edges)
    total_relations = sum(len(v) for v in predictions.values())
    print(f"\n[fix_d] Total predicted relations: {total_relations}", flush=True)

    if args.mode == "test":
        output_path = Path(args.output) if args.output else Path("submission_fix_d.csv")
        write_submission(predictions, [d.doc_id for d in documents], output_path)
        print(f"[fix_d] Submission written → {output_path}", flush=True)
    else:
        output_path = Path(args.output) if args.output else Path("fix_d_dev_predictions.json")
        save_predictions(predictions, output_path)
        print(f"[fix_d] Dev predictions saved → {output_path}", flush=True)
        metrics = compute_metrics(documents, predictions, list(schema.allowed_pairs.keys()))
        print("\n[fix_d] === Dev metrics ===", flush=True)
        print(
            f"  Micro F1: {metrics['micro']['f1']:.3f} | "
            f"P: {metrics['micro']['precision']:.3f} | "
            f"R: {metrics['micro']['recall']:.3f}",
            flush=True,
        )
        print(f"  Macro F1: {metrics['macro']['f1']:.3f}", flush=True)
        print("  Per-relation:")
        for rel, m in metrics["per_relation"].items():
            print(
                f"    {rel:<22} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | "
                f"tp={m['tp']} fp={m['fp']} fn={m['fn']}"
            )

    per_relation: dict[str, int] = {}
    for edges in predictions.values():
        for edge in edges:
            per_relation[edge.predicate] = per_relation.get(edge.predicate, 0) + 1
    print("\n[fix_d] Per-relation prediction counts:")
    for rel, count in sorted(per_relation.items()):
        print(f"  {rel:<22} {count}")


if __name__ == "__main__":
    main()
