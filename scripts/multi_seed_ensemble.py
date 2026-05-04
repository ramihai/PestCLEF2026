"""Train v21 across multiple seeds, average sigmoid logits, recalibrate, submit.

This is the v21d lever from the plan in
``.claude/plans/we-are-working-on-zany-micali.md``. It addresses the
high-variance behaviour observed in ``modernbert_sweep_summary.json`` (same
config, ±0.03 dev F1 just from initialisation).

What it does
------------

1. Iterates over ``--seeds`` (default ``13,42,1337,7,2025``).
2. For each seed, runs the full ModernBERT staged pipeline twice:

   * ``run_modernbert_dev_evaluation`` — dumps dev relation logits to
     ``artifacts/<artifacts_root>_seed<N>/dev_relation_logits.json`` along
     with the usual metrics/error reports.
   * ``run_modernbert_test_submission`` — dumps test relation logits and a
     per-seed submission CSV.

3. Aggregates dev and test logit records across seeds. For each
   ``(doc_id, subject, object, label)`` key, it averages the available
   sigmoid scores (treats keys missing from a seed as that seed having no
   prediction for that pair, since predicted entity sets can differ).
4. Recalibrates per-relation thresholds on the averaged dev logits against
   gold dev relations using a coarse grid search optimising micro F1.
5. Applies the recalibrated thresholds to the averaged test logits and
   writes the ensemble submission as
   ``submission_modernbert_e2e_v21_<N>seed.csv`` plus a
   ``v21_ensemble_summary.json`` with the seed-by-seed and ensemble metrics.

The script is intentionally CSV/JSON-only at the aggregation step so the
expensive transformer training is the only heavy work — the rest runs in
seconds and can be re-run with different seed subsets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pestclef.config import ExperimentConfig  # noqa: E402
from pestclef.data import load_documents  # noqa: E402
from pestclef.evaluation import compute_metrics  # noqa: E402
from pestclef.pipeline import (  # noqa: E402
    run_modernbert_dev_evaluation,
    run_modernbert_test_submission,
)
from pestclef.schema import RelationEdge  # noqa: E402
from pestclef.submission import write_submission  # noqa: E402


@dataclass
class EnsembleConfig:
    seeds: List[int] = field(default_factory=lambda: [13, 42, 1337, 7, 2025])
    artifacts_root: Path = Path("artifacts/modernbert_e2e_v21d")
    encoder_name: str = "answerdotai/ModernBERT-base"
    mention_encoder_name: str = "michiyasunaga/BioLinkBERT-base"
    mention_max_seq_length: int = 512
    mention_doc_stride: int = 128
    lexicon_external_path: str = "data/lexicons/eppo_pest_plant_disease.json"
    lexicon_external_confidence: float = 0.70
    mention_date_regex_enabled: bool = True
    # Defaults aligned with Path 2 (Fix C): full predicted-entity training,
    # higher positive oversampling, and the [0.35, 0.75] threshold band that
    # v21b validated. Override via CLI flags below.
    epochs: int = 8
    relation_oversampling_ratio: float = 0.20
    relation_predicted_entity_mix_ratio: float = 0.5  # v21d-validated; v21c proved 1.0 collapses the classifier
    relation_hard_negative_ratio: float = 1.0
    relation_context_sentence_radius: int = 2
    relation_threshold_search_min: float = 0.35
    relation_threshold_search_max: float = 0.75
    relation_use_focal_loss: bool = True
    relation_focal_gamma: float = 2.0
    batch_size: int = 4
    skip_train: bool = False
    skip_test: bool = False
    output_csv: Path = Path("submission_modernbert_e2e_v21d_5seed.csv")


def _materialise_seed_config(base: EnsembleConfig, seed: int) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model_name = "modernbert_staged"
    config.encoder_name = base.encoder_name
    config.mention_encoder_name = base.mention_encoder_name
    config.mention_max_seq_length = base.mention_max_seq_length
    config.mention_doc_stride = base.mention_doc_stride
    config.lexicon_external_path = base.lexicon_external_path
    config.lexicon_external_confidence = base.lexicon_external_confidence
    config.mention_date_regex_enabled = base.mention_date_regex_enabled
    config.epochs = base.epochs
    config.batch_size = base.batch_size
    config.train_batch_size = max(1, min(base.batch_size, 8))
    config.eval_batch_size = max(1, min(base.batch_size, 16))
    config.relation_oversampling_ratio = base.relation_oversampling_ratio
    config.relation_predicted_entity_mix_ratio = base.relation_predicted_entity_mix_ratio
    config.relation_hard_negative_ratio = base.relation_hard_negative_ratio
    config.relation_context_sentence_radius = base.relation_context_sentence_radius
    config.relation_threshold_search_min = base.relation_threshold_search_min
    config.relation_threshold_search_max = base.relation_threshold_search_max
    config.relation_use_focal_loss = base.relation_use_focal_loss
    config.relation_focal_gamma = base.relation_focal_gamma
    # v14 per-class thresholds (calibration overwrites these per seed)
    config.relation_thresholds = {
        "Located_in": 0.55,
        "Found_on": 0.52,
        "Occurs_on": 0.48,
        "Affects": 0.48,
        "Causes": 0.48,
        "Dispersed_by": 0.48,
        "Transmits": 0.48,
    }
    config.random_seed = seed
    config.save_relation_logits = True
    config.artifacts_dir = base.artifacts_root.parent / f"{base.artifacts_root.name}_seed{seed}"
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return config


def _load_logits(path: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    if not path.exists():
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("records", [])), list(raw.get("labels", []))


def _aggregate_logits(
    records_per_seed: Sequence[List[Dict[str, object]]],
    labels: Sequence[str],
    min_seed_count: int = 1,
) -> Tuple[Dict[Tuple[str, str, str, str, str], Dict[str, float]], Dict[str, int]]:
    """Average sigmoid scores per (doc, subj_type, subj, obj_type, obj) key.

    Returns ``({key: {label: averaged_score}}, stats_dict)``. Scores are averaged
    only over seeds that produced that key.

    With ``min_seed_count > 1``, drops keys that didn't appear in at least that
    many seeds. This is critical when per-seed candidate sets diverge: in the
    v21d 5-seed run only 18% of test pairs appeared in all 5 seeds, and 47%
    appeared in only 1 seed — the unfiltered average treats those single-seed
    predictions as if they were ensemble predictions, letting their noise
    flow through.
    """
    accumulator: Dict[Tuple[str, str, str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {label: [] for label in labels}
    )
    seed_count: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    for records in records_per_seed:
        seen_keys_in_seed: set = set()
        for row in records:
            key = (
                str(row["doc_id"]),
                str(row.get("subject_type", "")),
                str(row["subject"]),
                str(row.get("object_type", "")),
                str(row["object"]),
            )
            scores = row.get("scores", {})
            if not isinstance(scores, dict):
                continue
            if key not in seen_keys_in_seed:
                seen_keys_in_seed.add(key)
                seed_count[key] += 1
            for label in labels:
                if label in scores:
                    accumulator[key][label].append(float(scores[label]))

    total_keys = len(accumulator)
    kept = 0
    dropped = 0
    averaged: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}
    seed_count_hist: Dict[int, int] = defaultdict(int)
    for key, label_scores in accumulator.items():
        n = seed_count[key]
        seed_count_hist[n] += 1
        if n < min_seed_count:
            dropped += 1
            continue
        kept += 1
        averaged[key] = {
            label: (sum(values) / len(values)) if values else 0.0
            for label, values in label_scores.items()
        }
    stats = {
        "total_keys": total_keys,
        "kept": kept,
        "dropped": dropped,
        "min_seed_count": min_seed_count,
        **{f"in_{n}_seeds": int(c) for n, c in sorted(seed_count_hist.items())},
    }
    return averaged, stats


def _edges_from_aggregated(
    aggregated: Dict[Tuple[str, str, str, str, str], Dict[str, float]],
    thresholds: Dict[str, float],
) -> Dict[str, List[RelationEdge]]:
    edges_by_doc: Dict[str, List[RelationEdge]] = defaultdict(list)
    for (doc_id, _subj_type, subject, _obj_type, obj), scores in aggregated.items():
        for label, score in scores.items():
            if score >= float(thresholds.get(label, 0.5)):
                edges_by_doc[doc_id].append(
                    RelationEdge(subject=subject, predicate=label, object=obj)
                )
    deduped: Dict[str, List[RelationEdge]] = {}
    for doc_id, edges in edges_by_doc.items():
        seen: set = set()
        kept: List[RelationEdge] = []
        for edge in edges:
            key = (edge.subject, edge.predicate, edge.object)
            if key in seen:
                continue
            seen.add(key)
            kept.append(edge)
        deduped[doc_id] = kept
    return deduped


def _calibrate_thresholds(
    aggregated_dev: Dict[Tuple[str, str, str, str, str], Dict[str, float]],
    dev_documents,
    labels: Sequence[str],
    seed_thresholds: Dict[str, float],
    grid_min: float = 0.35,
    grid_max: float = 0.75,
    grid_step: float = 0.025,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Pick per-label thresholds maximising overall micro F1 on aggregated dev.

    Falls back to the median of the seed-level v14 thresholds when a label has
    fewer than 5 gold positives in dev (avoids overfitting on tiny support).
    """
    grid = []
    value = grid_min
    while value <= grid_max + 1e-9:
        grid.append(round(value, 6))
        value += grid_step
    chosen: Dict[str, float] = {}
    label_diagnostics: Dict[str, object] = {}
    label_set = set(labels)
    label_pos_counts: Dict[str, int] = {label: 0 for label in labels}
    for document in dev_documents:
        for edge in document.gold_relation_edges:
            if edge.predicate in label_set:
                label_pos_counts[edge.predicate] += 1

    for label in labels:
        if label_pos_counts[label] < 5:
            chosen[label] = float(seed_thresholds.get(label, 0.5))
            label_diagnostics[label] = {"strategy": "fallback_seed_threshold", "support": label_pos_counts[label]}
            continue
        best_threshold = float(seed_thresholds.get(label, 0.5))
        best_f1 = -1.0
        for threshold in grid:
            singleton_thresholds = {**chosen, label: threshold}
            for other in labels:
                singleton_thresholds.setdefault(other, float(seed_thresholds.get(other, 0.5)))
            edges_by_doc = _edges_from_aggregated(aggregated_dev, singleton_thresholds)
            metrics = compute_metrics(dev_documents, edges_by_doc, list(labels))
            f1 = float(metrics.get("per_relation", {}).get(label, {}).get("f1", 0.0))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        chosen[label] = best_threshold
        label_diagnostics[label] = {
            "strategy": "grid_search",
            "support": label_pos_counts[label],
            "best_f1": best_f1,
            "threshold": best_threshold,
        }
    return chosen, label_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="13,42,1337,7,2025")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/modernbert_e2e_v21d"))
    parser.add_argument("--mention-encoder", type=str, default="michiyasunaga/BioLinkBERT-base")
    parser.add_argument("--lexicon-path", type=str, default="data/lexicons/eppo_pest_plant_disease.json")
    parser.add_argument("--no-date-regex", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mix-ratio", type=float, default=0.5,
                        help="relation_predicted_entity_mix_ratio (default 0.5, v21d-validated; 1.0 collapsed the classifier in v21c)")
    parser.add_argument("--oversample", type=float, default=0.20,
                        help="relation_oversampling_ratio (default 0.20)")
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.75)
    parser.add_argument(
        "--no-focal", action="store_true",
        help="Disable focal loss for the relation classifier (default: enabled). v21d showed focal helps; only disable for ablation.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma when --no-focal is not set (default: 2.0).",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip per-seed training; only aggregate existing logits")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test inference; only run dev evaluation per seed")
    parser.add_argument(
        "--min-seed-count", type=int, default=1,
        help=(
            "Drop pairs that didn't appear in at least this many seeds before "
            "averaging logits (default: 1 = no filter). v21d 5-seed showed only "
            "18%% of test pairs appeared in all 5 seeds and 47%% appeared in "
            "only 1 seed — try 3 to filter mention-set noise."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("submission_modernbert_e2e_v21d_5seed.csv"))
    args = parser.parse_args()

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    base = EnsembleConfig(
        seeds=seeds,
        artifacts_root=args.artifacts_root,
        mention_encoder_name=args.mention_encoder,
        lexicon_external_path=args.lexicon_path,
        mention_date_regex_enabled=not args.no_date_regex,
        epochs=args.epochs,
        batch_size=args.batch_size,
        relation_predicted_entity_mix_ratio=args.mix_ratio,
        relation_oversampling_ratio=args.oversample,
        relation_threshold_search_min=args.threshold_min,
        relation_threshold_search_max=args.threshold_max,
        relation_use_focal_loss=not args.no_focal,
        relation_focal_gamma=args.focal_gamma,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        output_csv=args.output,
    )

    per_seed_metrics: Dict[int, Dict[str, object]] = {}
    seed_configs: Dict[int, ExperimentConfig] = {}

    for seed in seeds:
        seed_config = _materialise_seed_config(base, seed)
        seed_configs[seed] = seed_config
        if not base.skip_train:
            print(f"[seed={seed}] running dev evaluation -> {seed_config.artifacts_dir}")
            dev_result = run_modernbert_dev_evaluation(seed_config)
            per_seed_metrics[seed] = {"dev": dev_result.get("metrics")}
        if not base.skip_test:
            print(f"[seed={seed}] running test submission")
            run_modernbert_test_submission(seed_config, train_on_dev=False)

    # Aggregate dev logits
    dev_records_per_seed: List[List[Dict[str, object]]] = []
    test_records_per_seed: List[List[Dict[str, object]]] = []
    label_set: List[str] = []
    for seed in seeds:
        seed_config = seed_configs[seed]
        dev_records, dev_labels = _load_logits(seed_config.artifacts_dir / "dev_relation_logits.json")
        test_records, test_labels = _load_logits(seed_config.artifacts_dir / "test_relation_logits.json")
        if dev_labels and not label_set:
            label_set = list(dev_labels)
        elif test_labels and not label_set:
            label_set = list(test_labels)
        dev_records_per_seed.append(dev_records)
        test_records_per_seed.append(test_records)

    if not label_set:
        raise SystemExit("No relation logits were found. Run without --skip-train at least once.")

    print(f"\n[ensemble] Aggregating logits with min_seed_count={args.min_seed_count}", flush=True)
    aggregated_dev, dev_stats = _aggregate_logits(dev_records_per_seed, label_set, args.min_seed_count)
    aggregated_test, test_stats = _aggregate_logits(test_records_per_seed, label_set, args.min_seed_count)
    print(f"[ensemble]   dev: kept {dev_stats['kept']}/{dev_stats['total_keys']} keys, dropped {dev_stats['dropped']}")
    print(f"[ensemble]   test: kept {test_stats['kept']}/{test_stats['total_keys']} keys, dropped {test_stats['dropped']}")
    dev_hist = {k: v for k, v in dev_stats.items() if k.startswith("in_")}
    test_hist = {k: v for k, v in test_stats.items() if k.startswith("in_")}
    print(f"[ensemble]   dev seed-count histogram: {dev_hist}")
    print(f"[ensemble]   test seed-count histogram: {test_hist}")

    # Recalibrate thresholds in the same band the per-seed runs used
    dev_documents = load_documents("dev", ExperimentConfig())
    seed_thresholds = ExperimentConfig().relation_thresholds
    thresholds, threshold_diag = _calibrate_thresholds(
        aggregated_dev, dev_documents, label_set, seed_thresholds=seed_thresholds,
        grid_min=base.relation_threshold_search_min,
        grid_max=base.relation_threshold_search_max,
    )

    ensemble_dev_edges = _edges_from_aggregated(aggregated_dev, thresholds)
    ensemble_dev_metrics = compute_metrics(dev_documents, ensemble_dev_edges, list(label_set))

    # Apply to test
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
        json.dumps(
            {
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
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote ensemble summary -> {summary_path}")
    print(f"Wrote ensemble submission -> {base.output_csv}")


if __name__ == "__main__":
    main()
