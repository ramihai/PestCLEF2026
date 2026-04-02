from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .schema import CanonicalEntity, Document, RelationEdge


def compute_metrics(
    gold_documents: Sequence[Document],
    predicted_edges_by_doc: Dict[str, List[RelationEdge]],
    relation_labels: Sequence[str],
) -> Dict[str, object]:
    per_relation = {}
    micro_tp = micro_fp = micro_fn = 0
    macro_values = []
    for relation in relation_labels:
        tp = fp = fn = 0
        for document in gold_documents:
            gold = {
                (edge.subject, edge.predicate, edge.object)
                for edge in document.gold_relation_edges
                if edge.predicate == relation
            }
            pred = {
                (edge.subject, edge.predicate, edge.object)
                for edge in predicted_edges_by_doc.get(document.doc_id, [])
                if edge.predicate == relation
            }
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_relation[relation] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        macro_values.append((precision, recall, f1))

    micro_precision, micro_recall, micro_f1 = precision_recall_f1(micro_tp, micro_fp, micro_fn)
    macro_precision = sum(item[0] for item in macro_values) / len(macro_values)
    macro_recall = sum(item[1] for item in macro_values) / len(macro_values)
    macro_f1 = sum(item[2] for item in macro_values) / len(macro_values)
    return {
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "per_relation": per_relation,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_error_report(
    documents: Sequence[Document],
    predicted_edges_by_doc: Dict[str, List[RelationEdge]],
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] | None = None,
) -> Dict[str, object]:
    false_positive_counter = Counter()
    false_negative_counter = Counter()
    examples = {"false_positives": [], "false_negatives": []}
    entity_errors = defaultdict(list)

    gold_entities_by_doc = {
        document.doc_id: {(entity.canonical_form, entity.entity_type) for entity in document.canonical_entities}
        for document in documents
    }
    if predicted_entities_by_doc:
        for document in documents:
            gold_entities = gold_entities_by_doc[document.doc_id]
            predicted_entities = {
                (entity.canonical_form, entity.entity_type)
                for entity in predicted_entities_by_doc.get(document.doc_id, [])
            }
            for item in sorted(predicted_entities - gold_entities):
                entity_errors["spurious_entities"].append({"doc_id": document.doc_id, "entity": item[0], "type": item[1]})
            for item in sorted(gold_entities - predicted_entities):
                entity_errors["missed_entities"].append({"doc_id": document.doc_id, "entity": item[0], "type": item[1]})

    for document in documents:
        gold = {(edge.subject, edge.predicate, edge.object) for edge in document.gold_relation_edges}
        pred = {(edge.subject, edge.predicate, edge.object) for edge in predicted_edges_by_doc.get(document.doc_id, [])}
        for item in sorted(pred - gold):
            false_positive_counter[item[1]] += 1
            if len(examples["false_positives"]) < 50:
                examples["false_positives"].append(
                    {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]}
                )
        for item in sorted(gold - pred):
            false_negative_counter[item[1]] += 1
            if len(examples["false_negatives"]) < 50:
                examples["false_negatives"].append(
                    {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]}
                )

    return {
        "relation_false_positives": dict(false_positive_counter),
        "relation_false_negatives": dict(false_negative_counter),
        "entity_detection_errors": dict(entity_errors),
        "examples": examples,
    }


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_predictions(predicted_edges_by_doc: Dict[str, List[RelationEdge]], path: Path) -> None:
    payload = {
        doc_id: [asdict(edge) for edge in edges]
        for doc_id, edges in predicted_edges_by_doc.items()
    }
    save_json(payload, path)
