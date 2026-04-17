from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .data import normalize_text
from .schema import CanonicalEntity, Document, Mention, RelationEdge


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


def compute_mention_metrics(
    gold_documents: Sequence[Document],
    predicted_mentions_by_doc: Dict[str, List[Mention]],
    entity_types: Sequence[str],
) -> Dict[str, object]:
    per_type = {}
    micro_tp = micro_fp = micro_fn = 0
    macro_values = []
    for entity_type in entity_types:
        tp = fp = fn = 0
        for document in gold_documents:
            gold = {
                (mention.start, mention.end, mention.entity_type)
                for mention in document.mentions
                if mention.entity_type == entity_type
            }
            predicted = {
                (mention.start, mention.end, mention.entity_type)
                for mention in predicted_mentions_by_doc.get(document.doc_id, [])
                if mention.entity_type == entity_type
            }
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_type[entity_type] = {
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
        "per_type": per_type,
    }


def compute_entity_metrics(
    gold_documents: Sequence[Document],
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]],
    entity_types: Sequence[str],
) -> Dict[str, object]:
    per_type = {}
    micro_tp = micro_fp = micro_fn = 0
    macro_values = []
    for entity_type in entity_types:
        tp = fp = fn = 0
        for document in gold_documents:
            gold_entities = [entity for entity in document.canonical_entities if entity.entity_type == entity_type]
            predicted_entities = [entity for entity in predicted_entities_by_doc.get(document.doc_id, []) if entity.entity_type == entity_type]
            matched_pairs = _match_entities(gold_entities, predicted_entities)
            tp += len(matched_pairs)
            fp += max(0, len(predicted_entities) - len(matched_pairs))
            fn += max(0, len(gold_entities) - len(matched_pairs))
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_type[entity_type] = {
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
        "per_type": per_type,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_pair_volume(
    candidate_pairs_by_doc: Dict[str, Sequence[Tuple[CanonicalEntity, CanonicalEntity]]],
    counts_by_doc: Dict[str, Dict[str, int]] | None = None,
) -> Dict[str, object]:
    per_doc = {}
    totals = Counter()
    for doc_id, pairs in candidate_pairs_by_doc.items():
        row = dict(counts_by_doc.get(doc_id, {}) if counts_by_doc else {})
        row["candidate_pairs"] = len(pairs)
        per_doc[doc_id] = row
        totals.update(row)
    document_count = max(len(per_doc), 1)
    summary = {
        key: {
            "total": value,
            "average": value / document_count,
        }
        for key, value in totals.items()
    }
    return {
        "summary": summary,
        "per_doc": per_doc,
    }


def compute_candidate_recall(
    documents: Sequence[Document],
    predicted_mentions_by_doc: Dict[str, List[Mention]],
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]],
    candidate_pairs_by_doc: Dict[str, Sequence[Tuple[CanonicalEntity, CanonicalEntity]]],
    predicted_edges_by_doc: Dict[str, List[RelationEdge]] | None = None,
) -> Dict[str, object]:
    per_relation = defaultdict(lambda: Counter(total=0, bad_span=0, bad_clustering_or_canonicalization=0, candidate_missed_by_pruning=0, valid_candidate=0, predicted=0))
    stage_counts = Counter(total=0, bad_span=0, bad_clustering_or_canonicalization=0, candidate_missed_by_pruning=0, valid_candidate=0, predicted=0)
    sampled = defaultdict(list)

    for document in documents:
        predicted_mentions = predicted_mentions_by_doc.get(document.doc_id, [])
        predicted_entities = predicted_entities_by_doc.get(document.doc_id, [])
        candidate_pairs = {
            (subject.canonical_form, object_.canonical_form)
            for subject, object_ in candidate_pairs_by_doc.get(document.doc_id, [])
        }
        predicted_edges = {
            (edge.subject, edge.predicate, edge.object)
            for edge in (predicted_edges_by_doc or {}).get(document.doc_id, [])
        }
        gold_to_predicted = _build_gold_to_predicted_entity_map(document.canonical_entities, predicted_entities)
        gold_mentions_by_entity_id = {entity.entity_id: entity.mentions for entity in document.canonical_entities}

        for edge in document.gold_relation_edges:
            per_relation[edge.predicate]["total"] += 1
            stage_counts["total"] += 1
            subject_entity = next((entity for entity in document.canonical_entities if entity.canonical_form == edge.subject), None)
            object_entity = next((entity for entity in document.canonical_entities if entity.canonical_form == edge.object), None)
            if subject_entity is None or object_entity is None:
                continue
            predicted_subject = gold_to_predicted.get(subject_entity.entity_id)
            predicted_object = gold_to_predicted.get(object_entity.entity_id)
            if predicted_subject is None or predicted_object is None:
                bucket = "bad_clustering_or_canonicalization"
                if predicted_subject is None and not _has_matching_gold_mention(gold_mentions_by_entity_id[subject_entity.entity_id], predicted_mentions):
                    bucket = "bad_span"
                if predicted_object is None and not _has_matching_gold_mention(gold_mentions_by_entity_id[object_entity.entity_id], predicted_mentions):
                    bucket = "bad_span"
                per_relation[edge.predicate][bucket] += 1
                stage_counts[bucket] += 1
                _append_sample(
                    sampled[bucket],
                    {
                        "doc_id": document.doc_id,
                        "subject": edge.subject,
                        "predicate": edge.predicate,
                        "object": edge.object,
                    },
                )
                continue
            candidate_key = (predicted_subject.canonical_form, predicted_object.canonical_form)
            if candidate_key not in candidate_pairs:
                per_relation[edge.predicate]["candidate_missed_by_pruning"] += 1
                stage_counts["candidate_missed_by_pruning"] += 1
                _append_sample(
                    sampled["candidate_missed_by_pruning"],
                    {
                        "doc_id": document.doc_id,
                        "subject": edge.subject,
                        "predicate": edge.predicate,
                        "object": edge.object,
                    },
                )
                continue
            per_relation[edge.predicate]["valid_candidate"] += 1
            stage_counts["valid_candidate"] += 1
            if (predicted_subject.canonical_form, edge.predicate, predicted_object.canonical_form) in predicted_edges:
                per_relation[edge.predicate]["predicted"] += 1
                stage_counts["predicted"] += 1

    summary = {}
    for relation, counts in per_relation.items():
        total = max(1, int(counts["total"]))
        summary[relation] = {
            **dict(counts),
            "candidate_recall": counts["valid_candidate"] / total,
            "prediction_recall_after_candidates": (counts["predicted"] / counts["valid_candidate"]) if counts["valid_candidate"] else 0.0,
        }
    total_edges = max(1, int(stage_counts["total"]))
    return {
        "summary": {
            **dict(stage_counts),
            "candidate_recall": stage_counts["valid_candidate"] / total_edges,
            "prediction_recall_after_candidates": (stage_counts["predicted"] / stage_counts["valid_candidate"]) if stage_counts["valid_candidate"] else 0.0,
        },
        "per_relation": summary,
        "sampled_failures": dict(sampled),
    }


def build_error_report(
    documents: Sequence[Document],
    predicted_edges_by_doc: Dict[str, List[RelationEdge]],
    predicted_mentions_by_doc: Dict[str, List[Mention]] | None = None,
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] | None = None,
    candidate_pairs_by_doc: Dict[str, Sequence[Tuple[CanonicalEntity, CanonicalEntity]]] | None = None,
) -> Dict[str, object]:
    false_positive_counter = Counter()
    false_negative_counter = Counter()
    examples = {"false_positives": [], "false_negatives": []}
    entity_errors = defaultdict(list)
    failure_buckets = Counter()
    sampled_failure_examples = defaultdict(list)
    grouped_examples = {
        "false_positives_by_relation": defaultdict(list),
        "false_negatives_by_relation": defaultdict(list),
        "spurious_entities_by_type": defaultdict(list),
        "missed_entities_by_type": defaultdict(list),
    }

    if predicted_entities_by_doc:
        for document in documents:
            gold_entities = document.canonical_entities
            predicted_entities = predicted_entities_by_doc.get(document.doc_id, [])
            matched_pairs = _match_entities(gold_entities, predicted_entities)
            matched_gold_ids = {id(gold) for gold, _ in matched_pairs}
            matched_pred_ids = {id(predicted) for _, predicted in matched_pairs}
            for predicted in predicted_entities:
                if id(predicted) in matched_pred_ids:
                    continue
                payload = {"doc_id": document.doc_id, "entity": predicted.canonical_form, "type": predicted.entity_type}
                entity_errors["spurious_entities"].append(payload)
                _append_sample(grouped_examples["spurious_entities_by_type"][predicted.entity_type], payload)
            for gold in gold_entities:
                if id(gold) in matched_gold_ids:
                    continue
                payload = {"doc_id": document.doc_id, "entity": gold.canonical_form, "type": gold.entity_type}
                entity_errors["missed_entities"].append(payload)
                _append_sample(grouped_examples["missed_entities_by_type"][gold.entity_type], payload)

    for document in documents:
        gold = {(edge.subject, edge.predicate, edge.object) for edge in document.gold_relation_edges}
        pred = {(edge.subject, edge.predicate, edge.object) for edge in predicted_edges_by_doc.get(document.doc_id, [])}
        predicted_entities = predicted_entities_by_doc.get(document.doc_id, []) if predicted_entities_by_doc else []
        gold_to_predicted = _build_gold_to_predicted_entity_map(document.canonical_entities, predicted_entities) if predicted_entities_by_doc else {}
        predicted_to_gold = _build_predicted_to_gold_entity_map(document.canonical_entities, predicted_entities) if predicted_entities_by_doc else {}
        candidate_pairs = {
            (subject.canonical_form, object_.canonical_form)
            for subject, object_ in (candidate_pairs_by_doc or {}).get(document.doc_id, [])
        }
        predicted_mentions = predicted_mentions_by_doc.get(document.doc_id, []) if predicted_mentions_by_doc else []
        for item in sorted(pred - gold):
            false_positive_counter[item[1]] += 1
            if len(examples["false_positives"]) < 50:
                examples["false_positives"].append(
                    {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]}
                )
            _append_sample(
                grouped_examples["false_positives_by_relation"][item[1]],
                {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
            )
            if predicted_entities_by_doc:
                subject_gold = predicted_to_gold.get(item[0])
                object_gold = predicted_to_gold.get(item[2])
                if subject_gold is None or object_gold is None:
                    failure_buckets["bad_entity_fp"] += 1
                    _append_sample(
                        sampled_failure_examples["bad_entity_fp"],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )
                elif (subject_gold.canonical_form, item[1], object_gold.canonical_form) in gold:
                    failure_buckets["bad_clustering_or_canonicalization_fp"] += 1
                    _append_sample(
                        sampled_failure_examples["bad_clustering_or_canonicalization_fp"],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )
                elif candidate_pairs:
                    failure_buckets["classifier_fp_on_valid_candidate"] += 1
                    _append_sample(
                        sampled_failure_examples["classifier_fp_on_valid_candidate"],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )
        for item in sorted(gold - pred):
            false_negative_counter[item[1]] += 1
            if len(examples["false_negatives"]) < 50:
                examples["false_negatives"].append(
                    {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]}
                )
            _append_sample(
                grouped_examples["false_negatives_by_relation"][item[1]],
                {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
            )
            if predicted_entities_by_doc and predicted_mentions_by_doc is not None:
                subject_entity = next((entity for entity in document.canonical_entities if entity.canonical_form == item[0]), None)
                object_entity = next((entity for entity in document.canonical_entities if entity.canonical_form == item[2]), None)
                if subject_entity is None or object_entity is None:
                    continue
                predicted_subject = gold_to_predicted.get(subject_entity.entity_id)
                predicted_object = gold_to_predicted.get(object_entity.entity_id)
                if predicted_subject is None or predicted_object is None:
                    bucket = "bad_clustering_or_canonicalization"
                    if predicted_subject is None and not _has_matching_gold_mention(subject_entity.mentions, predicted_mentions):
                        bucket = "bad_span"
                    if predicted_object is None and not _has_matching_gold_mention(object_entity.mentions, predicted_mentions):
                        bucket = "bad_span"
                    failure_buckets[bucket] += 1
                    _append_sample(
                        sampled_failure_examples[bucket],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )
                elif candidate_pairs and (predicted_subject.canonical_form, predicted_object.canonical_form) not in candidate_pairs:
                    failure_buckets["candidate_missed_by_pruning"] += 1
                    _append_sample(
                        sampled_failure_examples["candidate_missed_by_pruning"],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )
                else:
                    failure_buckets["classifier_fn_on_valid_candidate"] += 1
                    _append_sample(
                        sampled_failure_examples["classifier_fn_on_valid_candidate"],
                        {"doc_id": document.doc_id, "subject": item[0], "predicate": item[1], "object": item[2]},
                    )

    return {
        "relation_false_positives": dict(false_positive_counter),
        "relation_false_negatives": dict(false_negative_counter),
        "entity_detection_errors": dict(entity_errors),
        "examples": examples,
        "failure_buckets": dict(failure_buckets),
        "sampled_failure_examples": {key: value for key, value in sampled_failure_examples.items()},
        "grouped_examples": {
            key: {group: values for group, values in value.items()}
            for key, value in grouped_examples.items()
        },
    }


def _append_sample(bucket: List[Dict[str, object]], payload: Dict[str, object], limit: int = 25) -> None:
    if len(bucket) < limit:
        bucket.append(payload)


def _entity_aliases(entity: CanonicalEntity) -> set[str]:
    aliases = {normalize_text(entity.canonical_form)}
    aliases.update(normalize_text(alias) for alias in entity.alias_forms if str(alias).strip())
    return {alias for alias in aliases if alias}


def _entity_match_score(gold: CanonicalEntity, predicted: CanonicalEntity) -> int:
    if gold.entity_type != predicted.entity_type:
        return 0
    overlap = _entity_aliases(gold) & _entity_aliases(predicted)
    if overlap:
        return 100 + max(len(item) for item in overlap)
    return 0


def _match_entities(
    gold_entities: Sequence[CanonicalEntity],
    predicted_entities: Sequence[CanonicalEntity],
) -> List[Tuple[CanonicalEntity, CanonicalEntity]]:
    scored_pairs: List[Tuple[int, int, int]] = []
    for gold_index, gold in enumerate(gold_entities):
        for predicted_index, predicted in enumerate(predicted_entities):
            score = _entity_match_score(gold, predicted)
            if score > 0:
                scored_pairs.append((score, gold_index, predicted_index))
    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gold = set()
    matched_predicted = set()
    matches: List[Tuple[CanonicalEntity, CanonicalEntity]] = []
    for _score, gold_index, predicted_index in scored_pairs:
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
        matches.append((gold_entities[gold_index], predicted_entities[predicted_index]))
    return matches


def _build_gold_to_predicted_entity_map(
    gold_entities: Sequence[CanonicalEntity],
    predicted_entities: Sequence[CanonicalEntity],
) -> Dict[str, CanonicalEntity]:
    return {
        gold.entity_id: predicted
        for gold, predicted in _match_entities(gold_entities, predicted_entities)
    }


def _build_predicted_to_gold_entity_map(
    gold_entities: Sequence[CanonicalEntity],
    predicted_entities: Sequence[CanonicalEntity],
) -> Dict[str, CanonicalEntity]:
    mapping = {}
    for gold, predicted in _match_entities(gold_entities, predicted_entities):
        mapping[predicted.canonical_form] = gold
    return mapping


def _has_matching_gold_mention(gold_mentions: Sequence[Mention], predicted_mentions: Sequence[Mention]) -> bool:
    gold_offsets = {(mention.start, mention.end, mention.entity_type) for mention in gold_mentions}
    predicted_offsets = {(mention.start, mention.end, mention.entity_type) for mention in predicted_mentions}
    return bool(gold_offsets & predicted_offsets)


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_predictions(predicted_edges_by_doc: Dict[str, List[RelationEdge]], path: Path) -> None:
    payload = {
        doc_id: [asdict(edge) for edge in edges]
        for doc_id, edges in predicted_edges_by_doc.items()
    }
    save_json(payload, path)
