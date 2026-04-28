from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from .config import ExperimentConfig
from .data import deduplicate_edges, export_documents, load_documents
from .evaluation import (
    build_error_report,
    compute_candidate_recall,
    compute_entity_metrics,
    compute_mention_metrics,
    compute_metrics,
    compute_pair_volume,
    save_json,
    save_predictions,
)
from .features import RelationSchema, enumerate_candidate_entity_pairs, generate_relation_examples, should_consider_pair
from .mention_detection import MentionLexicon, predict_canonical_entities
from .model import RelationModel, train_relation_model
from .schema import CanonicalEntity, Document, RelationEdge
from .submission import write_submission


ENTITY_TYPES = ["Disease", "Date", "Dissemination_pathway", "Location", "Pest", "Plant", "Vector"]


def _predict_edges_with_optional_logits(
    relation_model,
    examples: Sequence[Dict[str, object]],
    entity_pairs: Sequence[tuple],
    schema: RelationSchema,
    doc_id: str,
    score_records: List[Dict[str, object]] | None,
) -> List[RelationEdge]:
    """Apply per-label thresholds locally so we can also dump raw scores.

    Returns the edges that survive the schema validity check. If
    ``score_records`` is provided, appends one row per (doc, subject, object)
    pair containing the raw sigmoid scores for every label.
    """
    if not examples:
        return []
    scores = relation_model.predict_scores(examples)
    edges: List[RelationEdge] = []
    labels = relation_model.labels
    thresholds = relation_model.thresholds
    for row_index, (subject, obj) in enumerate(entity_pairs):
        row = scores[row_index]
        per_label_scores = {label: float(row[col]) for col, label in enumerate(labels)}
        if score_records is not None:
            score_records.append(
                {
                    "doc_id": doc_id,
                    "subject": subject.canonical_form,
                    "subject_type": subject.entity_type,
                    "object": obj.canonical_form,
                    "object_type": obj.entity_type,
                    "scores": per_label_scores,
                }
            )
        for label, score in per_label_scores.items():
            if score < float(thresholds.get(label, 0.5)):
                continue
            if not schema.is_valid_pair(label, subject.entity_type, obj.entity_type):
                continue
            edges.append(RelationEdge(subject=subject.canonical_form, predicate=label, object=obj.canonical_form))
    return edges


def _collect_predicted_entity_state(
    documents: Sequence[Document],
    mention_detector,
    mention_lexicon: MentionLexicon | None,
):
    from .modernbert import build_canonical_entities_from_mentions, build_hybrid_span_candidates

    neural_predicted_spans_by_doc = {}
    lexicon_predicted_spans_by_doc = {}
    blended_predicted_spans_by_doc = {}
    predicted_mentions_by_doc = {}
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] = {}
    counts_by_doc: Dict[str, Dict[str, int]] = {}
    for document in documents:
        neural_spans, lexicon_spans, blended_spans = build_hybrid_span_candidates(document, mention_detector, mention_lexicon)
        predicted_mentions = mention_detector.predict_mentions_from_spans(blended_spans, document)
        predicted_entities = build_canonical_entities_from_mentions(predicted_mentions, config=mention_detector.config)
        neural_predicted_spans_by_doc[document.doc_id] = neural_spans
        lexicon_predicted_spans_by_doc[document.doc_id] = lexicon_spans
        blended_predicted_spans_by_doc[document.doc_id] = blended_spans
        predicted_mentions_by_doc[document.doc_id] = predicted_mentions
        predicted_entities_by_doc[document.doc_id] = predicted_entities
        counts_by_doc[document.doc_id] = {
            "raw_neural_spans": len(neural_spans),
            "raw_lexicon_spans": len(lexicon_spans),
            "blended_spans": len(blended_spans),
            "final_mentions": len(predicted_mentions),
            "final_entities": len(predicted_entities),
        }
    return (
        neural_predicted_spans_by_doc,
        lexicon_predicted_spans_by_doc,
        blended_predicted_spans_by_doc,
        predicted_mentions_by_doc,
        predicted_entities_by_doc,
        counts_by_doc,
    )


def _save_model_artifacts_safely(
    mention_detector,
    relation_model,
    artifacts_dir: Path,
) -> Dict[str, str]:
    warnings: Dict[str, str] = {}
    mention_model_dir = artifacts_dir / "mention_model"
    relation_model_dir = artifacts_dir / "relation_model"
    try:
        mention_detector.save(mention_model_dir)
    except Exception as exc:  # pragma: no cover - defensive path for long real runs
        warnings["mention_model_save"] = str(exc)
    try:
        relation_model.save(relation_model_dir)
    except Exception as exc:  # pragma: no cover - defensive path for long real runs
        warnings["relation_model_save"] = str(exc)
    return warnings


def train_gold_entity_baseline(config: ExperimentConfig) -> Dict[str, object]:
    if config.model_name == "modernbert_staged":
        return train_gold_entity_modernbert_baseline(config)
    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    schema = RelationSchema.from_documents(train_documents)

    train_examples = generate_relation_examples(train_documents, schema, config)
    model = train_relation_model(train_examples, schema, config)
    model_suffix = "joblib" if config.model_name == "sklearn" else "json"
    model_path = config.artifacts_dir / f"baseline_model_{config.model_name}.{model_suffix}"
    model.save(model_path)
    save_json({"schema": schema.to_serializable()}, config.artifacts_dir / "relation_schema.json")
    save_json({"config": asdict(config)}, config.artifacts_dir / "experiment_config.json")
    save_json({"train_documents": export_documents(train_documents[:3])}, config.artifacts_dir / "data_preview.json")
    dev_predictions = predict_with_gold_entities(dev_documents, schema, model, config)
    metrics = compute_metrics(dev_documents, dev_predictions, config.relation_labels)
    error_report = build_error_report(dev_documents, dev_predictions)
    save_predictions(dev_predictions, config.artifacts_dir / "dev_gold_entity_predictions.json")
    save_json(metrics, config.artifacts_dir / "dev_gold_entity_metrics.json")
    save_json(error_report, config.artifacts_dir / "dev_gold_entity_error_report.json")
    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "error_report": error_report,
    }


def predict_with_gold_entities(
    documents: Sequence[Document],
    schema: RelationSchema,
    model: RelationModel,
    config: ExperimentConfig,
) -> Dict[str, List[RelationEdge]]:
    predictions: Dict[str, List[RelationEdge]] = {}
    for document in documents:
        feature_rows = []
        pairs = []
        entities = sorted(document.canonical_entities, key=lambda entity: entity.earliest_start)
        for subject in entities:
            for obj in entities:
                if subject.entity_id == obj.entity_id:
                    continue
                if not should_consider_pair(subject, obj, schema, config):
                    continue
                from .features import extract_pair_features  # local import to keep module dependency simple

                feature_rows.append(extract_pair_features(document, subject, obj))
                pairs.append((subject.canonical_form, subject.entity_type, obj.canonical_form, obj.entity_type))
        if not feature_rows:
            predictions[document.doc_id] = []
            continue
        predicted_labels = model.predict_labels(feature_rows)
        edges = []
        for label_row, pair in zip(predicted_labels, pairs):
            subject_name, subject_type, object_name, object_type = pair
            for label in label_row:
                if not schema.is_valid_pair(label, subject_type, object_type):
                    continue
                edges.append(RelationEdge(subject=subject_name, predicate=label, object=object_name))
        predictions[document.doc_id] = deduplicate_edges(edges)
    return predictions


def run_dev_evaluation(config: ExperimentConfig) -> Dict[str, object]:
    if config.model_name == "modernbert_staged":
        return run_modernbert_dev_evaluation(config)
    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    schema = RelationSchema.from_documents(train_documents)
    train_examples = generate_relation_examples(train_documents, schema, config)
    model = train_relation_model(train_examples, schema, config)
    lexicon = MentionLexicon.from_documents(train_documents, config)
    predicted_edges_by_doc: Dict[str, List[RelationEdge]] = {}
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] = {}
    for document in dev_documents:
        predicted_entities = predict_canonical_entities(document, lexicon)
        predicted_entities_by_doc[document.doc_id] = predicted_entities
        predicted_edges_by_doc[document.doc_id] = predict_document_edges(document, predicted_entities, schema, model, config)
    metrics = compute_metrics(dev_documents, predicted_edges_by_doc, config.relation_labels)
    error_report = build_error_report(dev_documents, predicted_edges_by_doc, predicted_entities_by_doc)
    save_predictions(predicted_edges_by_doc, config.artifacts_dir / "dev_end_to_end_predictions.json")
    save_json(metrics, config.artifacts_dir / "dev_end_to_end_metrics.json")
    save_json(error_report, config.artifacts_dir / "dev_end_to_end_error_report.json")
    lexicon_path = config.artifacts_dir / "mention_lexicon.json"
    lexicon_path.write_text(json.dumps({"alias_to_type": lexicon.alias_to_type}, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "error_report": error_report,
    }


def predict_document_edges(
    document: Document,
    canonical_entities: Sequence[CanonicalEntity],
    schema: RelationSchema,
    model: RelationModel,
    config: ExperimentConfig,
) -> List[RelationEdge]:
    from .features import extract_pair_features

    if not canonical_entities:
        return []
    feature_rows = []
    pairs = []
    entities = sorted(canonical_entities, key=lambda entity: entity.earliest_start)
    for subject in entities:
        for obj in entities:
            if subject.entity_id == obj.entity_id:
                continue
            if not should_consider_pair(subject, obj, schema, config):
                continue
            feature_rows.append(extract_pair_features(document, subject, obj))
            pairs.append((subject, obj))
    if not feature_rows:
        return []
    labels = model.predict_labels(feature_rows)
    edges = []
    for label_row, (subject, obj) in zip(labels, pairs):
        for label in label_row:
            if schema.is_valid_pair(label, subject.entity_type, obj.entity_type):
                edges.append(RelationEdge(subject=subject.canonical_form, predicate=label, object=obj.canonical_form))
    return deduplicate_edges(edges)


def run_test_submission(config: ExperimentConfig, train_on_dev: bool = False) -> Dict[str, object]:
    if config.model_name == "modernbert_staged":
        return run_modernbert_test_submission(config, train_on_dev=train_on_dev)
    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    if train_on_dev:
        train_documents = train_documents + load_documents("dev", config)
    schema = RelationSchema.from_documents(train_documents)
    train_examples = generate_relation_examples(train_documents, schema, config)
    model = train_relation_model(train_examples, schema, config)
    lexicon = MentionLexicon.from_documents(train_documents, config)
    test_documents = load_documents("test", config)
    predictions: Dict[str, List[RelationEdge]] = {}
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] = {}
    for document in test_documents:
        predicted_entities = predict_canonical_entities(document, lexicon)
        predicted_entities_by_doc[document.doc_id] = predicted_entities
        predictions[document.doc_id] = predict_document_edges(document, predicted_entities, schema, model, config)
    output_path = config.project_root / "submission.csv"
    write_submission(predictions, [document.doc_id for document in test_documents], output_path)
    save_predictions(predictions, config.artifacts_dir / "test_predictions.json")
    save_json(
        {
            "predicted_entities": {
                doc_id: [
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "canonical_form": entity.canonical_form,
                    }
                    for entity in entities
                ]
                for doc_id, entities in predicted_entities_by_doc.items()
            }
        },
        config.artifacts_dir / "test_predicted_entities.json",
    )
    return {"submission_path": str(output_path), "documents": len(test_documents)}


def train_gold_entity_modernbert_baseline(config: ExperimentConfig) -> Dict[str, object]:
    from .modernbert import predict_with_gold_entities_modernbert, train_modernbert_gold_relation_pipeline

    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    schema = RelationSchema.from_documents(train_documents)
    relation_model = train_modernbert_gold_relation_pipeline(train_documents, dev_documents, schema, config)

    relation_model_dir = config.artifacts_dir / "relation_model"
    relation_model.save(relation_model_dir)
    save_json({"schema": schema.to_serializable()}, config.artifacts_dir / "relation_schema.json")
    save_json({"config": asdict(config)}, config.artifacts_dir / "experiment_config.json")
    save_json({"train_documents": export_documents(train_documents[:3])}, config.artifacts_dir / "data_preview.json")
    dev_predictions = predict_with_gold_entities_modernbert(dev_documents, schema, relation_model, config)
    metrics = compute_metrics(dev_documents, dev_predictions, config.relation_labels)
    error_report = build_error_report(dev_documents, dev_predictions)
    save_predictions(dev_predictions, config.artifacts_dir / "dev_gold_entity_predictions.json")
    save_json(metrics, config.artifacts_dir / "dev_gold_entity_metrics.json")
    save_json(error_report, config.artifacts_dir / "dev_gold_entity_error_report.json")
    return {
        "model_path": str(relation_model_dir),
        "metrics": metrics,
        "error_report": error_report,
    }


def run_modernbert_dev_evaluation(config: ExperimentConfig) -> Dict[str, object]:
    from .modernbert import (
        build_relation_inference_examples,
        generate_relation_text_examples,
        mix_relation_examples,
        serialize_predicted_spans,
        train_modernbert_mention_detector,
        train_modernbert_relation_model,
    )

    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    dev_document_lookup = {document.doc_id: document for document in dev_documents}
    schema = RelationSchema.from_documents(train_documents)
    mention_lexicon = MentionLexicon.from_documents(train_documents, config) if config.mention_hybrid_lexicon else None

    mention_detector = train_modernbert_mention_detector(
        train_documents,
        config,
        validation_documents=dev_documents,
        relation_schema=schema,
    )
    (
        neural_predicted_spans_by_doc,
        lexicon_predicted_spans_by_doc,
        blended_predicted_spans_by_doc,
        predicted_mentions_by_doc,
        predicted_entities_by_doc,
        counts_by_doc,
    ) = _collect_predicted_entity_state(dev_documents, mention_detector, mention_lexicon)
    (
        _train_neural_predicted_spans_by_doc,
        _train_lexicon_predicted_spans_by_doc,
        _train_blended_predicted_spans_by_doc,
        _train_predicted_mentions_by_doc,
        train_predicted_entities_by_doc,
        _train_counts_by_doc,
    ) = _collect_predicted_entity_state(train_documents, mention_detector, mention_lexicon)

    train_examples = generate_relation_text_examples(train_documents, schema, config)
    if config.relation_train_with_predicted_entities:
        predicted_train_examples = generate_relation_text_examples(
            train_documents,
            schema,
            config,
            entities_override=train_predicted_entities_by_doc,
        )
        train_examples = mix_relation_examples(
            train_examples,
            predicted_train_examples,
            config.relation_predicted_entity_mix_ratio,
            config.random_seed,
        )
    dev_gold_examples = generate_relation_text_examples(dev_documents, schema, config)
    dev_predicted_examples = generate_relation_text_examples(
        dev_documents,
        schema,
        config,
        entities_override=predicted_entities_by_doc,
    )
    relation_calibration_examples = dev_predicted_examples if config.relation_calibrate_on_predicted_entities else dev_gold_examples
    relation_model = train_modernbert_relation_model(
        train_examples,
        config,
        calibration_examples=relation_calibration_examples,
        validation_examples=relation_calibration_examples,
    )

    predicted_edges_by_doc: Dict[str, List[RelationEdge]] = {}
    candidate_pairs_by_doc: Dict[str, List[tuple[CanonicalEntity, CanonicalEntity]]] = {}
    relation_score_records: List[Dict[str, object]] | None = (
        [] if getattr(config, "save_relation_logits", False) else None
    )
    for document in dev_documents:
        examples, entity_pairs = build_relation_inference_examples(
            document,
            predicted_entities_by_doc[document.doc_id],
            schema,
            config,
        )
        candidate_pairs_by_doc[document.doc_id] = entity_pairs
        counts_by_doc.setdefault(document.doc_id, {})["candidate_pairs"] = len(entity_pairs)
        edges = _predict_edges_with_optional_logits(
            relation_model, examples, entity_pairs, schema, document.doc_id, relation_score_records,
        )
        predicted_edges_by_doc[document.doc_id] = deduplicate_edges(edges)
    if relation_score_records is not None:
        save_json(
            {"records": relation_score_records, "labels": list(relation_model.labels)},
            config.artifacts_dir / "dev_relation_logits.json",
        )

    save_json({"schema": schema.to_serializable()}, config.artifacts_dir / "relation_schema.json")
    save_json({"config": asdict(config)}, config.artifacts_dir / "experiment_config.json")
    save_json({"train_documents": export_documents(train_documents[:3])}, config.artifacts_dir / "data_preview.json")

    metrics = compute_metrics(dev_documents, predicted_edges_by_doc, config.relation_labels)
    mention_metrics = compute_mention_metrics(
        dev_documents,
        predicted_mentions_by_doc,
        ENTITY_TYPES,
    )
    entity_metrics = compute_entity_metrics(dev_documents, predicted_entities_by_doc, ENTITY_TYPES)
    candidate_recall = compute_candidate_recall(
        dev_documents,
        predicted_mentions_by_doc,
        predicted_entities_by_doc,
        candidate_pairs_by_doc,
        predicted_edges_by_doc,
    )
    pair_volume = compute_pair_volume(candidate_pairs_by_doc, counts_by_doc)
    error_report = build_error_report(
        dev_documents,
        predicted_edges_by_doc,
        predicted_mentions_by_doc,
        predicted_entities_by_doc,
        candidate_pairs_by_doc,
    )
    save_predictions(predicted_edges_by_doc, config.artifacts_dir / "dev_end_to_end_predictions.json")
    save_json(metrics, config.artifacts_dir / "dev_end_to_end_metrics.json")
    save_json(mention_metrics, config.artifacts_dir / "dev_mention_metrics.json")
    save_json(entity_metrics, config.artifacts_dir / "dev_entity_metrics.json")
    save_json(candidate_recall, config.artifacts_dir / "dev_candidate_recall.json")
    save_json(pair_volume, config.artifacts_dir / "dev_pair_volume.json")
    save_json(error_report, config.artifacts_dir / "dev_end_to_end_error_report.json")
    save_json(
        {
            "pre_cleanup_predicted_mentions_neural": {
                doc_id: serialize_predicted_spans(neural_predicted_spans_by_doc.get(doc_id, []), dev_document_lookup[doc_id])
                for doc_id in neural_predicted_spans_by_doc
            },
            "pre_cleanup_predicted_mentions_lexicon": {
                doc_id: serialize_predicted_spans(lexicon_predicted_spans_by_doc.get(doc_id, []), dev_document_lookup[doc_id])
                for doc_id in lexicon_predicted_spans_by_doc
            },
            "pre_cleanup_predicted_mentions_blended": {
                doc_id: serialize_predicted_spans(blended_predicted_spans_by_doc.get(doc_id, []), dev_document_lookup[doc_id])
                for doc_id in blended_predicted_spans_by_doc
            },
            "mention_thresholds": mention_detector.mention_thresholds,
            "hybrid_lexicon_enabled": bool(mention_lexicon is not None),
        },
        config.artifacts_dir / "pre_cleanup_predicted_mentions.json",
    )
    save_json(
        {
            "predicted_mentions": {
                doc_id: [
                    {
                        "mention_id": mention.mention_id,
                        "entity_type": mention.entity_type,
                        "form": mention.form,
                        "offsets": mention.offsets,
                        "sentence_index": mention.sentence_index,
                        "layout_index": mention.layout_index,
                        "confidence": next(
                            (
                                span.confidence
                                for span in blended_predicted_spans_by_doc.get(doc_id, [])
                                if span.entity_type == mention.entity_type and span.start == mention.start and span.end == mention.end
                            ),
                            None,
                        ),
                    }
                    for mention in mentions
                ]
                for doc_id, mentions in predicted_mentions_by_doc.items()
            }
        },
        config.artifacts_dir / "dev_predicted_mentions.json",
    )
    save_json(
        {
            "predicted_entities": {
                doc_id: [
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "canonical_form": entity.canonical_form,
                        "alias_forms": sorted(entity.alias_forms),
                        "mention_count": len(entity.mentions),
                    }
                    for entity in entities
                ]
                for doc_id, entities in predicted_entities_by_doc.items()
            }
        },
        config.artifacts_dir / "dev_predicted_entities.json",
    )
    save_warnings = _save_model_artifacts_safely(mention_detector, relation_model, config.artifacts_dir)
    if save_warnings:
        save_json({"warnings": save_warnings}, config.artifacts_dir / "model_save_warnings.json")
    return {
        "metrics": metrics,
        "mention_metrics": mention_metrics,
        "entity_metrics": entity_metrics,
        "candidate_recall": candidate_recall,
        "error_report": error_report,
        "model_save_warnings": save_warnings,
    }


def run_modernbert_test_submission(config: ExperimentConfig, train_on_dev: bool = False) -> Dict[str, object]:
    from .modernbert import (
        build_relation_inference_examples,
        generate_relation_text_examples,
        mix_relation_examples,
        train_modernbert_mention_detector,
        train_modernbert_relation_model,
    )

    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    if train_on_dev:
        mention_train_documents = train_documents + dev_documents
        relation_train_documents = train_documents + dev_documents
        relation_calibration_documents: Sequence[Document] = train_documents
    else:
        mention_train_documents = train_documents
        relation_train_documents = train_documents
        relation_calibration_documents = dev_documents

    schema = RelationSchema.from_documents(relation_train_documents)
    mention_detector = train_modernbert_mention_detector(
        mention_train_documents,
        config,
        validation_documents=dev_documents if not train_on_dev else train_documents,
        relation_schema=schema,
    )
    mention_lexicon = MentionLexicon.from_documents(mention_train_documents, config) if config.mention_hybrid_lexicon else None
    relation_train_examples = generate_relation_text_examples(relation_train_documents, schema, config)
    if config.relation_train_with_predicted_entities:
        (
            _train_neural_predicted_spans_by_doc,
            _train_lexicon_predicted_spans_by_doc,
            _train_blended_predicted_spans_by_doc,
            _train_predicted_mentions_by_doc,
            relation_train_predicted_entities_by_doc,
            _train_counts_by_doc,
        ) = _collect_predicted_entity_state(relation_train_documents, mention_detector, mention_lexicon)
        predicted_relation_train_examples = generate_relation_text_examples(
            relation_train_documents,
            schema,
            config,
            entities_override=relation_train_predicted_entities_by_doc,
        )
        relation_train_examples = mix_relation_examples(
            relation_train_examples,
            predicted_relation_train_examples,
            config.relation_predicted_entity_mix_ratio,
            config.random_seed,
        )
    if config.relation_calibrate_on_predicted_entities:
        (
            _cal_neural_predicted_spans_by_doc,
            _cal_lexicon_predicted_spans_by_doc,
            _cal_blended_predicted_spans_by_doc,
            _cal_predicted_mentions_by_doc,
            relation_calibration_predicted_entities_by_doc,
            _cal_counts_by_doc,
        ) = _collect_predicted_entity_state(relation_calibration_documents, mention_detector, mention_lexicon)
        relation_calibration_examples = generate_relation_text_examples(
            relation_calibration_documents,
            schema,
            config,
            entities_override=relation_calibration_predicted_entities_by_doc,
        )
    else:
        relation_calibration_examples = generate_relation_text_examples(relation_calibration_documents, schema, config)
    relation_model = train_modernbert_relation_model(
        relation_train_examples,
        config,
        calibration_examples=relation_calibration_examples,
        validation_examples=relation_calibration_examples,
    )

    test_documents = load_documents("test", config)
    predictions: Dict[str, List[RelationEdge]] = {}
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] = {}
    candidate_pairs_by_doc: Dict[str, List[tuple[CanonicalEntity, CanonicalEntity]]] = {}
    pair_counts_by_doc: Dict[str, Dict[str, int]] = {}
    test_relation_score_records: List[Dict[str, object]] | None = (
        [] if getattr(config, "save_relation_logits", False) else None
    )
    for document in test_documents:
        (
            _test_neural_spans_by_doc,
            _test_lexicon_spans_by_doc,
            _test_blended_spans_by_doc,
            _test_mentions_by_doc,
            test_predicted_entities_by_doc,
            test_counts_by_doc,
        ) = _collect_predicted_entity_state([document], mention_detector, mention_lexicon)
        predicted_entities = test_predicted_entities_by_doc[document.doc_id]
        predicted_entities_by_doc[document.doc_id] = predicted_entities
        examples, entity_pairs = build_relation_inference_examples(
            document,
            predicted_entities,
            schema,
            config,
        )
        candidate_pairs_by_doc[document.doc_id] = entity_pairs
        pair_counts_by_doc[document.doc_id] = {
            **test_counts_by_doc[document.doc_id],
            "candidate_pairs": len(entity_pairs),
        }
        edges = _predict_edges_with_optional_logits(
            relation_model, examples, entity_pairs, schema, document.doc_id, test_relation_score_records,
        )
        predictions[document.doc_id] = deduplicate_edges(edges)
    if test_relation_score_records is not None:
        save_json(
            {"records": test_relation_score_records, "labels": list(relation_model.labels)},
            config.artifacts_dir / "test_relation_logits.json",
        )

    output_path = config.project_root / "submission.csv"
    write_submission(predictions, [document.doc_id for document in test_documents], output_path)
    save_predictions(predictions, config.artifacts_dir / "test_predictions.json")
    save_json(
        {
            "predicted_entities": {
                doc_id: [
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "canonical_form": entity.canonical_form,
                    }
                    for entity in entities
                ]
                for doc_id, entities in predicted_entities_by_doc.items()
            }
        },
        config.artifacts_dir / "test_predicted_entities.json",
    )
    save_json(compute_pair_volume(candidate_pairs_by_doc, pair_counts_by_doc), config.artifacts_dir / "test_pair_volume.json")
    save_warnings = _save_model_artifacts_safely(mention_detector, relation_model, config.artifacts_dir)
    if save_warnings:
        save_json({"warnings": save_warnings}, config.artifacts_dir / "model_save_warnings.json")
    return {"submission_path": str(output_path), "documents": len(test_documents), "model_save_warnings": save_warnings}
