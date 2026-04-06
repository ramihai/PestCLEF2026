from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from .config import ExperimentConfig
from .data import deduplicate_edges, export_documents, load_documents
from .evaluation import build_error_report, compute_mention_metrics, compute_metrics, save_json, save_predictions
from .features import RelationSchema, generate_relation_examples, should_consider_pair
from .mention_detection import MentionLexicon, predict_canonical_entities
from .model import RelationModel, train_relation_model
from .schema import CanonicalEntity, Document, RelationEdge
from .submission import write_submission


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
        build_canonical_entities_from_mentions,
        generate_relation_text_examples,
        predict_document_edges_with_relation_model,
        train_modernbert_mention_detector,
        train_modernbert_relation_model,
    )

    config.ensure_artifacts_dir()
    train_documents = load_documents("train", config)
    dev_documents = load_documents("dev", config)
    schema = RelationSchema.from_documents(train_documents)

    mention_detector = train_modernbert_mention_detector(train_documents, config, validation_documents=dev_documents)
    predicted_mentions_by_doc = {}
    predicted_entities_by_doc: Dict[str, List[CanonicalEntity]] = {}
    for document in dev_documents:
        predicted_mentions = mention_detector.predict_mentions(document)
        predicted_mentions_by_doc[document.doc_id] = predicted_mentions
        predicted_entities_by_doc[document.doc_id] = build_canonical_entities_from_mentions(predicted_mentions)

    train_examples = generate_relation_text_examples(train_documents, schema, config)
    dev_examples = generate_relation_text_examples(dev_documents, schema, config)
    relation_model = train_modernbert_relation_model(
        train_examples,
        config,
        calibration_examples=dev_examples,
        validation_examples=dev_examples,
    )

    predicted_edges_by_doc: Dict[str, List[RelationEdge]] = {}
    for document in dev_documents:
        predicted_edges_by_doc[document.doc_id] = predict_document_edges_with_relation_model(
            document,
            predicted_entities_by_doc[document.doc_id],
            schema,
            relation_model,
            config,
        )

    mention_model_dir = config.artifacts_dir / "mention_model"
    relation_model_dir = config.artifacts_dir / "relation_model"
    mention_detector.save(mention_model_dir)
    relation_model.save(relation_model_dir)
    save_json({"schema": schema.to_serializable()}, config.artifacts_dir / "relation_schema.json")
    save_json({"config": asdict(config)}, config.artifacts_dir / "experiment_config.json")
    save_json({"train_documents": export_documents(train_documents[:3])}, config.artifacts_dir / "data_preview.json")

    metrics = compute_metrics(dev_documents, predicted_edges_by_doc, config.relation_labels)
    mention_metrics = compute_mention_metrics(
        dev_documents,
        predicted_mentions_by_doc,
        ["Disease", "Date", "Dissemination_pathway", "Location", "Pest", "Plant", "Vector"],
    )
    error_report = build_error_report(dev_documents, predicted_edges_by_doc, predicted_entities_by_doc)
    save_predictions(predicted_edges_by_doc, config.artifacts_dir / "dev_end_to_end_predictions.json")
    save_json(metrics, config.artifacts_dir / "dev_end_to_end_metrics.json")
    save_json(mention_metrics, config.artifacts_dir / "dev_mention_metrics.json")
    save_json(error_report, config.artifacts_dir / "dev_end_to_end_error_report.json")
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
                    }
                    for entity in entities
                ]
                for doc_id, entities in predicted_entities_by_doc.items()
            }
        },
        config.artifacts_dir / "dev_predicted_entities.json",
    )
    return {
        "metrics": metrics,
        "mention_metrics": mention_metrics,
        "error_report": error_report,
    }


def run_modernbert_test_submission(config: ExperimentConfig, train_on_dev: bool = False) -> Dict[str, object]:
    from .modernbert import (
        generate_relation_text_examples,
        predict_canonical_entities_with_detector,
        predict_document_edges_with_relation_model,
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

    mention_detector = train_modernbert_mention_detector(
        mention_train_documents,
        config,
        validation_documents=dev_documents if not train_on_dev else train_documents,
    )
    schema = RelationSchema.from_documents(relation_train_documents)
    relation_train_examples = generate_relation_text_examples(relation_train_documents, schema, config)
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
    for document in test_documents:
        predicted_entities = predict_canonical_entities_with_detector(document, mention_detector)
        predicted_entities_by_doc[document.doc_id] = predicted_entities
        predictions[document.doc_id] = predict_document_edges_with_relation_model(
            document,
            predicted_entities,
            schema,
            relation_model,
            config,
        )

    mention_model_dir = config.artifacts_dir / "mention_model"
    relation_model_dir = config.artifacts_dir / "relation_model"
    mention_detector.save(mention_model_dir)
    relation_model.save(relation_model_dir)

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
