from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.config import ExperimentConfig
from pestclef.data import load_documents
from pestclef.pipeline import predict_document_edges, train_gold_entity_baseline
from pestclef.submission import validate_submission_rows
from pestclef.features import RelationSchema
from pestclef.model import train_linear_multilabel, train_relation_model
from pestclef.features import generate_relation_examples


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = ExperimentConfig(
            project_root=self.root,
            artifacts_dir=Path(self.temp_dir.name),
            epochs=2,
            batch_size=32,
            feature_cap=5000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_offsets_align_with_text(self) -> None:
        documents = load_documents("train", self.config)
        sample = documents[0]
        for mention in sample.mentions[:10]:
            text = " ".join(sample.text[a:b] for a, b in mention.offsets)
            self.assertEqual(text, mention.form)

    def test_submission_validator_accepts_strings(self) -> None:
        rows = [{"doc_id": "123", "knowledge_graph": json.dumps([{"subject": "a", "predicate": "b", "object": "c"}])}]
        self.assertEqual(validate_submission_rows(rows), [])

    def test_candidate_pruning_keeps_positive_docs(self) -> None:
        documents = load_documents("train", self.config)[:5]
        schema = RelationSchema.from_documents(documents)
        examples = generate_relation_examples(documents, schema, self.config)
        positives = sum(1 for example in examples if example["labels"])
        self.assertGreater(positives, 0)

    def test_hardcoded_schema_blocks_invalid_pairs(self) -> None:
        schema = RelationSchema.hardcoded()
        self.assertFalse(schema.is_valid_pair("Found_on", "Date", "Pest"))
        self.assertFalse(schema.is_valid_pair("Located_in", "Dissemination_pathway", "Location"))
        self.assertFalse(schema.is_valid_pair("Occurs_on", "Dissemination_pathway", "Date"))
        self.assertTrue(schema.is_valid_pair("Causes", "Pest", "Disease"))

    def test_gold_baseline_trains(self) -> None:
        result = train_gold_entity_baseline(self.config)
        self.assertIn("metrics", result)
        self.assertIn("micro", result["metrics"])

    def test_predict_document_edges_runs(self) -> None:
        train_documents = load_documents("train", self.config)[:20]
        schema = RelationSchema.from_documents(train_documents)
        examples = generate_relation_examples(train_documents, schema, self.config)
        model = train_linear_multilabel(
            [example["features"] for example in examples],
            [example["labels"] for example in examples],
            self.config,
        )
        dev_document = load_documents("dev", self.config)[0]
        predictions = predict_document_edges(dev_document, dev_document.canonical_entities, schema, model, self.config)
        self.assertIsInstance(predictions, list)

    def test_sklearn_baseline_trains(self) -> None:
        sklearn_config = ExperimentConfig(
            project_root=self.root,
            artifacts_dir=Path(self.temp_dir.name),
            model_name="sklearn",
            epochs=2,
            batch_size=32,
            feature_cap=5000,
        )
        train_documents = load_documents("train", sklearn_config)[:20]
        schema = RelationSchema.from_documents(train_documents)
        examples = generate_relation_examples(train_documents, schema, sklearn_config)
        model = train_relation_model(examples, schema, sklearn_config)
        dev_document = load_documents("dev", sklearn_config)[0]
        predictions = predict_document_edges(dev_document, dev_document.canonical_entities, schema, model, sklearn_config)
        self.assertIsInstance(predictions, list)


if __name__ == "__main__":
    unittest.main()
