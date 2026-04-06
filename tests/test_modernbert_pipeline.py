from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.config import ExperimentConfig
from pestclef.data import load_documents
from pestclef.features import RelationSchema
from pestclef.modernbert import (
    build_mention_training_rows,
    build_relation_context_text,
    build_tiny_encoder_files,
    decode_window_predictions,
    generate_relation_text_examples,
    get_bio_labels,
    merge_predicted_mentions,
    ModernBertRelationModel,
    PredictedMentionSpan,
    predict_canonical_entities_with_detector,
    train_modernbert_relation_model,
)
from pestclef.pipeline import run_dev_evaluation, run_test_submission, train_gold_entity_baseline


class ModernBertPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.encoder_dir = self.temp_path / "tiny_encoder"
        build_tiny_encoder_files(self.encoder_dir)
        self.config = ExperimentConfig(
            project_root=self.temp_path,
            artifacts_dir=self.temp_path / "artifacts",
            model_name="modernbert_staged",
            encoder_name=str(self.encoder_dir),
            encoder_random_init=True,
            local_files_only=True,
            device="cpu",
            epochs=1,
            batch_size=4,
            train_batch_size=1,
            eval_batch_size=1,
            gradient_accumulation_steps=1,
            max_seq_length=64,
            doc_stride=16,
            relation_max_seq_length=96,
            relation_context_sentence_radius=1,
            early_stopping_patience=1,
        )
        self.data_config = ExperimentConfig(project_root=self.repo_root)
        self.train_documents = load_documents("train", self.data_config)[:4]
        self.dev_documents = load_documents("dev", self.data_config)[:2]
        self.test_documents = load_documents("test", self.data_config)[:2]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _fake_load_documents(self, split: str, _config: ExperimentConfig):
        return {
            "train": self.train_documents,
            "dev": self.dev_documents,
            "test": self.test_documents,
        }[split]

    def test_bio_training_rows_include_positive_labels(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_dir, use_fast=True, local_files_only=True)
        label_to_id = {label: index for index, label in enumerate(get_bio_labels())}
        rows = build_mention_training_rows(self.train_documents[:1], tokenizer, label_to_id, self.config)
        positive_count = sum(int((row["labels"] > 0).sum().item()) for row in rows)
        self.assertGreater(positive_count, 0)

    def test_merge_mentions_prefers_high_confidence_span(self) -> None:
        document = self.train_documents[0]
        mentions = merge_predicted_mentions(
            [
                PredictedMentionSpan(entity_type="Pest", start=10, end=22, confidence=0.9),
                PredictedMentionSpan(entity_type="Pest", start=12, end=20, confidence=0.4),
            ],
            document,
        )
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].offsets[0], (10, 22))

    def test_decode_predictions_collapses_subtokens_to_word_spans(self) -> None:
        text = "Bursaphelenchus xylophilus"
        spans = decode_window_predictions(
            text,
            [(0, 6), (6, 15), (16, 19), (19, 27), (0, 0)],
            [9, 10, 10, 10, 0],
            [0.95, 0.93, 0.91, 0.92, 0.0],
            {0: "O", 9: "B-Pest", 10: "I-Pest"},
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].start, spans[0].end), (0, len(text)))

    def test_merge_mentions_filters_fragmentary_spans(self) -> None:
        document = self.train_documents[0]
        document = type(document)(
            doc_id="demo",
            split="dev",
            text="Bursaphelenchus xylophilus was detected.",
            layout=[],
        )
        mentions = merge_predicted_mentions(
            [
                PredictedMentionSpan(entity_type="Pest", start=0, end=27, confidence=0.82),
                PredictedMentionSpan(entity_type="Pest", start=16, end=18, confidence=0.86),
                PredictedMentionSpan(entity_type="Pest", start=0, end=1, confidence=0.99),
            ],
            document,
        )
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].form, "Bursaphelenchus xylophilus")

    def test_relation_context_adds_role_markers(self) -> None:
        schema = RelationSchema.from_documents(self.train_documents)
        document = self.train_documents[0]
        pair = None
        entities = document.canonical_entities
        for subject in entities:
            for obj in entities:
                if subject.entity_id == obj.entity_id:
                    continue
                if schema.compatible_relations(subject.entity_type, obj.entity_type):
                    pair = (subject, obj)
                    break
            if pair:
                break
        assert pair is not None
        text = build_relation_context_text(document, pair[0], pair[1], self.config)
        self.assertIn(f"<SUBJ_{pair[0].entity_type}>", text)
        self.assertIn(f"<OBJ_{pair[1].entity_type}>", text)

    def test_relation_model_save_and_load(self) -> None:
        schema = RelationSchema.from_documents(self.train_documents)
        train_examples = generate_relation_text_examples(self.train_documents, schema, self.config)
        dev_examples = generate_relation_text_examples(self.dev_documents, schema, self.config)
        relation_model = train_modernbert_relation_model(
            train_examples,
            self.config,
            calibration_examples=dev_examples,
            validation_examples=dev_examples,
        )
        save_path = self.temp_path / "relation_model"
        relation_model.save(save_path)
        reloaded = ModernBertRelationModel.load(save_path, self.config)
        self.assertEqual(reloaded.labels, relation_model.labels)
        self.assertEqual(set(reloaded.thresholds.keys()), set(relation_model.thresholds.keys()))

    def test_predicted_mentions_can_be_canonicalized(self) -> None:
        document = self.train_documents[0]

        class StubDetector:
            def predict_mentions(self, _document: object):
                return document.mentions[:3]

        entities = predict_canonical_entities_with_detector(document, StubDetector())  # type: ignore[arg-type]
        self.assertGreater(len(entities), 0)

    def test_modernbert_gold_pipeline_runs(self) -> None:
        with patch("pestclef.pipeline.load_documents", side_effect=self._fake_load_documents):
            result = train_gold_entity_baseline(self.config)
        self.assertIn("metrics", result)
        self.assertTrue((self.config.artifacts_dir / "relation_model").exists())

    def test_modernbert_end_to_end_pipeline_runs(self) -> None:
        with patch("pestclef.pipeline.load_documents", side_effect=self._fake_load_documents):
            result = run_dev_evaluation(self.config)
        self.assertIn("metrics", result)
        self.assertIn("mention_metrics", result)
        self.assertTrue((self.config.artifacts_dir / "mention_model").exists())
        self.assertTrue((self.config.artifacts_dir / "relation_model").exists())
        self.assertTrue((self.config.artifacts_dir / "dev_predicted_mentions.json").exists())
        self.assertTrue((self.config.artifacts_dir / "dev_mention_metrics.json").exists())

    def test_modernbert_submission_pipeline_runs(self) -> None:
        with patch("pestclef.pipeline.load_documents", side_effect=self._fake_load_documents):
            result = run_test_submission(self.config, train_on_dev=False)
        self.assertIn("submission_path", result)
        submission_path = Path(result["submission_path"])
        self.assertTrue(submission_path.exists())
        rows = submission_path.read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
