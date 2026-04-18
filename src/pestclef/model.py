from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .config import ExperimentConfig
from .features import FeatureVectorizer, RelationSchema


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


class RelationModel(Protocol):
    labels: List[str]
    thresholds: Dict[str, float]

    def predict_scores(self, feature_dicts: Sequence[Dict[str, float]]) -> np.ndarray:
        ...

    def predict_labels(self, feature_dicts: Sequence[Dict[str, float]]) -> List[List[str]]:
        ...

    def save(self, path: Path) -> None:
        ...


@dataclass
class LinearMultiLabelModel:
    labels: List[str]
    vectorizer: FeatureVectorizer
    weights: np.ndarray
    bias: np.ndarray
    thresholds: Dict[str, float]

    def predict_scores(self, feature_dicts: Sequence[Dict[str, float]]) -> np.ndarray:
        x = self.vectorizer.transform(feature_dicts)
        return sigmoid(x @ self.weights + self.bias)

    def predict_labels(self, feature_dicts: Sequence[Dict[str, float]]) -> List[List[str]]:
        scores = self.predict_scores(feature_dicts)
        outputs: List[List[str]] = []
        for row in scores:
            labels = [
                label
                for index, label in enumerate(self.labels)
                if row[index] >= self.thresholds[label]
            ]
            outputs.append(labels)
        return outputs

    def save(self, path: Path) -> None:
        payload = {
            "labels": self.labels,
            "feature_to_index": self.vectorizer.feature_to_index,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "thresholds": self.thresholds,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearMultiLabelModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vectorizer = FeatureVectorizer(feature_cap=len(payload["feature_to_index"]))
        vectorizer.feature_to_index = {key: int(value) for key, value in payload["feature_to_index"].items()}
        return cls(
            labels=list(payload["labels"]),
            vectorizer=vectorizer,
            weights=np.array(payload["weights"], dtype=np.float32),
            bias=np.array(payload["bias"], dtype=np.float32),
            thresholds={key: float(value) for key, value in payload["thresholds"].items()},
        )


@dataclass
class BinaryLabelModel:
    label: str
    estimator: LogisticRegression | None
    constant_score: float = 0.0

    def predict_scores(self, x: object) -> np.ndarray:
        if self.estimator is None:
            rows = getattr(x, "shape")[0]
            return np.full(rows, self.constant_score, dtype=np.float32)
        return self.estimator.predict_proba(x)[:, 1].astype(np.float32)


@dataclass
class SklearnRelationModel:
    labels: List[str]
    vectorizer: DictVectorizer
    label_models: Dict[str, BinaryLabelModel]
    thresholds: Dict[str, float]

    def predict_scores(self, feature_dicts: Sequence[Dict[str, float]]) -> np.ndarray:
        x = self.vectorizer.transform(feature_dicts)
        scores = np.zeros((x.shape[0], len(self.labels)), dtype=np.float32)
        for column, label in enumerate(self.labels):
            scores[:, column] = self.label_models[label].predict_scores(x)
        return scores

    def predict_labels(self, feature_dicts: Sequence[Dict[str, float]]) -> List[List[str]]:
        scores = self.predict_scores(feature_dicts)
        outputs: List[List[str]] = []
        for row in scores:
            labels = [
                label
                for index, label in enumerate(self.labels)
                if row[index] >= self.thresholds[label]
            ]
            outputs.append(labels)
        return outputs

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnRelationModel":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Unexpected model type loaded from {path}")
        return loaded


def train_relation_model(
    examples: Sequence[Dict[str, object]],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> RelationModel:
    if config.model_name == "numpy":
        return train_linear_multilabel(
            [example["features"] for example in examples],
            [example["labels"] for example in examples],
            config,
        )
    if config.model_name == "sklearn":
        return train_sklearn_relation_model(examples, schema, config)
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def train_linear_multilabel(
    feature_dicts: Sequence[Dict[str, float]],
    label_sets: Sequence[set[str]],
    config: ExperimentConfig,
) -> LinearMultiLabelModel:
    vectorizer = FeatureVectorizer(feature_cap=config.feature_cap)
    vectorizer.fit(feature_dicts)
    x = vectorizer.transform(feature_dicts)
    labels = list(config.relation_labels)
    y = np.zeros((len(label_sets), len(labels)), dtype=np.float32)
    for row_index, active_labels in enumerate(label_sets):
        for label in active_labels:
            y[row_index, labels.index(label)] = 1.0

    rng = np.random.default_rng(config.random_seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(labels))).astype(np.float32)
    bias = np.zeros(len(labels), dtype=np.float32)

    class_frequency = np.maximum(y.mean(axis=0), 1e-4)
    positive_weights = (1.0 / class_frequency).astype(np.float32)
    positive_weights = positive_weights / positive_weights.mean()

    batch_size = max(1, min(config.batch_size, len(x)))
    for _ in range(config.epochs):
        order = rng.permutation(len(x))
        for batch_start in range(0, len(x), batch_size):
            batch_indices = order[batch_start : batch_start + batch_size]
            xb = x[batch_indices]
            yb = y[batch_indices]
            logits = xb @ weights + bias
            predictions = sigmoid(logits)
            errors = predictions - yb
            errors *= np.where(yb > 0.0, positive_weights.reshape(1, -1), 1.0)
            grad_w = xb.T @ errors / len(batch_indices) + config.l2 * weights
            grad_b = errors.mean(axis=0)
            weights -= config.learning_rate * grad_w
            bias -= config.learning_rate * grad_b

    thresholds = {}
    train_scores = sigmoid(x @ weights + bias)
    for label_index, label in enumerate(labels):
        gold = y[:, label_index]
        score_column = train_scores[:, label_index]
        thresholds[label] = calibrate_threshold(score_column, gold, config.relation_thresholds.get(label, 0.5))

    return LinearMultiLabelModel(
        labels=labels,
        vectorizer=vectorizer,
        weights=weights,
        bias=bias,
        thresholds=thresholds,
    )


def train_sklearn_relation_model(
    examples: Sequence[Dict[str, object]],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> SklearnRelationModel:
    feature_dicts = [example["features"] for example in examples]
    labels = list(config.relation_labels)
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(feature_dicts)

    label_models: Dict[str, BinaryLabelModel] = {}
    thresholds: Dict[str, float] = {}
    for label in labels:
        mask = np.array(
            [
                schema.is_valid_pair(label, str(example["subject_type"]), str(example["object_type"]))
                for example in examples
            ],
            dtype=bool,
        )
        y = np.array(
            [1 if label in example["labels"] else 0 for example in examples],
            dtype=np.int32,
        )[mask]
        if y.size == 0:
            label_models[label] = BinaryLabelModel(label=label, estimator=None, constant_score=0.0)
            thresholds[label] = config.relation_thresholds.get(label, 0.5)
            continue

        x_label = x[mask]
        if np.unique(y).size < 2:
            constant_score = float(y[0])
            label_models[label] = BinaryLabelModel(label=label, estimator=None, constant_score=constant_score)
            thresholds[label] = calibrate_threshold(
                np.full(y.shape[0], constant_score, dtype=np.float32),
                y.astype(np.float32),
                config.relation_thresholds.get(label, 0.5),
            )
            continue

        estimator = LogisticRegression(
            C=config.sklearn_c,
            class_weight="balanced",
            max_iter=config.sklearn_max_iter,
            random_state=config.random_seed,
            solver="liblinear",
        )
        estimator.fit(x_label, y)
        score_column = estimator.predict_proba(x_label)[:, 1].astype(np.float32)
        thresholds[label] = calibrate_threshold(
            score_column,
            y.astype(np.float32),
            config.relation_thresholds.get(label, 0.5),
        )
        label_models[label] = BinaryLabelModel(label=label, estimator=estimator)

    return SklearnRelationModel(
        labels=labels,
        vectorizer=vectorizer,
        label_models=label_models,
        thresholds=thresholds,
    )


def calibrate_threshold(
    scores: np.ndarray,
    gold: np.ndarray,
    default_threshold: float,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
) -> float:
    if gold.sum() == 0:
        return default_threshold
    clipped_default = float(np.clip(default_threshold, min_threshold, max_threshold))
    candidates = np.unique(np.quantile(scores, np.linspace(0.6, 0.995, 30)))
    candidates = np.concatenate(([scores.min()], candidates, [scores.max()], [clipped_default, min_threshold, max_threshold]))
    candidates = np.array([float(candidate) for candidate in candidates if min_threshold <= float(candidate) <= max_threshold], dtype=np.float32)
    if candidates.size == 0:
        return clipped_default
    best_threshold = default_threshold
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.float32)
        tp = float(((predicted == 1) & (gold == 1)).sum())
        fp = float(((predicted == 1) & (gold == 0)).sum())
        fn = float(((predicted == 0) & (gold == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return float(np.clip(best_threshold, min_threshold, max_threshold))
