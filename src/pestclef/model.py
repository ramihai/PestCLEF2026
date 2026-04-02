from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import ExperimentConfig
from .features import FeatureVectorizer


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


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


def calibrate_threshold(scores: np.ndarray, gold: np.ndarray, default_threshold: float) -> float:
    if gold.sum() == 0:
        return default_threshold
    candidates = np.unique(np.quantile(scores, np.linspace(0.6, 0.995, 30)))
    candidates = np.concatenate(([scores.min()], candidates, [scores.max()]))
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
    return best_threshold
