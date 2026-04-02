"""PestCLEF 2026 baseline pipeline."""

from .config import ExperimentConfig
from .pipeline import (
    run_dev_evaluation,
    run_test_submission,
    train_gold_entity_baseline,
)

__all__ = [
    "ExperimentConfig",
    "run_dev_evaluation",
    "run_test_submission",
    "train_gold_entity_baseline",
]
