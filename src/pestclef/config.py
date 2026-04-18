from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentConfig:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    json_dir: Path = Path("data/json")
    docs_dir: Path = Path("data/EPOP_documents")
    artifacts_dir: Path = Path("artifacts")
    model_name: str = "numpy"
    encoder_name: str = "answerdotai/ModernBERT-base"
    random_seed: int = 13
    max_sentence_distance: int = 3
    max_layout_distance: int = 6
    min_alias_length: int = 3
    learning_rate: float = 0.08
    learning_rate_encoder: float = 2e-5
    epochs: int = 8
    batch_size: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    l2: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 2
    feature_cap: int = 25000
    sklearn_c: float = 1.0
    sklearn_max_iter: int = 400
    max_seq_length: int = 1024
    doc_stride: int = 256
    relation_max_seq_length: int = 768
    relation_context_sentence_radius: int = 1
    mention_positive_sentence_radius: int = 1
    mention_negative_sentence_ratio: float = 0.5
    mention_min_windows_per_document: int = 1
    mention_use_focal_loss: bool = True
    mention_focal_gamma: float = 2.0
    mention_class_weight_cap: float = 12.0
    mention_hybrid_lexicon: bool = True
    mention_hybrid_lexicon_confidence: float = 0.55
    mention_cleanup_profile: str = "strict_v2"
    mention_type_denylists_enabled: bool = True
    entity_alias_merge_strategy: str = "heuristic_v1"
    mention_threshold_tuning_strategy: str = "relation_aware_v1"
    mention_threshold_candidate_recall_weight: float = 0.75
    mention_threshold_mention_f1_weight: float = 0.25
    relation_train_with_predicted_entities: bool = True
    relation_predicted_entity_mix_ratio: float = 0.5
    relation_calibrate_on_predicted_entities: bool = True
    relation_pair_pruning_profile: str = "precision_v1"
    relation_threshold_search_min: float = 0.05
    relation_threshold_search_max: float = 0.95
    relation_threshold_min_positives_for_full_tuning: int = 20
    relation_threshold_low_support_margin: float = 0.15
    device: str = "auto"
    local_files_only: bool = False
    encoder_random_init: bool = False
    relation_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "Located_in": 0.48,
            "Found_on": 0.48,
            "Occurs_on": 0.48,
            "Affects": 0.48,
            "Causes": 0.48,
            "Dispersed_by": 0.48,
            "Transmits": 0.48,
        }
    )
    relation_labels: List[str] = field(
        default_factory=lambda: [
            "Located_in",
            "Found_on",
            "Occurs_on",
            "Affects",
            "Causes",
            "Dispersed_by",
            "Transmits",
        ]
    )
    mps_device_note: str = "Designed to stay CPU/NumPy friendly; suitable for M4 Pro notebook workflow."

    def ensure_artifacts_dir(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir
