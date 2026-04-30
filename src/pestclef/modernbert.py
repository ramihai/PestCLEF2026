from __future__ import annotations

import json
import logging
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from .config import ExperimentConfig
from .data import build_canonical_entities, locate_span, normalize_text, split_sentences
from .evaluation import compute_candidate_recall, compute_mention_metrics
from .features import RelationSchema, enumerate_candidate_entity_pairs, should_consider_pair
from .mention_detection import MentionLexicon, build_coref_guess, detect_mentions_as_mentions
from .model import calibrate_threshold
from .schema import CanonicalEntity, Document, Mention, RelationEdge


ENTITY_TYPES = [
    "Disease",
    "Date",
    "Dissemination_pathway",
    "Location",
    "Pest",
    "Plant",
    "Vector",
]

WORD_SPAN_PATTERN = re.compile(r"\w+(?:[./-]\w+)*", flags=re.UNICODE)
SPAN_TRIM_CHARS = string.whitespace + string.punctuation
MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
DATE_CUE_TOKENS = {
    "early",
    "late",
    "mid",
    "beginning",
    "end",
    "during",
    "since",
    "from",
    "between",
    "until",
    "year",
    "years",
    "month",
    "months",
    "season",
    "spring",
    "summer",
    "autumn",
    "fall",
    "winter",
    "century",
    "decade",
    "annual",
    "weekly",
    "daily",
}
CENTURY_TOKENS = {
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
    "twentyfirst",
    "first",
    "second",
    "third",
    "fourth",
}
TYPE_DENYLISTS: Dict[str, set[str]] = {
    "Date": {
        "at",
        "if",
        "latest",
        "supplementary",
        "time",
        "times",
    },
    "Location": {
        "green",
        "red",
        "supplementary",
        "latest",
        "faostat",
        "territory of the union",
        "entire provinces",
        "individuals",
        "farmers",
    },
    "Vector": {
        "individuals",
        "individual",
        "orf",
        "supplementary",
        "red",
        "immunoglobulin-like",
        "5-10",
        "genbank",
        "vector",
        "vectors",
        "species",
        "pathotype",
        "st2",
    },
    "Dissemination_pathway": {
        "fruit",
        "fruits",
        "plant",
        "plants",
    },
}


def resolve_torch_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif requested_device == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            print("[device] MPS requested but not available — falling back to cpu", flush=True)
            device = torch.device("cpu")
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[device] CUDA requested but not available — falling back to cpu", flush=True)
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    print(f"[device] Using device: {device}", flush=True)
    return device


def get_bio_labels(entity_types: Sequence[str] = ENTITY_TYPES) -> List[str]:
    labels = ["O"]
    for entity_type in entity_types:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    return labels


def get_relation_marker_tokens(entity_types: Sequence[str] = ENTITY_TYPES) -> List[str]:
    tokens = []
    for entity_type in entity_types:
        tokens.extend(
            [
                f"<SUBJ_{entity_type}>",
                f"</SUBJ_{entity_type}>",
                f"<OBJ_{entity_type}>",
                f"</OBJ_{entity_type}>",
            ]
        )
    return tokens


def _resolve_encoder_name(config: ExperimentConfig, encoder_role: str) -> str:
    if encoder_role == "mention":
        override = getattr(config, "mention_encoder_name", None)
        if override:
            return override
    return config.encoder_name


def resolve_mention_seq_length(config: ExperimentConfig) -> int:
    override = getattr(config, "mention_max_seq_length", None)
    if override is not None:
        return int(override)
    return int(config.max_seq_length)


def resolve_mention_doc_stride(config: ExperimentConfig) -> int:
    override = getattr(config, "mention_doc_stride", None)
    if override is not None:
        return int(override)
    return int(config.doc_stride)


def _load_tokenizer(config: ExperimentConfig, encoder_role: str = "relation") -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        _resolve_encoder_name(config, encoder_role),
        use_fast=True,
        local_files_only=config.local_files_only,
    )


def _load_model_config(
    config: ExperimentConfig,
    num_labels: int,
    task_type: str,
    encoder_role: str = "relation",
) -> AutoConfig:
    model_config = AutoConfig.from_pretrained(
        _resolve_encoder_name(config, encoder_role),
        local_files_only=config.local_files_only,
    )
    model_config.num_labels = num_labels
    if task_type == "relation":
        model_config.problem_type = "multi_label_classification"
    return model_config


def _load_token_classification_model(config: ExperimentConfig, num_labels: int) -> PreTrainedModel:
    model_config = _load_model_config(config, num_labels, task_type="mention", encoder_role="mention")
    if config.encoder_random_init:
        return AutoModelForTokenClassification.from_config(model_config)
    return AutoModelForTokenClassification.from_pretrained(
        _resolve_encoder_name(config, "mention"),
        config=model_config,
        local_files_only=config.local_files_only,
    )


def _load_sequence_classification_model(config: ExperimentConfig, num_labels: int) -> PreTrainedModel:
    model_config = _load_model_config(config, num_labels, task_type="relation", encoder_role="relation")
    if config.encoder_random_init:
        return AutoModelForSequenceClassification.from_config(model_config)
    return AutoModelForSequenceClassification.from_pretrained(
        _resolve_encoder_name(config, "relation"),
        config=model_config,
        local_files_only=config.local_files_only,
    )


def _stack_feature_rows(rows: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return list(rows)


def _collate_batches(rows: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = rows[0].keys()
    return {key: torch.stack([row[key] for row in rows]) for key in keys}


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clone_state_dict(model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _evaluate_dataset(
    model: torch.nn.Module,
    dataset_rows: Sequence[Dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> float:
    if not dataset_rows:
        return 0.0
    loader = DataLoader(_stack_feature_rows(dataset_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate_batches)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(**_move_batch_to_device(batch, device))
            losses.append(float(outputs.loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _train_transformer_model(
    model: torch.nn.Module,
    train_rows: Sequence[Dict[str, torch.Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    validation_rows: Sequence[Dict[str, torch.Tensor]] | None = None,
    stage_label: str = "model",
) -> torch.nn.Module:
    n_epochs = max(1, config.epochs)
    n_samples = len(train_rows)
    print(
        f"\n[{stage_label}] Starting training | "
        f"device={device} | samples={n_samples} | epochs={n_epochs} | "
        f"batch={config.train_batch_size} | lr={config.learning_rate_encoder:.2e}",
        flush=True,
    )
    if not train_rows:
        print(f"[{stage_label}] No training rows — skipping.", flush=True)
        return model.to(device)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate_encoder, weight_decay=config.weight_decay)
    loader = DataLoader(
        _stack_feature_rows(train_rows),
        batch_size=max(1, config.train_batch_size),
        shuffle=True,
        collate_fn=_collate_batches,
    )
    n_batches = len(loader)
    total_updates_per_epoch = max(1, (n_batches + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
    total_steps = max(1, total_updates_per_epoch * n_epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(
        f"[{stage_label}] Loader: {n_batches} batches/epoch | "
        f"{total_updates_per_epoch} updates/epoch | warmup={warmup_steps} steps",
        flush=True,
    )

    best_state_dict: Dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    patience = 0
    optimizer.zero_grad(set_to_none=True)
    t_train_start = time.time()

    for epoch in range(n_epochs):
        t_epoch_start = time.time()
        model.train()
        epoch_losses: List[float] = []
        for step, batch in enumerate(loader, start=1):
            outputs = model(**_move_batch_to_device(batch, device))
            loss = outputs.loss / max(1, config.gradient_accumulation_steps)
            loss.backward()
            epoch_losses.append(float(outputs.loss.detach().cpu().item()))
            if step % max(1, config.gradient_accumulation_steps) == 0 or step == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_elapsed = time.time() - t_epoch_start
        val_str = ""
        stopped_early = False
        if validation_rows:
            validation_loss = _evaluate_dataset(model, validation_rows, config.eval_batch_size, device)
            val_str = f" | val_loss={validation_loss:.4f}"
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state_dict = _clone_state_dict(model)
                patience = 0
                val_str += " ✓"
            else:
                patience += 1
                val_str += f" (patience {patience}/{config.early_stopping_patience})"
                if patience >= config.early_stopping_patience:
                    stopped_early = True

        print(
            f"[{stage_label}] Epoch {epoch + 1}/{n_epochs} | "
            f"train_loss={train_loss:.4f}{val_str} | {epoch_elapsed:.1f}s",
            flush=True,
        )
        if stopped_early:
            print(f"[{stage_label}] Early stopping at epoch {epoch + 1}.", flush=True)
            break

    total_elapsed = time.time() - t_train_start
    print(f"[{stage_label}] Training complete in {total_elapsed:.1f}s", flush=True)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _build_word_spans(text: str) -> List[tuple[int, int]]:
    return [match.span() for match in WORD_SPAN_PATTERN.finditer(text)]


def _locate_covering_span(start: int, end: int, spans: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    for span_start, span_end in spans:
        if start >= span_start and end <= span_end:
            return (span_start, span_end)
    return None


def _has_word_char_between(text: str, start: int, end: int) -> bool:
    return any(character.isalnum() for character in text[start:end])


def _build_mention_segments(document: Document) -> List[tuple[str, str, int, int]]:
    segments = []
    for mention in document.mentions:
        for start, end in mention.offsets:
            segments.append((mention.mention_id, mention.entity_type, start, end))
    return segments


def _label_word_offset(
    word_start: int,
    word_end: int,
    segments: Sequence[tuple[str, str, int, int]],
    text: str,
) -> tuple[str, str] | None:
    for _mention_id, entity_type, span_start, span_end in segments:
        if word_start >= span_start and word_end <= span_end:
            prefix = "B" if not _has_word_char_between(text, span_start, word_start) else "I"
            return prefix, entity_type
    return None


def _partially_overlaps_segment(
    token_start: int,
    token_end: int,
    segments: Sequence[tuple[str, str, int, int]],
) -> bool:
    for _mention_id, _entity_type, span_start, span_end in segments:
        if _overlaps((token_start, token_end), (span_start, span_end)) and not (token_start >= span_start and token_end <= span_end):
            return True
    return False


def _build_mention_window_examples(
    documents: Sequence[Document],
    config: ExperimentConfig,
) -> List[Dict[str, object]]:
    window_examples: List[Dict[str, object]] = []
    for document in documents:
        if not document.text.strip():
            continue
        sentence_spans = split_sentences(document.text)
        if not sentence_spans:
            sentence_spans = [(0, len(document.text))]
        positive_sentence_indices = sorted(
            {
                mention.sentence_index
                for mention in document.mentions
                if 0 <= mention.sentence_index < len(sentence_spans)
            }
        )
        positive_index_set = set(positive_sentence_indices)
        window_bounds: set[tuple[int, int]] = set()
        for sentence_index in positive_sentence_indices:
            start_sentence = max(0, sentence_index - max(0, config.mention_positive_sentence_radius))
            end_sentence = min(len(sentence_spans) - 1, sentence_index + max(0, config.mention_positive_sentence_radius))
            window_bounds.add((sentence_spans[start_sentence][0], sentence_spans[end_sentence][1]))

        negative_indices = [index for index in range(len(sentence_spans)) if index not in positive_index_set]
        if positive_sentence_indices:
            max_negative = int(round(len(positive_sentence_indices) * max(0.0, config.mention_negative_sentence_ratio)))
            for index in negative_indices[:max_negative]:
                window_bounds.add(sentence_spans[index])
        else:
            minimum = max(1, int(config.mention_min_windows_per_document))
            for index in negative_indices[:minimum]:
                window_bounds.add(sentence_spans[index])

        if not window_bounds:
            window_bounds.add((0, len(document.text)))

        minimum_windows = max(1, int(config.mention_min_windows_per_document))
        if len(window_bounds) < minimum_windows:
            for index in range(len(sentence_spans)):
                window_bounds.add(sentence_spans[index])
                if len(window_bounds) >= minimum_windows:
                    break

        for window_start, window_end in sorted(window_bounds):
            window_text = document.text[window_start:window_end]
            if not window_text.strip():
                continue
            segments: List[tuple[str, str, int, int]] = []
            for mention in document.mentions:
                for mention_start, mention_end in mention.offsets:
                    if mention_start >= window_start and mention_end <= window_end:
                        segments.append(
                            (
                                mention.mention_id,
                                mention.entity_type,
                                mention_start - window_start,
                                mention_end - window_start,
                            )
                        )
            window_examples.append(
                {
                    "text": window_text,
                    "segments": segments,
                }
            )
    return window_examples


def build_mention_training_rows(
    documents: Sequence[Document],
    tokenizer: PreTrainedTokenizerBase,
    label_to_id: Dict[str, int],
    config: ExperimentConfig,
) -> List[Dict[str, torch.Tensor]]:
    if not documents:
        return []
    window_examples = _build_mention_window_examples(documents, config)
    if not window_examples:
        return []
    encoded = tokenizer(
        [str(example["text"]) for example in window_examples],
        truncation=True,
        max_length=resolve_mention_seq_length(config),
        stride=resolve_mention_doc_stride(config),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = encoded["overflow_to_sample_mapping"]
    rows = []
    for row_index in range(len(encoded["input_ids"])):
        example = window_examples[sample_mapping[row_index]]
        window_text = str(example["text"])
        segments = list(example["segments"])
        word_spans = _build_word_spans(window_text)
        labels = []
        for token_start, token_end in encoded["offset_mapping"][row_index]:
            if token_start == token_end == 0:
                labels.append(-100)
                continue
            word_span = _locate_covering_span(token_start, token_end, word_spans)
            if word_span is not None and token_start > word_span[0]:
                labels.append(-100)
                continue
            span_start, span_end = word_span or (token_start, token_end)
            label_parts = _label_word_offset(span_start, span_end, segments, window_text)
            if label_parts is not None:
                labels.append(label_to_id[f"{label_parts[0]}-{label_parts[1]}"])
                continue
            if _partially_overlaps_segment(span_start, span_end, segments):
                labels.append(-100)
                continue
            labels.append(label_to_id["O"])
        rows.append(
            {
                "input_ids": torch.tensor(encoded["input_ids"][row_index], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"][row_index], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
    return rows


def _build_mention_class_weights(
    train_rows: Sequence[Dict[str, torch.Tensor]],
    num_labels: int,
    cap: float,
) -> torch.Tensor:
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for row in train_rows:
        labels = row["labels"].view(-1)
        valid_labels = labels[labels >= 0]
        if valid_labels.numel() == 0:
            continue
        bincount = torch.bincount(valid_labels, minlength=num_labels).to(dtype=torch.float32)
        counts += bincount
    if counts.sum().item() == 0:
        return torch.ones(num_labels, dtype=torch.float32)
    frequencies = counts / counts.sum()
    weights = 1.0 / torch.clamp(frequencies, min=1e-6)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max(1.0, float(cap)))
    if num_labels > 0:
        weights[0] = min(float(weights[0].item()), 1.0)
    return weights


class FocalTokenClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            valid_mask = flat_labels != -100
            if bool(valid_mask.any()):
                valid_logits = flat_logits[valid_mask]
                valid_labels = flat_labels[valid_mask]
                ce_loss = F.cross_entropy(
                    valid_logits,
                    valid_labels,
                    reduction="none",
                    weight=self.class_weights.to(valid_logits.device) if self.class_weights is not None else None,
                )
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** max(0.0, float(self.gamma))) * ce_loss
                loss = focal_loss.mean()
            else:
                loss = logits.sum() * 0.0
        return type("TokenClassificationOutput", (), {"logits": logits, "loss": loss})()


@dataclass
class PredictedMentionSpan:
    entity_type: str
    start: int
    end: int
    confidence: float


def decode_window_predictions(
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    label_scores: Sequence[float],
    id_to_label: Dict[int, str],
) -> List[PredictedMentionSpan]:
    word_spans = _build_word_spans(text)
    word_level_predictions: List[tuple[int, int, int, float]] = []
    seen_spans: set[tuple[int, int]] = set()
    for (token_start, token_end), label_id, score in zip(offsets, label_ids, label_scores):
        if token_start == token_end == 0:
            continue
        word_span = _locate_covering_span(token_start, token_end, word_spans)
        if word_span is None:
            continue
        if token_start > word_span[0] or word_span in seen_spans:
            continue
        seen_spans.add(word_span)
        word_level_predictions.append((word_span[0], word_span[1], int(label_id), float(score)))

    candidates: List[PredictedMentionSpan] = []
    current_type: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_scores: List[float] = []

    def flush() -> None:
        nonlocal current_type, current_start, current_end, current_scores
        if current_type is None or current_start is None or current_end is None:
            current_type = None
            current_start = None
            current_end = None
            current_scores = []
            return
        candidates.append(
            PredictedMentionSpan(
                entity_type=current_type,
                start=current_start,
                end=current_end,
                confidence=float(np.mean(current_scores)) if current_scores else 0.0,
            )
        )
        current_type = None
        current_start = None
        current_end = None
        current_scores = []

    for token_start, token_end, label_id, score in word_level_predictions:
        label = id_to_label[int(label_id)]
        if label == "O":
            flush()
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "I" and current_type != entity_type:
            prefix = "B"
        if prefix == "B" or current_type != entity_type:
            flush()
            current_type = entity_type
            current_start = token_start
            current_end = token_end
            current_scores = [float(score)]
        else:
            current_end = token_end
            current_scores.append(float(score))
    flush()
    return candidates


def _normalize_predicted_span(
    span: PredictedMentionSpan,
    document: Document,
    config: ExperimentConfig | None = None,
) -> PredictedMentionSpan | None:
    start = span.start
    end = span.end
    while start < end and document.text[start] in SPAN_TRIM_CHARS:
        start += 1
    while end > start and document.text[end - 1] in SPAN_TRIM_CHARS:
        end -= 1
    surface = document.text[start:end]
    if not surface:
        return None
    if not any(character.isalnum() for character in surface):
        return None
    if span.entity_type != "Date" and surface.isdigit():
        return None
    alpha_numeric = sum(character.isalnum() for character in surface)
    if alpha_numeric <= 1:
        return None
    if config is not None and not _passes_type_specific_cleanup(surface, span.entity_type, config):
        return None
    return PredictedMentionSpan(
        entity_type=span.entity_type,
        start=start,
        end=end,
        confidence=span.confidence,
    )


def _span_sort_score(span: PredictedMentionSpan, document: Document) -> float:
    surface = document.text[span.start : span.end]
    alpha_chars = sum(character.isalpha() for character in surface)
    punctuation_chars = sum(not character.isalnum() and not character.isspace() for character in surface)
    digit_chars = sum(character.isdigit() for character in surface)
    length_bonus = min(span.end - span.start, 48) / 160.0
    word_bonus = min(len(surface.split()), 4) * 0.03
    alpha_bonus = min(alpha_chars, 24) / 400.0
    punctuation_penalty = punctuation_chars / max(len(surface), 1) * 0.08
    digit_penalty = 0.06 if span.entity_type != "Date" and digit_chars and digit_chars >= alpha_chars else 0.0
    return float(span.confidence + length_bonus + word_bonus + alpha_bonus - punctuation_penalty - digit_penalty)


def merge_predicted_mentions(
    spans: Sequence[PredictedMentionSpan],
    document: Document,
    thresholds: Dict[str, float] | None = None,
    config: ExperimentConfig | None = None,
) -> List[Mention]:
    thresholds = thresholds or {}
    normalized_spans: List[PredictedMentionSpan] = []
    for span in spans:
        normalized = _normalize_predicted_span(span, document, config=config)
        if normalized is None or normalized.start >= normalized.end:
            continue
        if normalized.confidence < float(thresholds.get(normalized.entity_type, 0.0)):
            continue
        normalized_spans.append(normalized)

    sentence_spans = split_sentences(document.text)
    layout_spans = [tuple(block["offsets"]) for block in document.layout]
    kept: List[PredictedMentionSpan] = []
    for span in sorted(
        normalized_spans,
        key=lambda item: (-_span_sort_score(item, document), item.start, item.end, item.entity_type),
    ):
        overlapping_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if _overlaps((span.start, span.end), (existing.start, existing.end))
            ),
            None,
        )
        if overlapping_index is None:
            kept.append(span)
            continue
        existing = kept[overlapping_index]
        if _span_sort_score(span, document) > _span_sort_score(existing, document) + 1e-6:
            kept[overlapping_index] = span

    mentions = []
    for index, span in enumerate(sorted(kept, key=lambda item: (item.start, item.end, item.entity_type))):
        mentions.append(
            Mention(
                mention_id=f"P{index}",
                entity_type=span.entity_type,
                form=document.text[span.start : span.end],
                offsets=[(span.start, span.end)],
                normalizations=[],
                sentence_index=locate_span(span.start, span.end, sentence_spans),
                layout_index=locate_span(span.start, span.end, layout_spans) if layout_spans else 0,
            )
        )
    return mentions


def serialize_predicted_spans(
    spans: Sequence[PredictedMentionSpan],
    document: Document,
) -> List[Dict[str, object]]:
    return [
        {
            "entity_type": span.entity_type,
            "start": span.start,
            "end": span.end,
            "form": document.text[span.start : span.end],
            "confidence": span.confidence,
        }
        for span in spans
    ]


def combine_span_candidates(
    span_groups: Sequence[Sequence[PredictedMentionSpan]],
) -> List[PredictedMentionSpan]:
    best_by_span: Dict[tuple[str, int, int], PredictedMentionSpan] = {}
    for spans in span_groups:
        for span in spans:
            key = (span.entity_type, span.start, span.end)
            existing = best_by_span.get(key)
            if existing is None or span.confidence > existing.confidence:
                best_by_span[key] = span
    return sorted(best_by_span.values(), key=lambda item: (item.start, item.end, item.entity_type))


_DATE_REGEX_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(?:early|late|mid|since|during|from|until|by|before|after)\s+"
        r"(?:the\s+)?(?:19|20)\d{2}s?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:early|late|mid|since|during|from|until|by|before|after)\s+"
        r"the\s+(?:nineteenth|twentieth|twenty[\s-]first|twentyfirst|"
        r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
        r"seventeenth|eighteenth)\s+(?:century|decade)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:beginning|end|middle)\s+of\s+(?:the\s+)?(?:19|20)\d{2}s?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:beginning|end|middle)\s+of\s+the\s+(?:nineteenth|twentieth|"
        r"twenty[\s-]first|twentyfirst)\s+century\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|"
        r"Nov|Dec)\.?\s+(?:19|20)\d{2}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\b(?:19|20)\d{2}\s*[-/–]\s*(?:19|20)?\d{2}s?\b"),
    re.compile(r"\b(?:19|20)\d{2}s\b"),
]


def _date_regex_candidates(
    document: Document,
    confidence: float,
) -> List[PredictedMentionSpan]:
    """High-precision Date span candidates from regex.

    Conservative patterns intentionally avoid bare 4-digit years without context
    cues (a noise source we previously hit in v11's rule-based NER attempt).
    """
    text = document.text
    if not text:
        return []
    spans: List[PredictedMentionSpan] = []
    seen: set[tuple[int, int]] = set()
    for pattern in _DATE_REGEX_PATTERNS:
        for match in pattern.finditer(text):
            key = match.span()
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                PredictedMentionSpan(
                    entity_type="Date",
                    start=match.start(),
                    end=match.end(),
                    confidence=float(confidence),
                )
            )
    return spans


def build_hybrid_span_candidates(
    document: Document,
    mention_detector: "ModernBertMentionDetector",
    lexicon: MentionLexicon | None,
) -> tuple[List[PredictedMentionSpan], List[PredictedMentionSpan], List[PredictedMentionSpan]]:
    neural_spans = mention_detector.predict_span_candidates(document)
    lexicon_spans: List[PredictedMentionSpan] = []
    if lexicon is not None:
        alias_confidence = getattr(lexicon, "alias_confidence", {}) or {}
        lexicon_mentions = detect_mentions_as_mentions(document, lexicon)
        for mention in lexicon_mentions:
            for start, end in mention.offsets:
                threshold = float(mention_detector.mention_thresholds.get(mention.entity_type, 0.0))
                base_confidence = float(mention_detector.config.mention_hybrid_lexicon_confidence)
                surface_normalized = mention.form.casefold()
                external_confidence = float(alias_confidence.get(surface_normalized, 0.0))
                confidence = max(base_confidence, external_confidence, threshold + 0.05)
                lexicon_spans.append(
                    PredictedMentionSpan(
                        entity_type=mention.entity_type,
                        start=start,
                        end=end,
                        confidence=confidence,
                    )
                )
    regex_spans: List[PredictedMentionSpan] = []
    if getattr(mention_detector.config, "mention_date_regex_enabled", False):
        date_threshold = float(mention_detector.mention_thresholds.get("Date", 0.0))
        boost = float(getattr(mention_detector.config, "mention_date_regex_confidence_boost", 0.10))
        regex_confidence = max(
            float(mention_detector.config.mention_hybrid_lexicon_confidence) + boost,
            date_threshold + 0.05,
        )
        regex_spans = _date_regex_candidates(document, confidence=regex_confidence)
    blended_spans = combine_span_candidates([neural_spans, lexicon_spans, regex_spans])
    return neural_spans, lexicon_spans, blended_spans


def _passes_type_specific_cleanup(surface: str, entity_type: str, config: ExperimentConfig) -> bool:
    profile = str(getattr(config, "mention_cleanup_profile", "legacy") or "legacy")
    if profile == "legacy":
        return True
    normalized = normalize_text(surface)
    if getattr(config, "mention_type_denylists_enabled", False) and normalized in TYPE_DENYLISTS.get(entity_type, set()):
        return False
    if entity_type == "Date":
        return _looks_like_date_with_profile(surface, profile=profile)
    if entity_type == "Location":
        return _looks_like_location(surface)
    if entity_type == "Vector":
        return _looks_like_vector(surface)
    if entity_type == "Dissemination_pathway":
        return _looks_like_pathway(surface)
    return True


def _looks_like_date(surface: str) -> bool:
    return _looks_like_date_with_profile(surface, profile="strict_v1")


def _looks_like_date_with_profile(surface: str, profile: str) -> bool:
    normalized = normalize_text(surface)
    if normalized in TYPE_DENYLISTS["Date"]:
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    has_month = any(token in MONTH_TOKENS for token in tokens)
    has_century = any(token in CENTURY_TOKENS for token in tokens) or "century" in tokens
    has_year = bool(re.search(r"\b(18|19|20)\d{2}s?\b", normalized))
    has_date_digits = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", normalized)) or bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    )
    has_digit = any(character.isdigit() for character in normalized)
    has_date_cue = any(token in DATE_CUE_TOKENS for token in tokens)
    if any(token in MONTH_TOKENS for token in tokens):
        return True
    if re.fullmatch(r"(19|20)\d{2}", normalized):
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?", normalized):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        return True
    if normalized.isdigit():
        return len(normalized) == 4 and normalized.startswith(("19", "20"))
    if profile == "strict_v1":
        if len(tokens) <= 2 and any(token.isdigit() for token in tokens):
            return False
        return has_digit and has_month
    if has_year or has_date_digits:
        return True
    if has_month and len(tokens) <= 5:
        return True
    if has_century and len(tokens) <= 8:
        return True
    if has_digit and has_date_cue and len(tokens) <= 8:
        return True
    if has_month and has_date_cue and len(tokens) <= 8:
        return True
    if re.fullmatch(r"(early|late|mid)[ -]?(18|19|20)\d{2}s?", normalized):
        return True
    if len(tokens) <= 2 and any(token.isdigit() for token in tokens):
        return False
    return (has_digit or has_century) and has_date_cue and len(tokens) <= 8


def _looks_like_location(surface: str) -> bool:
    normalized = normalize_text(surface)
    tokens = normalized.split()
    if not tokens:
        return False
    if normalized in TYPE_DENYLISTS["Location"]:
        return False
    if len(tokens) == 1 and tokens[0].islower() and len(tokens[0]) <= 4:
        return False
    return True


def _looks_like_vector(surface: str) -> bool:
    normalized = normalize_text(surface)
    if normalized in TYPE_DENYLISTS["Vector"]:
        return False
    if len(normalized) <= 2:
        return False
    return any(character.isalpha() for character in normalized)


def _looks_like_pathway(surface: str) -> bool:
    normalized = normalize_text(surface)
    if normalized in TYPE_DENYLISTS["Dissemination_pathway"]:
        return False
    if len(normalized.split()) == 1 and normalized.endswith("s"):
        return False
    return len(normalized) > 3


def _normalize_alias_form(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.-]", " ", normalize_text(value))).strip()


def _punctuation_light_normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalize_text(value))).strip()


def _singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _singularize_phrase(value: str) -> str:
    return " ".join(_singularize_token(token) for token in value.split())


def _is_species_abbreviation_match(left: str, right: str) -> bool:
    left_tokens = _punctuation_light_normalize(left).split()
    right_tokens = _punctuation_light_normalize(right).split()
    if len(left_tokens) != 2 or len(right_tokens) != 2:
        return False
    if len(left_tokens[0]) != 1 or len(right_tokens[0]) < 2:
        return False
    if left_tokens[0] == right_tokens[0][0] and left_tokens[1] == right_tokens[1]:
        return True
    if len(right_tokens[0]) == 1 and len(left_tokens[0]) >= 2:
        return right_tokens[0] == left_tokens[0][0] and right_tokens[1] == left_tokens[1]
    return False


def _is_acronym_match(left: str, right: str) -> bool:
    left_surface = re.sub(r"[^A-Za-z]", "", left)
    right_tokens = [token for token in _punctuation_light_normalize(right).split() if token]
    if not left_surface or not right_tokens:
        return False
    if left_surface.isupper() and 2 <= len(left_surface) <= 8:
        initials = "".join(token[0].upper() for token in right_tokens if token[0].isalpha())
        return left_surface == initials
    right_surface = re.sub(r"[^A-Za-z]", "", right)
    left_tokens = [token for token in _punctuation_light_normalize(left).split() if token]
    if right_surface.isupper() and 2 <= len(right_surface) <= 8:
        initials = "".join(token[0].upper() for token in left_tokens if token[0].isalpha())
        return right_surface == initials
    return False


def _should_merge_predicted_mentions(left: Mention, right: Mention) -> bool:
    if left.entity_type != right.entity_type:
        return False
    left_norm = _normalize_alias_form(left.form)
    right_norm = _normalize_alias_form(right.form)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if _singularize_phrase(left_norm) == _singularize_phrase(right_norm):
        return True
    if _punctuation_light_normalize(left_norm) == _punctuation_light_normalize(right_norm):
        return True
    if _is_species_abbreviation_match(left.form, right.form):
        return True
    if _is_acronym_match(left.form, right.form):
        return True
    left_tokens = _singularize_phrase(_punctuation_light_normalize(left_norm)).split()
    right_tokens = _singularize_phrase(_punctuation_light_normalize(right_norm)).split()
    if len(left_tokens) >= 2 and len(right_tokens) >= 2 and left_tokens[-2:] == right_tokens[-2:]:
        return True
    return False


def _build_predicted_coref_clusters(mentions: Sequence[Mention], config: ExperimentConfig | None = None) -> List[List[str]]:
    strategy = str(getattr(config, "entity_alias_merge_strategy", "legacy") if config is not None else "legacy")
    if strategy == "legacy":
        return build_coref_guess(mentions)
    parent = {mention.mention_id: mention.mention_id for mention in mentions}

    def find(mention_id: str) -> str:
        while parent[mention_id] != mention_id:
            parent[mention_id] = parent[parent[mention_id]]
            mention_id = parent[mention_id]
        return mention_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(mentions):
        for right in mentions[index + 1 :]:
            if _should_merge_predicted_mentions(left, right):
                union(left.mention_id, right.mention_id)

    groups: Dict[str, List[str]] = {}
    for mention in mentions:
        groups.setdefault(find(mention.mention_id), []).append(mention.mention_id)
    return [cluster for cluster in groups.values() if len(cluster) > 1]


def tune_mention_thresholds(
    raw_spans_by_doc: Dict[str, List[PredictedMentionSpan]],
    validation_documents: Sequence[Document],
    schema: RelationSchema | None = None,
    config: ExperimentConfig | None = None,
) -> Dict[str, float]:
    strategy = str(getattr(config, "mention_threshold_tuning_strategy", "mention_f1") if config is not None else "mention_f1")
    thresholds = {entity_type: 0.0 for entity_type in ENTITY_TYPES}
    for entity_type in ENTITY_TYPES:
        candidate_thresholds = {0.0}
        for spans in raw_spans_by_doc.values():
            for span in spans:
                if span.entity_type == entity_type:
                    candidate_thresholds.add(round(float(span.confidence), 4))

        threshold_grid = _compress_threshold_grid(sorted(candidate_thresholds), strategy=strategy)

        best_threshold = 0.0
        best_score = -1.0
        best_candidate_recall = -1.0
        best_f1 = -1.0
        best_recall = -1.0
        for threshold in threshold_grid:
            trial_thresholds = dict(thresholds)
            trial_thresholds[entity_type] = float(threshold)
            predicted_mentions_by_doc = {
                document.doc_id: merge_predicted_mentions(
                    raw_spans_by_doc.get(document.doc_id, []),
                    document,
                    trial_thresholds,
                    config=config,
                )
                for document in validation_documents
            }
            metrics = compute_mention_metrics(validation_documents, predicted_mentions_by_doc, [entity_type])
            score_row = metrics["per_type"][entity_type]
            f1 = float(score_row["f1"])
            recall = float(score_row["recall"])
            candidate_recall = -1.0
            score = f1
            if strategy == "relation_aware_v1" and schema is not None and config is not None:
                predicted_entities_by_doc = {
                    document.doc_id: build_canonical_entities_from_mentions(
                        predicted_mentions_by_doc.get(document.doc_id, []),
                        config=config,
                    )
                    for document in validation_documents
                }
                candidate_pairs_by_doc = {
                    document.doc_id: enumerate_candidate_entity_pairs(
                        document,
                        predicted_entities_by_doc.get(document.doc_id, []),
                        schema,
                        config,
                    )
                    for document in validation_documents
                }
                candidate_recall_payload = compute_candidate_recall(
                    validation_documents,
                    predicted_mentions_by_doc,
                    predicted_entities_by_doc,
                    candidate_pairs_by_doc,
                )
                candidate_recall = float(candidate_recall_payload["summary"]["candidate_recall"])
                candidate_weight = float(getattr(config, "mention_threshold_candidate_recall_weight", 0.75))
                mention_weight = float(getattr(config, "mention_threshold_mention_f1_weight", 0.25))
                score = candidate_weight * candidate_recall + mention_weight * f1
            if (
                score > best_score
                or (abs(score - best_score) <= 1e-9 and candidate_recall > best_candidate_recall)
                or (abs(score - best_score) <= 1e-9 and abs(candidate_recall - best_candidate_recall) <= 1e-9 and f1 > best_f1)
                or (abs(score - best_score) <= 1e-9 and abs(candidate_recall - best_candidate_recall) <= 1e-9 and abs(f1 - best_f1) <= 1e-9 and recall > best_recall)
                or (
                    abs(score - best_score) <= 1e-9
                    and abs(candidate_recall - best_candidate_recall) <= 1e-9
                    and abs(f1 - best_f1) <= 1e-9
                    and abs(recall - best_recall) <= 1e-9
                    and threshold < best_threshold
                )
            ):
                best_threshold = float(threshold)
                best_score = score
                best_candidate_recall = candidate_recall
                best_f1 = f1
                best_recall = recall
        thresholds[entity_type] = best_threshold
    return thresholds


def _compress_threshold_grid(candidate_thresholds: Sequence[float], strategy: str, max_points: int = 25) -> List[float]:
    if strategy != "relation_aware_v1" or len(candidate_thresholds) <= max_points:
        return list(candidate_thresholds)
    if not candidate_thresholds:
        return [0.0]
    indices = np.linspace(0, len(candidate_thresholds) - 1, num=max_points, dtype=int)
    selected = sorted({float(candidate_thresholds[index]) for index in indices})
    if 0.0 not in selected:
        selected.insert(0, 0.0)
    return selected


@dataclass
class ModernBertMentionDetector:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    device: torch.device
    config: ExperimentConfig
    mention_thresholds: Dict[str, float]

    def predict_span_candidates(self, document: Document) -> List[PredictedMentionSpan]:
        if not document.text.strip():
            return []
        self.model.to(self.device)
        self.model.eval()
        encoded = self.tokenizer(
            document.text,
            truncation=True,
            max_length=resolve_mention_seq_length(self.config),
            stride=resolve_mention_doc_stride(self.config),
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        spans: List[PredictedMentionSpan] = []
        with torch.no_grad():
            for row_index in range(encoded["input_ids"].shape[0]):
                batch = {
                    "input_ids": encoded["input_ids"][row_index : row_index + 1].to(self.device),
                    "attention_mask": encoded["attention_mask"][row_index : row_index + 1].to(self.device),
                }
                logits = self.model(**batch).logits[0]
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                label_ids = probabilities.argmax(axis=-1)
                label_scores = probabilities.max(axis=-1)
                offsets = [tuple(item) for item in encoded["offset_mapping"][row_index].tolist()]
                spans.extend(
                    decode_window_predictions(
                        document.text,
                        offsets,
                        label_ids.tolist(),
                        label_scores.tolist(),
                        self.id_to_label,
                    )
                )
        return spans

    def predict_mentions_from_spans(
        self,
        spans: Sequence[PredictedMentionSpan],
        document: Document,
    ) -> List[Mention]:
        return merge_predicted_mentions(spans, document, self.mention_thresholds, config=self.config)

    def predict_mentions(self, document: Document) -> List[Mention]:
        return self.predict_mentions_from_spans(self.predict_span_candidates(document), document)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        metadata = {
            "labels": [self.id_to_label[index] for index in sorted(self.id_to_label)],
            "mention_thresholds": self.mention_thresholds,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_modernbert_mention_detector(
    train_documents: Sequence[Document],
    config: ExperimentConfig,
    validation_documents: Sequence[Document] | None = None,
    relation_schema: RelationSchema | None = None,
) -> ModernBertMentionDetector:
    labels = get_bio_labels()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = _load_tokenizer(config, encoder_role="mention")
    base_model = _load_token_classification_model(config, num_labels=len(labels))
    train_rows = build_mention_training_rows(train_documents, tokenizer, label_to_id, config)
    validation_rows = build_mention_training_rows(validation_documents or [], tokenizer, label_to_id, config)
    device = resolve_torch_device(config.device)
    training_model: torch.nn.Module = base_model
    if config.mention_use_focal_loss:
        class_weights = _build_mention_class_weights(train_rows, len(labels), config.mention_class_weight_cap)
        training_model = FocalTokenClassifier(
            base_model,
            class_weights=class_weights,
            gamma=config.mention_focal_gamma,
        )
    trained_model = _train_transformer_model(training_model, train_rows, config, device, validation_rows, stage_label="mention-detector")
    inference_model = trained_model.base_model if isinstance(trained_model, FocalTokenClassifier) else trained_model
    mention_detector = ModernBertMentionDetector(
        tokenizer=tokenizer,
        model=inference_model,  # type: ignore[arg-type]
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        device=device,
        config=config,
        mention_thresholds={entity_type: 0.0 for entity_type in ENTITY_TYPES},
    )
    if validation_documents:
        raw_spans_by_doc = {
            document.doc_id: mention_detector.predict_span_candidates(document)
            for document in validation_documents
        }
        mention_detector.mention_thresholds = tune_mention_thresholds(
            raw_spans_by_doc,
            validation_documents,
            schema=relation_schema,
            config=config,
        )
    return mention_detector


def predict_canonical_entities_with_detector(
    document: Document,
    mention_detector: ModernBertMentionDetector,
) -> List[CanonicalEntity]:
    mentions = mention_detector.predict_mentions(document)
    return build_canonical_entities_from_mentions(mentions, config=getattr(mention_detector, "config", None))


def build_canonical_entities_from_mentions(
    mentions: Sequence[Mention],
    config: ExperimentConfig | None = None,
) -> List[CanonicalEntity]:
    if not mentions:
        return []
    annotation_block = {"identity_coreferences": _build_predicted_coref_clusters(mentions, config=config)}
    return build_canonical_entities(mentions, annotation_block)


def pick_closest_role_mentions(subject: CanonicalEntity, obj: CanonicalEntity) -> Tuple[Mention, Mention]:
    best = (subject.mentions[0], obj.mentions[0])
    best_distance = float("inf")
    for subject_mention in subject.mentions:
        for object_mention in obj.mentions:
            distance = abs(subject_mention.start - object_mention.start)
            if distance < best_distance:
                best = (subject_mention, object_mention)
                best_distance = distance
    return best


def build_relation_context_text(
    document: Document,
    subject: CanonicalEntity,
    obj: CanonicalEntity,
    config: ExperimentConfig,
) -> str:
    subject_mention, object_mention = pick_closest_role_mentions(subject, obj)
    sentence_spans = split_sentences(document.text)
    start_sentence = max(0, min(subject_mention.sentence_index, object_mention.sentence_index) - config.relation_context_sentence_radius)
    end_sentence = min(len(sentence_spans) - 1, max(subject_mention.sentence_index, object_mention.sentence_index) + config.relation_context_sentence_radius)
    span_start = sentence_spans[start_sentence][0]
    span_end = sentence_spans[end_sentence][1]
    snippet = document.text[span_start:span_end]

    subject_start = subject_mention.start - span_start
    subject_end = subject_mention.end - span_start
    object_start = object_mention.start - span_start
    object_end = object_mention.end - span_start

    inserts = [
        (subject_end, f" </SUBJ_{subject.entity_type}> "),
        (subject_start, f" <SUBJ_{subject.entity_type}> "),
        (object_end, f" </OBJ_{obj.entity_type}> "),
        (object_start, f" <OBJ_{obj.entity_type}> "),
    ]
    for position, marker in sorted(inserts, key=lambda item: item[0], reverse=True):
        snippet = snippet[:position] + marker + snippet[position:]
    return snippet


DISTANT_SUPERVISION_KB = {
    "Transmits": {
        ("monochamus", "b. xylophilus"),
        ("monochamus spp.", "b. xylophilus"),
        ("monochamus galloprovincialis", "b. xylophilus"),
        ("aphid", "plum pox virus"),
        ("aphids", "plum pox virus"),
        ("whitefly", "tomato yellow leaf curl virus"),
    },
    "Causes": {
        ("xylella fastidiosa", "olive quick decline syndrome"),
        ("b. xylophilus", "pine wilt disease"),
        ("bursaphelenchus xylophilus", "pine wilt disease"),
        ("candidatus liberibacter asiaticus", "huanglongbing"),
        ("candidatus liberibacter asiaticus", "hlb"),
    },
}

def generate_relation_text_examples(
    documents: Sequence[Document],
    schema: RelationSchema,
    config: ExperimentConfig,
    entities_override: Optional[Dict[str, Sequence[CanonicalEntity]]] = None,
) -> List[Dict[str, object]]:
    examples = []
    distant_supervision_enabled = getattr(config, "relation_distant_supervision_enabled", False)
    injected_count = 0
    
    for document in documents:
        edge_lookup: Dict[tuple[str, str], set[str]] = {}
        for edge in document.gold_relation_edges:
            edge_lookup.setdefault((edge.subject, edge.object), set()).add(edge.predicate)
        entities = sorted(
            list((entities_override or {}).get(document.doc_id, document.canonical_entities)),
            key=lambda entity: entity.earliest_start,
        )
        gold_entity_lookup = _build_gold_entity_alignment_lookup(document.canonical_entities)
        for subject, obj in enumerate_candidate_entity_pairs(document, entities, schema, config):
            aligned_subject = _align_entity_to_gold(subject, gold_entity_lookup)
            aligned_object = _align_entity_to_gold(obj, gold_entity_lookup)
            labels = (
                edge_lookup.get((aligned_subject.canonical_form, aligned_object.canonical_form), set())
                if aligned_subject is not None and aligned_object is not None
                else set()
            )
            
            is_distant_supervision = False
            if distant_supervision_enabled and not labels:
                norm_sub = normalize_text(subject.canonical_form)
                norm_obj = normalize_text(obj.canonical_form)
                for relation, pairs in DISTANT_SUPERVISION_KB.items():
                    if (norm_sub, norm_obj) in pairs and schema.is_valid_pair(relation, subject.entity_type, obj.entity_type):
                        labels = {relation}
                        is_distant_supervision = True
                        injected_count += 1
                        break

            examples.append(
                {
                    "doc_id": document.doc_id,
                    "document": document,
                    "subject": subject.canonical_form,
                    "object": obj.canonical_form,
                    "subject_type": subject.entity_type,
                    "object_type": obj.entity_type,
                    "text": build_relation_context_text(document, subject, obj, config),
                    "labels": labels,
                    "subject_entity": subject,
                    "object_entity": obj,
                    "aligned_subject": aligned_subject.canonical_form if aligned_subject is not None else None,
                    "aligned_object": aligned_object.canonical_form if aligned_object is not None else None,
                    "is_distant_supervision": is_distant_supervision,
                }
            )
            
    if distant_supervision_enabled and injected_count > 0:
        print(f"Distant Supervision: Injected {injected_count} positive relation examples based on KB.")
        
    return examples


def _build_gold_entity_alignment_lookup(
    gold_entities: Sequence[CanonicalEntity],
) -> Dict[str, List[CanonicalEntity]]:
    lookup: Dict[str, List[CanonicalEntity]] = {}
    for entity in gold_entities:
        keys = {normalize_text(entity.canonical_form)}
        keys.update(normalize_text(alias) for alias in entity.alias_forms if str(alias).strip())
        for key in keys:
            lookup.setdefault(key, []).append(entity)
    return lookup


def _align_entity_to_gold(
    entity: CanonicalEntity,
    gold_entity_lookup: Dict[str, List[CanonicalEntity]],
) -> CanonicalEntity | None:
    candidate_counts: Dict[str, Tuple[int, CanonicalEntity]] = {}
    keys = {normalize_text(entity.canonical_form)}
    keys.update(normalize_text(alias) for alias in entity.alias_forms if str(alias).strip())
    for key in keys:
        for gold_entity in gold_entity_lookup.get(key, []):
            if gold_entity.entity_type != entity.entity_type:
                continue
            score, _existing = candidate_counts.get(gold_entity.entity_id, (0, gold_entity))
            candidate_counts[gold_entity.entity_id] = (score + 1, gold_entity)
    if not candidate_counts:
        return None
    return sorted(
        candidate_counts.values(),
        key=lambda item: (-item[0], item[1].earliest_start, item[1].canonical_form),
    )[0][1]


def build_relation_inference_examples(
    document: Document,
    canonical_entities: Sequence[CanonicalEntity],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, object]], List[Tuple[CanonicalEntity, CanonicalEntity]]]:
    examples = []
    entity_pairs: List[Tuple[CanonicalEntity, CanonicalEntity]] = []
    for subject, obj in enumerate_candidate_entity_pairs(document, canonical_entities, schema, config):
        examples.append(
            {
                "text": build_relation_context_text(document, subject, obj, config),
                "subject_type": subject.entity_type,
                "object_type": obj.entity_type,
            }
        )
        entity_pairs.append((subject, obj))
    return examples, entity_pairs


def mix_relation_examples(
    gold_examples: Sequence[Dict[str, object]],
    predicted_examples: Sequence[Dict[str, object]],
    predicted_mix_ratio: float,
    random_seed: int,
) -> List[Dict[str, object]]:
    mixed = list(gold_examples)
    if not predicted_examples or predicted_mix_ratio <= 0.0:
        return mixed
    maximum_predicted = max(1, int(round(len(gold_examples) * predicted_mix_ratio))) if gold_examples else len(predicted_examples)
    if len(predicted_examples) <= maximum_predicted:
        mixed.extend(predicted_examples)
        return mixed
    rng = np.random.default_rng(random_seed)
    selected_indices = sorted(rng.choice(len(predicted_examples), size=maximum_predicted, replace=False).tolist())
    mixed.extend(predicted_examples[index] for index in selected_indices)
    return mixed


def build_relation_training_rows(
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Dict[str, object]],
    labels: Sequence[str],
    config: ExperimentConfig,
) -> List[Dict[str, torch.Tensor]]:
    if not examples:
        return []
    encoded = tokenizer(
        [str(example["text"]) for example in examples],
        truncation=True,
        padding="max_length",
        max_length=config.relation_max_seq_length,
    )
    rows = []
    for row_index, example in enumerate(examples):
        label_vector = np.zeros(len(labels), dtype=np.float32)
        active_labels = set(example["labels"])
        for label_index, label in enumerate(labels):
            if label in active_labels:
                label_vector[label_index] = 1.0
        rows.append(
            {
                "input_ids": torch.tensor(encoded["input_ids"][row_index], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"][row_index], dtype=torch.long),
                "labels": torch.tensor(label_vector, dtype=torch.float32),
            }
        )
    return rows


class MultiLabelSequenceClassifier(torch.nn.Module):
    def __init__(self, base_model: PreTrainedModel, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.base_model = base_model
        self.pos_weight = pos_weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None)
            loss = loss_fn(logits, labels)
        return type("MultiLabelOutput", (), {"logits": logits, "loss": loss})()

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.base_model.parameters(recurse=recurse)

    def train(self, mode: bool = True):  # type: ignore[override]
        self.base_model.train(mode)
        return super().train(mode)

    def eval(self):  # type: ignore[override]
        self.base_model.eval()
        return super().eval()

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def save_pretrained(self, path: Path) -> None:
        self.base_model.save_pretrained(path)


@dataclass
class ModernBertRelationModel:
    labels: List[str]
    tokenizer: PreTrainedTokenizerBase
    model: MultiLabelSequenceClassifier
    thresholds: Dict[str, float]
    device: torch.device
    config: ExperimentConfig

    def predict_scores(self, examples: Sequence[Dict[str, object]]) -> np.ndarray:
        if not examples:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        encoded = self.tokenizer(
            [str(example["text"]) for example in examples],
            truncation=True,
            padding="max_length",
            max_length=self.config.relation_max_seq_length,
            return_tensors="pt",
        )
        dataset = [
            {
                "input_ids": encoded["input_ids"][row_index],
                "attention_mask": encoded["attention_mask"][row_index],
            }
            for row_index in range(encoded["input_ids"].shape[0])
        ]
        loader = DataLoader(dataset, batch_size=max(1, self.config.eval_batch_size), shuffle=False, collate_fn=_collate_batches)
        self.model.to(self.device)
        self.model.eval()
        score_rows = []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(**_move_batch_to_device(batch, self.device)).logits
                score_rows.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(score_rows, axis=0) if score_rows else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_labels(self, examples: Sequence[Dict[str, object]]) -> List[List[str]]:
        scores = self.predict_scores(examples)
        outputs = []
        for row in scores:
            outputs.append([label for label, score in zip(self.labels, row) if float(score) >= self.thresholds[label]])
        return outputs

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        payload = {"labels": self.labels, "thresholds": self.thresholds}
        (path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, config: ExperimentConfig) -> "ModernBertRelationModel":
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model_config = AutoConfig.from_pretrained(path)
        base_model = AutoModelForSequenceClassification.from_pretrained(path, config=model_config)
        wrapped_model = MultiLabelSequenceClassifier(base_model)
        return cls(
            labels=list(metadata["labels"]),
            tokenizer=tokenizer,
            model=wrapped_model,
            thresholds={key: float(value) for key, value in metadata["thresholds"].items()},
            device=resolve_torch_device(config.device),
            config=config,
        )


def augment_relation_example(example: Dict[str, object], config: ExperimentConfig) -> List[Dict[str, object]]:
    augmented = []
    document = example.get("document")
    if not isinstance(document, Document):
        return augmented

    subject = example.get("subject_entity")
    obj = example.get("object_entity")
    if not isinstance(subject, CanonicalEntity) or not isinstance(obj, CanonicalEntity):
        return augmented

    base_radius = int(config.relation_context_sentence_radius)
    for radius in [base_radius - 1, base_radius + 1]:
        if radius < 0:
            continue
        jitter_config = ExperimentConfig(relation_context_sentence_radius=radius)
        jitter_text = build_relation_context_text(document, subject, obj, jitter_config)
        if jitter_text != example["text"]:
            aug_example = example.copy()
            aug_example["text"] = jitter_text
            augmented.append(aug_example)
    return augmented


def oversample_relation_examples(
    examples: Sequence[Dict[str, object]], 
    config: ExperimentConfig,
    dynamic_hard_negative_indices: List[int] | None = None
) -> List[Dict[str, object]]:
    if not examples:
        return []
    labels = list(config.relation_labels)
    minority_labels = set(config.relation_minority_labels)
    positive_indices: Dict[str, List[int]] = {label: [] for label in labels}
    negative_indices_by_type_pair: Dict[tuple[str, str], List[int]] = {}

    for idx, example in enumerate(examples):
        active_labels = example["labels"]
        subject_type = str(example.get("subject_type", ""))
        object_type = str(example.get("object_type", ""))
        type_pair = (subject_type, object_type)

        if not active_labels:
            negative_indices_by_type_pair.setdefault(type_pair, []).append(idx)
        else:
            for label in active_labels:
                if label in positive_indices:
                    positive_indices[label].append(idx)

    counts = {label: len(indices) for label, indices in positive_indices.items()}
    if not counts:
        return list(examples)
    max_count = max(counts.values())
    target_count = int(max_count * config.relation_oversampling_ratio)

    oversampled = list(examples)
    rng = np.random.default_rng(config.random_seed)

    for label in minority_labels:
        indices = positive_indices.get(label, [])
        if not indices or len(indices) >= target_count:
            continue
        to_add = target_count - len(indices)
        selected_indices = rng.choice(indices, size=to_add, replace=True)
        
        type_pairs_added = []
        for idx in selected_indices:
            example = examples[idx]
            oversampled.append(example)
            subject_type = str(example.get("subject_type", ""))
            object_type = str(example.get("object_type", ""))
            type_pairs_added.append((subject_type, object_type))
            if config.relation_augmentation_enabled:
                oversampled.extend(augment_relation_example(example, config))
                
        # Hard Negative Mining
        if config.relation_hard_negative_ratio > 0:
            hard_negatives_to_add = int(to_add * config.relation_hard_negative_ratio)
            if hard_negatives_to_add > 0:
                if dynamic_hard_negative_indices is not None and dynamic_hard_negative_indices:
                    sampled_neg_indices = rng.choice(
                        dynamic_hard_negative_indices, 
                        size=hard_negatives_to_add, 
                        replace=len(dynamic_hard_negative_indices) < hard_negatives_to_add
                    )
                    for neg_idx in sampled_neg_indices:
                        oversampled.append(examples[neg_idx])
                elif type_pairs_added:
                    # Fallback to V14 Type-Aware Hard Negative Mining
                    candidate_negative_indices = []
                    for tp in set(type_pairs_added):
                        candidate_negative_indices.extend(negative_indices_by_type_pair.get(tp, []))
                    
                    if candidate_negative_indices:
                        sampled_neg_indices = rng.choice(
                            candidate_negative_indices, 
                            size=hard_negatives_to_add, 
                            replace=len(candidate_negative_indices) < hard_negatives_to_add
                        )
                        for neg_idx in sampled_neg_indices:
                            oversampled.append(examples[neg_idx])

    return oversampled


def train_modernbert_relation_model(
    train_examples: Sequence[Dict[str, object]],
    config: ExperimentConfig,
    calibration_examples: Sequence[Dict[str, object]] | None = None,
    validation_examples: Sequence[Dict[str, object]] | None = None,
) -> ModernBertRelationModel:
    labels = list(config.relation_labels)
    tokenizer = _load_tokenizer(config)
    marker_tokens = get_relation_marker_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})
    base_model = _load_sequence_classification_model(config, num_labels=len(labels))
    base_model.resize_token_embeddings(len(tokenizer))

    calibration_examples = list(calibration_examples or [])
    validation_examples = list(validation_examples or [])

    device = resolve_torch_device(config.device)
    dynamic_epochs = getattr(config, "relation_dynamic_hard_negative_epoch", 0)

    if dynamic_epochs > 0 and dynamic_epochs < config.epochs:
        from copy import deepcopy
        
        # Stage 1: Warmup with Type-Aware Hard Negatives
        config_stage1 = deepcopy(config)
        config_stage1.epochs = dynamic_epochs
        
        processed_train_examples = oversample_relation_examples(train_examples, config_stage1)
        train_rows = build_relation_training_rows(tokenizer, processed_train_examples, labels, config_stage1)
        validation_rows = build_relation_training_rows(tokenizer, validation_examples, labels, config_stage1)
        
        label_matrix = np.stack([row["labels"].numpy() for row in train_rows], axis=0) if train_rows else np.zeros((0, len(labels)), dtype=np.float32)
        positive_rate = label_matrix.mean(axis=0) if len(label_matrix) else np.zeros(len(labels), dtype=np.float32)
        pos_weight = np.where(positive_rate > 0, (1.0 - positive_rate) / np.maximum(positive_rate, 1e-4), 1.0).astype(np.float32)
        
        model = MultiLabelSequenceClassifier(base_model, pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        trained_model = _train_transformer_model(model, train_rows, config_stage1, device, validation_rows, stage_label="relation-stage1")

        # Dynamic Scoring of Negatives
        wrapped_temp = ModernBertRelationModel(
            labels=labels,
            tokenizer=tokenizer,
            model=trained_model,
            thresholds={label: 0.5 for label in labels},
            device=device,
            config=config,
        )
        scores = wrapped_temp.predict_scores(train_examples)
        
        minority_indices = [labels.index(m_label) for m_label in config.relation_minority_labels if m_label in labels]
        negative_scores = []
        for idx, example in enumerate(train_examples):
            if not example["labels"]:
                max_minority_score = float(np.max(scores[idx, minority_indices])) if minority_indices else float(np.max(scores[idx]))
                negative_scores.append((max_minority_score, idx))
                
        negative_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = max(100, int(len(negative_scores) * 0.2))
        dynamic_hard_negative_indices = [idx for _, idx in negative_scores[:top_k]]
        
        # Stage 2: Refinement with Dynamic Hard Negatives
        config_stage2 = deepcopy(config)
        config_stage2.epochs = config.epochs - dynamic_epochs
        
        processed_train_examples_stage2 = oversample_relation_examples(
            train_examples, config_stage2, dynamic_hard_negative_indices=dynamic_hard_negative_indices
        )
        train_rows_stage2 = build_relation_training_rows(tokenizer, processed_train_examples_stage2, labels, config_stage2)
        
        label_matrix_stage2 = np.stack([row["labels"].numpy() for row in train_rows_stage2], axis=0) if train_rows_stage2 else np.zeros((0, len(labels)), dtype=np.float32)
        positive_rate_stage2 = label_matrix_stage2.mean(axis=0) if len(label_matrix_stage2) else np.zeros(len(labels), dtype=np.float32)
        pos_weight_stage2 = np.where(positive_rate_stage2 > 0, (1.0 - positive_rate_stage2) / np.maximum(positive_rate_stage2, 1e-4), 1.0).astype(np.float32)
        trained_model.pos_weight = torch.tensor(pos_weight_stage2, dtype=torch.float32).to(device)
        
        trained_model = _train_transformer_model(trained_model, train_rows_stage2, config_stage2, device, validation_rows, stage_label="relation-stage2")

    else:
        processed_train_examples = oversample_relation_examples(train_examples, config)
        train_rows = build_relation_training_rows(tokenizer, processed_train_examples, labels, config)
        validation_rows = build_relation_training_rows(tokenizer, validation_examples, labels, config)

        label_matrix = np.stack([row["labels"].numpy() for row in train_rows], axis=0) if train_rows else np.zeros((0, len(labels)), dtype=np.float32)
        positive_rate = label_matrix.mean(axis=0) if len(label_matrix) else np.zeros(len(labels), dtype=np.float32)
        pos_weight = np.where(positive_rate > 0, (1.0 - positive_rate) / np.maximum(positive_rate, 1e-4), 1.0).astype(np.float32)

        model = MultiLabelSequenceClassifier(base_model, pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        device = resolve_torch_device(config.device)
        print(
            f"\n[relation] Building training rows | "
            f"raw_examples={len(train_examples)} | after_oversample={len(processed_train_examples)} | rows={len(train_rows)}",
            flush=True,
        )
        trained_model = _train_transformer_model(model, train_rows, config, device, validation_rows, stage_label="relation")

    wrapped = ModernBertRelationModel(
        labels=labels,
        tokenizer=tokenizer,
        model=trained_model,
        thresholds={label: config.relation_thresholds.get(label, 0.5) for label in labels},
        device=device,
        config=config,
    )
    threshold_examples = calibration_examples or list(train_examples)
    if threshold_examples:
        threshold_scores = wrapped.predict_scores(threshold_examples)
        for label_index, label in enumerate(labels):
            gold = np.array([1.0 if label in example["labels"] else 0.0 for example in threshold_examples], dtype=np.float32)
            positive_count = int(gold.sum())
            default_threshold = config.relation_thresholds.get(label, 0.5)
            min_threshold = config.relation_threshold_search_min
            max_threshold = config.relation_threshold_search_max
            if positive_count < config.relation_threshold_min_positives_for_full_tuning:
                margin = max(0.0, config.relation_threshold_low_support_margin)
                min_threshold = max(min_threshold, default_threshold - margin)
                max_threshold = min(max_threshold, default_threshold + margin)
            wrapped.thresholds[label] = calibrate_threshold(
                threshold_scores[:, label_index],
                gold,
                default_threshold,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )
    return wrapped


def predict_document_edges_with_relation_model(
    document: Document,
    canonical_entities: Sequence[CanonicalEntity],
    schema: RelationSchema,
    relation_model: ModernBertRelationModel,
    config: ExperimentConfig,
) -> List[RelationEdge]:
    examples, entity_pairs = build_relation_inference_examples(document, canonical_entities, schema, config)
    if not examples:
        return []

    edges = []
    for predicted_labels, (subject, obj) in zip(relation_model.predict_labels(examples), entity_pairs):
        for label in predicted_labels:
            if schema.is_valid_pair(label, subject.entity_type, obj.entity_type):
                edges.append(RelationEdge(subject=subject.canonical_form, predicate=label, object=obj.canonical_form))
    deduped = []
    seen = set()
    for edge in edges:
        key = (edge.subject, edge.predicate, edge.object)
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    return deduped


def predict_with_gold_entities_modernbert(
    documents: Sequence[Document],
    schema: RelationSchema,
    relation_model: ModernBertRelationModel,
    config: ExperimentConfig,
) -> Dict[str, List[RelationEdge]]:
    return {
        document.doc_id: predict_document_edges_with_relation_model(
            document,
            document.canonical_entities,
            schema,
            relation_model,
            config,
        )
        for document in documents
    }


def train_modernbert_gold_relation_pipeline(
    train_documents: Sequence[Document],
    dev_documents: Sequence[Document],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> ModernBertRelationModel:
    train_examples = generate_relation_text_examples(train_documents, schema, config)
    dev_examples = generate_relation_text_examples(dev_documents, schema, config)
    return train_modernbert_relation_model(
        train_examples,
        config,
        calibration_examples=dev_examples,
        validation_examples=dev_examples,
    )


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_tiny_encoder_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "tomato",
        "virus",
        "disease",
        "pest",
        "plant",
        "located",
        "in",
        "found",
        "on",
        "causes",
        "vector",
        "transmits",
    ]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True, "tokenizer_class": "BertTokenizerFast"}, indent=2),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "hidden_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 96,
                "max_position_embeddings": 2048,
                "vocab_size": len(vocab),
                "type_vocab_size": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
