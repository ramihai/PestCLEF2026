from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from .data import build_canonical_entities, locate_span, split_sentences
from .evaluation import compute_mention_metrics
from .features import RelationSchema, should_consider_pair
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


def resolve_torch_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


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


def _load_tokenizer(config: ExperimentConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        config.encoder_name,
        use_fast=True,
        local_files_only=config.local_files_only,
    )


def _load_model_config(config: ExperimentConfig, num_labels: int, task_type: str) -> AutoConfig:
    model_config = AutoConfig.from_pretrained(
        config.encoder_name,
        local_files_only=config.local_files_only,
    )
    model_config.num_labels = num_labels
    if task_type == "relation":
        model_config.problem_type = "multi_label_classification"
    return model_config


def _load_token_classification_model(config: ExperimentConfig, num_labels: int) -> PreTrainedModel:
    model_config = _load_model_config(config, num_labels, task_type="mention")
    if config.encoder_random_init:
        return AutoModelForTokenClassification.from_config(model_config)
    return AutoModelForTokenClassification.from_pretrained(
        config.encoder_name,
        config=model_config,
        local_files_only=config.local_files_only,
    )


def _load_sequence_classification_model(config: ExperimentConfig, num_labels: int) -> PreTrainedModel:
    model_config = _load_model_config(config, num_labels, task_type="relation")
    if config.encoder_random_init:
        return AutoModelForSequenceClassification.from_config(model_config)
    return AutoModelForSequenceClassification.from_pretrained(
        config.encoder_name,
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
) -> torch.nn.Module:
    if not train_rows:
        return model.to(device)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate_encoder, weight_decay=config.weight_decay)
    loader = DataLoader(
        _stack_feature_rows(train_rows),
        batch_size=max(1, config.train_batch_size),
        shuffle=True,
        collate_fn=_collate_batches,
    )
    total_updates_per_epoch = max(1, (len(loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
    total_steps = max(1, total_updates_per_epoch * max(1, config.epochs))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_state_dict: Dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    patience = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(max(1, config.epochs)):
        model.train()
        for step, batch in enumerate(loader, start=1):
            outputs = model(**_move_batch_to_device(batch, device))
            loss = outputs.loss / max(1, config.gradient_accumulation_steps)
            loss.backward()
            if step % max(1, config.gradient_accumulation_steps) == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if validation_rows:
            validation_loss = _evaluate_dataset(model, validation_rows, config.eval_batch_size, device)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state_dict = _clone_state_dict(model)
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

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
        max_length=config.max_seq_length,
        stride=config.doc_stride,
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


def _normalize_predicted_span(span: PredictedMentionSpan, document: Document) -> PredictedMentionSpan | None:
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
) -> List[Mention]:
    thresholds = thresholds or {}
    normalized_spans: List[PredictedMentionSpan] = []
    for span in spans:
        normalized = _normalize_predicted_span(span, document)
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


def build_hybrid_span_candidates(
    document: Document,
    mention_detector: "ModernBertMentionDetector",
    lexicon: MentionLexicon | None,
) -> tuple[List[PredictedMentionSpan], List[PredictedMentionSpan], List[PredictedMentionSpan]]:
    neural_spans = mention_detector.predict_span_candidates(document)
    lexicon_spans: List[PredictedMentionSpan] = []
    if lexicon is not None:
        lexicon_mentions = detect_mentions_as_mentions(document, lexicon)
        for mention in lexicon_mentions:
            for start, end in mention.offsets:
                threshold = float(mention_detector.mention_thresholds.get(mention.entity_type, 0.0))
                confidence = max(float(mention_detector.config.mention_hybrid_lexicon_confidence), threshold + 0.05)
                lexicon_spans.append(
                    PredictedMentionSpan(
                        entity_type=mention.entity_type,
                        start=start,
                        end=end,
                        confidence=confidence,
                    )
                )
    blended_spans = combine_span_candidates([neural_spans, lexicon_spans])
    return neural_spans, lexicon_spans, blended_spans


def tune_mention_thresholds(
    raw_spans_by_doc: Dict[str, List[PredictedMentionSpan]],
    validation_documents: Sequence[Document],
) -> Dict[str, float]:
    thresholds = {entity_type: 0.0 for entity_type in ENTITY_TYPES}
    for entity_type in ENTITY_TYPES:
        candidate_thresholds = {0.0}
        for spans in raw_spans_by_doc.values():
            for span in spans:
                if span.entity_type == entity_type:
                    candidate_thresholds.add(round(float(span.confidence), 4))

        best_threshold = 0.0
        best_f1 = -1.0
        best_recall = -1.0
        for threshold in sorted(candidate_thresholds):
            trial_thresholds = dict(thresholds)
            trial_thresholds[entity_type] = float(threshold)
            predicted_mentions_by_doc = {
                document.doc_id: merge_predicted_mentions(
                    raw_spans_by_doc.get(document.doc_id, []),
                    document,
                    trial_thresholds,
                )
                for document in validation_documents
            }
            metrics = compute_mention_metrics(validation_documents, predicted_mentions_by_doc, [entity_type])
            score_row = metrics["per_type"][entity_type]
            f1 = float(score_row["f1"])
            recall = float(score_row["recall"])
            if (
                f1 > best_f1
                or (abs(f1 - best_f1) <= 1e-9 and recall > best_recall)
                or (abs(f1 - best_f1) <= 1e-9 and abs(recall - best_recall) <= 1e-9 and threshold < best_threshold)
            ):
                best_threshold = float(threshold)
                best_f1 = f1
                best_recall = recall
        thresholds[entity_type] = best_threshold
    return thresholds


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
            max_length=self.config.max_seq_length,
            stride=self.config.doc_stride,
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
        return merge_predicted_mentions(spans, document, self.mention_thresholds)

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
) -> ModernBertMentionDetector:
    labels = get_bio_labels()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = _load_tokenizer(config)
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
    trained_model = _train_transformer_model(training_model, train_rows, config, device, validation_rows)
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
        mention_detector.mention_thresholds = tune_mention_thresholds(raw_spans_by_doc, validation_documents)
    return mention_detector


def predict_canonical_entities_with_detector(
    document: Document,
    mention_detector: ModernBertMentionDetector,
) -> List[CanonicalEntity]:
    mentions = mention_detector.predict_mentions(document)
    return build_canonical_entities_from_mentions(mentions)


def build_canonical_entities_from_mentions(mentions: Sequence[Mention]) -> List[CanonicalEntity]:
    if not mentions:
        return []
    annotation_block = {"identity_coreferences": build_coref_guess(mentions)}
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


def generate_relation_text_examples(
    documents: Sequence[Document],
    schema: RelationSchema,
    config: ExperimentConfig,
    entities_override: Optional[Dict[str, Sequence[CanonicalEntity]]] = None,
) -> List[Dict[str, object]]:
    examples = []
    for document in documents:
        edge_lookup: Dict[tuple[str, str], set[str]] = {}
        for edge in document.gold_relation_edges:
            edge_lookup.setdefault((edge.subject, edge.object), set()).add(edge.predicate)
        entities = sorted(
            list((entities_override or {}).get(document.doc_id, document.canonical_entities)),
            key=lambda entity: entity.earliest_start,
        )
        for subject in entities:
            for obj in entities:
                if subject.entity_id == obj.entity_id:
                    continue
                if not should_consider_pair(subject, obj, schema, config):
                    continue
                examples.append(
                    {
                        "doc_id": document.doc_id,
                        "subject": subject.canonical_form,
                        "object": obj.canonical_form,
                        "subject_type": subject.entity_type,
                        "object_type": obj.entity_type,
                        "text": build_relation_context_text(document, subject, obj, config),
                        "labels": edge_lookup.get((subject.canonical_form, obj.canonical_form), set()),
                        "subject_entity": subject,
                        "object_entity": obj,
                    }
                )
    return examples


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
    train_rows = build_relation_training_rows(tokenizer, train_examples, labels, config)
    validation_rows = build_relation_training_rows(tokenizer, validation_examples, labels, config)
    label_matrix = np.stack([row["labels"].numpy() for row in train_rows], axis=0) if train_rows else np.zeros((0, len(labels)), dtype=np.float32)
    positive_rate = label_matrix.mean(axis=0) if len(label_matrix) else np.zeros(len(labels), dtype=np.float32)
    pos_weight = np.where(positive_rate > 0, (1.0 - positive_rate) / np.maximum(positive_rate, 1e-4), 1.0).astype(np.float32)
    model = MultiLabelSequenceClassifier(base_model, pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    device = resolve_torch_device(config.device)
    trained_model = _train_transformer_model(model, train_rows, config, device, validation_rows)

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
            wrapped.thresholds[label] = calibrate_threshold(
                threshold_scores[:, label_index],
                gold,
                config.relation_thresholds.get(label, 0.5),
            )
    return wrapped


def predict_document_edges_with_relation_model(
    document: Document,
    canonical_entities: Sequence[CanonicalEntity],
    schema: RelationSchema,
    relation_model: ModernBertRelationModel,
    config: ExperimentConfig,
) -> List[RelationEdge]:
    examples = []
    entity_pairs: List[Tuple[CanonicalEntity, CanonicalEntity]] = []
    entities = sorted(canonical_entities, key=lambda entity: entity.earliest_start)
    for subject in entities:
        for obj in entities:
            if subject.entity_id == obj.entity_id:
                continue
            if not should_consider_pair(subject, obj, schema, config):
                continue
            examples.append(
                {
                    "text": build_relation_context_text(document, subject, obj, config),
                    "subject_type": subject.entity_type,
                    "object_type": obj.entity_type,
                }
            )
            entity_pairs.append((subject, obj))
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
