from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .config import ExperimentConfig
from .data import build_canonical_entities, build_mentions, normalize_text
from .schema import CanonicalEntity, Document, Mention


@dataclass
class MentionLexicon:
    alias_to_type: Dict[str, str]
    aliases: List[str]

    @classmethod
    def from_documents(cls, documents: Sequence[Document], config: ExperimentConfig) -> "MentionLexicon":
        alias_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for document in documents:
            for entity in document.canonical_entities:
                for alias in entity.alias_forms:
                    if len(alias) >= config.min_alias_length:
                        alias_counts[alias][entity.entity_type] += 1
                normalized_name = normalize_text(entity.canonical_form)
                if len(normalized_name) >= config.min_alias_length:
                    alias_counts[normalized_name][entity.entity_type] += 1
        alias_to_type = {
            alias: counts.most_common(1)[0][0]
            for alias, counts in alias_counts.items()
        }
        aliases = sorted(alias_to_type.keys(), key=lambda alias: (-len(alias), alias))
        return cls(alias_to_type=alias_to_type, aliases=aliases)


def detect_mentions(document: Document, lexicon: MentionLexicon) -> List[Mention]:
    text_lower = document.text.casefold()
    occupied: List[tuple[int, int]] = []
    mentions: List[Mention] = []
    sentence_spans = [(mention.start, mention.end) for mention in build_mentions_cache(document)]
    if not sentence_spans:
        sentence_spans = [(0, len(document.text))]
    mention_index = 0
    for alias in lexicon.aliases:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(text_lower):
            start, end = match.span()
            if overlaps_existing(start, end, occupied):
                continue
            occupied.append((start, end))
            sentence_index = locate_from_mentions(start, end, document, sentence_spans)
            layout_index = locate_layout(start, end, document)
            surface = document.text[start:end]
            mentions.append(
                Mention(
                    mention_id=f"P{mention_index}",
                    entity_type=lexicon.alias_to_type[alias],
                    form=surface,
                    offsets=[(start, end)],
                    normalizations=[],
                    sentence_index=sentence_index,
                    layout_index=layout_index,
                )
            )
            mention_index += 1
    mentions.sort(key=lambda mention: (mention.start, mention.end))
    if not mentions:
        return []
    annotation_block = {
        "identity_coreferences": build_coref_guess(mentions),
    }
    return build_canonical_entities(mentions, annotation_block)  # type: ignore[return-value]


def predict_canonical_entities(document: Document, lexicon: MentionLexicon) -> List[CanonicalEntity]:
    detected_mentions = detect_mentions_as_mentions(document, lexicon)
    annotation_block = {
        "identity_coreferences": build_coref_guess(detected_mentions),
    }
    return build_canonical_entities(detected_mentions, annotation_block)


def detect_mentions_as_mentions(document: Document, lexicon: MentionLexicon) -> List[Mention]:
    text_lower = document.text.casefold()
    occupied: List[tuple[int, int]] = []
    mentions: List[Mention] = []
    sentence_spans = build_sentence_spans(document)
    mention_index = 0
    for alias in lexicon.aliases:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(text_lower):
            start, end = match.span()
            if overlaps_existing(start, end, occupied):
                continue
            occupied.append((start, end))
            sentence_index = locate_in_spans(start, end, sentence_spans)
            layout_index = locate_layout(start, end, document)
            mentions.append(
                Mention(
                    mention_id=f"P{mention_index}",
                    entity_type=lexicon.alias_to_type[alias],
                    form=document.text[start:end],
                    offsets=[(start, end)],
                    normalizations=[],
                    sentence_index=sentence_index,
                    layout_index=layout_index,
                )
            )
            mention_index += 1
    mentions.sort(key=lambda mention: (mention.start, mention.end))
    return mentions


def build_sentence_spans(document: Document) -> List[tuple[int, int]]:
    if document.mentions:
        sentence_map = defaultdict(list)
        for mention in document.mentions:
            sentence_map[mention.sentence_index].append((mention.start, mention.end))
        spans = []
        for index in sorted(sentence_map):
            starts, ends = zip(*sentence_map[index])
            spans.append((min(starts), max(ends)))
        if spans:
            return spans
    text = document.text
    if not text:
        return [(0, 0)]
    spans = []
    start = 0
    for match in re.finditer(r"[.!?]\s+", text):
        end = match.end()
        spans.append((start, end))
        start = end
    spans.append((start, len(text)))
    return spans


def build_coref_guess(mentions: Sequence[Mention]) -> List[List[str]]:
    buckets: Dict[tuple[str, str], List[str]] = defaultdict(list)
    for mention in mentions:
        key = (mention.entity_type, normalize_text(mention.form))
        buckets[key].append(mention.mention_id)
    return [cluster for cluster in buckets.values() if len(cluster) > 1]


def overlaps_existing(start: int, end: int, occupied: Iterable[tuple[int, int]]) -> bool:
    for existing_start, existing_end in occupied:
        if start < existing_end and end > existing_start:
            return True
    return False


def locate_in_spans(start: int, end: int, spans: Sequence[tuple[int, int]]) -> int:
    for index, (span_start, span_end) in enumerate(spans):
        if start >= span_start and end <= span_end:
            return index
    return 0


def locate_layout(start: int, end: int, document: Document) -> int:
    for index, block in enumerate(document.layout):
        block_start, block_end = block["offsets"]
        if start >= block_start and end <= block_end:
            return index
    return 0


def build_mentions_cache(document: Document) -> List[Mention]:
    return list(document.mentions)


def locate_from_mentions(start: int, end: int, document: Document, spans: Sequence[tuple[int, int]]) -> int:
    return locate_in_spans(start, end, spans or build_sentence_spans(document))
