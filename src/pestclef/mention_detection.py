from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import ExperimentConfig
from .data import build_canonical_entities, build_mentions, normalize_text
from .schema import CanonicalEntity, Document, Mention

# Maximum aliases before we switch from per-alias loops to the fast-path.
# Above this threshold the per-alias approach is too slow (O(n_aliases × text_len)).
_FAST_PATH_THRESHOLD = 500


def _build_automaton(aliases: List[str]):
    """Build an Aho-Corasick automaton for O(n_text) multi-pattern search.

    Aliases are expected to be already casefolded.  Falls back to ``None``
    if ``pyahocorasick`` is not installed.
    """
    try:
        import ahocorasick  # type: ignore[import]
    except ImportError:
        return None
    A = ahocorasick.Automaton()
    for alias in aliases:
        A.add_word(alias, alias)
    A.make_automaton()
    return A


@dataclass
class MentionLexicon:
    alias_to_type: Dict[str, str]
    aliases: List[str]
    alias_confidence: Dict[str, float] = None  # type: ignore[assignment]
    # Aho-Corasick automaton for large lexicons (set in __post_init__).
    _automaton: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.alias_confidence is None:
            self.alias_confidence = {}
        if len(self.aliases) > _FAST_PATH_THRESHOLD:
            self._automaton = _build_automaton(self.aliases)
        else:
            self._automaton = None

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
        external = _load_external_lexicon(
            getattr(config, "lexicon_external_path", None),
            disabled_types=set(getattr(config, "lexicon_external_disabled_types", []) or []),
            min_alias_length=config.min_alias_length,
        )
        external_confidence = float(getattr(config, "lexicon_external_confidence", 0.0) or 0.0)
        alias_confidence: Dict[str, float] = {}
        for alias, entity_type in external.items():
            # train-derived counts dominate; only add alias if not already typed differently with higher support
            if alias not in alias_counts:
                alias_counts[alias][entity_type] += 1
                if external_confidence > 0:
                    alias_confidence[alias] = external_confidence
        alias_to_type = {
            alias: counts.most_common(1)[0][0]
            for alias, counts in alias_counts.items()
        }
        aliases = sorted(alias_to_type.keys(), key=lambda alias: (-len(alias), alias))
        return cls(alias_to_type=alias_to_type, aliases=aliases, alias_confidence=alias_confidence)


def _load_external_lexicon(
    path: Optional[str],
    disabled_types: Optional[set] = None,
    min_alias_length: int = 3,
) -> Dict[str, str]:
    """Load an external lexicon JSON file mapping {entity_type: [alias, ...]}.

    Casefolds aliases and skips entries below ``min_alias_length``. Returns a
    flat dict alias -> entity_type. Aliases conflicting between types are
    skipped (we leave conflict resolution to the train-derived counts).
    """
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    disabled = disabled_types or set()
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    flat: Dict[str, str] = {}
    conflicts: set = set()
    for entity_type, aliases in raw.items():
        if not isinstance(entity_type, str) or entity_type in disabled:
            continue
        if not isinstance(aliases, (list, tuple)):
            continue
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            normalized = normalize_text(alias)
            if len(normalized) < min_alias_length:
                continue
            if normalized in conflicts:
                continue
            existing = flat.get(normalized)
            if existing is not None and existing != entity_type:
                conflicts.add(normalized)
                flat.pop(normalized, None)
                continue
            flat[normalized] = entity_type
    return flat


def _iter_automaton_matches(
    text_lower: str, automaton
) -> Iterable[tuple[int, int, str]]:
    """Yield (start, end, alias) for every word-boundary match in *text_lower*.

    The Aho-Corasick ``iter()`` method returns ``(end_index, alias)`` where
    ``end_index`` is the index of the *last* character of the match.  We convert
    to half-open ``(start, end)`` and enforce ``\\b``-style word boundaries.
    """
    n = len(text_lower)
    for end_idx, alias in automaton.iter(text_lower):
        end = end_idx + 1
        start = end - len(alias)
        # Word-boundary check: character before start and after end must not be \w
        if start > 0 and text_lower[start - 1].isalnum():
            continue
        if end < n and text_lower[end].isalnum():
            continue
        yield start, end, alias


def detect_mentions(document: Document, lexicon: MentionLexicon) -> List[Mention]:
    text_lower = document.text.casefold()
    occupied: List[tuple[int, int]] = []
    mentions: List[Mention] = []
    sentence_spans = [(mention.start, mention.end) for mention in build_mentions_cache(document)]
    if not sentence_spans:
        sentence_spans = [(0, len(document.text))]
    mention_index = 0

    if lexicon._automaton is not None:
        raw_matches: List[tuple[int, int, str]] = sorted(
            _iter_automaton_matches(text_lower, lexicon._automaton),
            key=lambda m: (m[0], -(m[1] - m[0])),  # left-to-right, longest first
        )
        for start, end, alias in raw_matches:
            if overlaps_existing(start, end, occupied):
                continue
            entity_type = lexicon.alias_to_type.get(alias)
            if entity_type is None:
                continue
            occupied.append((start, end))
            sentence_index = locate_from_mentions(start, end, document, sentence_spans)
            layout_index = locate_layout(start, end, document)
            mentions.append(
                Mention(
                    mention_id=f"P{mention_index}",
                    entity_type=entity_type,
                    form=document.text[start:end],
                    offsets=[(start, end)],
                    normalizations=[],
                    sentence_index=sentence_index,
                    layout_index=layout_index,
                )
            )
            mention_index += 1
    else:
        for alias in lexicon.aliases:
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                if overlaps_existing(start, end, occupied):
                    continue
                occupied.append((start, end))
                sentence_index = locate_from_mentions(start, end, document, sentence_spans)
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

    if lexicon._automaton is not None:
        # Fast path: Aho-Corasick O(n_text) scan, one pass over the document.
        raw_matches: List[tuple[int, int, str]] = sorted(
            _iter_automaton_matches(text_lower, lexicon._automaton),
            key=lambda m: (m[0], -(m[1] - m[0])),  # left-to-right, longest first
        )
        for start, end, alias in raw_matches:
            if overlaps_existing(start, end, occupied):
                continue
            entity_type = lexicon.alias_to_type.get(alias)
            if entity_type is None:
                continue
            occupied.append((start, end))
            sentence_index = locate_in_spans(start, end, sentence_spans)
            layout_index = locate_layout(start, end, document)
            mentions.append(
                Mention(
                    mention_id=f"P{mention_index}",
                    entity_type=entity_type,
                    form=document.text[start:end],
                    offsets=[(start, end)],
                    normalizations=[],
                    sentence_index=sentence_index,
                    layout_index=layout_index,
                )
            )
            mention_index += 1
    else:
        # Small-lexicon path: original per-alias loop (correct & fast enough for
        # lexicons with <= _FAST_PATH_THRESHOLD aliases).
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
