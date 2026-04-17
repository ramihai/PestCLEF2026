from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import ExperimentConfig
from .schema import CanonicalEntity, Document, Mention


HARDCODED_RELATION_SCHEMA: Dict[str, set[Tuple[str, str]]] = {
    "Affects": {("Disease", "Plant")},
    "Causes": {("Pest", "Disease")},
    "Dispersed_by": {
        ("Disease", "Dissemination_pathway"),
        ("Pest", "Dissemination_pathway"),
    },
    "Found_on": {
        ("Pest", "Plant"),
        ("Pest", "Dissemination_pathway"),
        ("Vector", "Plant"),
        ("Vector", "Dissemination_pathway"),
    },
    "Located_in": {
        ("Disease", "Location"),
        ("Pest", "Location"),
        ("Plant", "Location"),
        ("Vector", "Location"),
    },
    "Occurs_on": {
        ("Disease", "Date"),
        ("Pest", "Date"),
        ("Plant", "Date"),
        ("Vector", "Date"),
    },
    "Transmits": {
        ("Vector", "Disease"),
        ("Vector", "Pest"),
    },
}

TOKEN_PATTERN = re.compile(r"[a-z][a-z\-]+")
RELATION_TRIGGER_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Affects": ("affect", "attack", "damage", "damage", "impact", "symptom"),
    "Causes": ("cause", "caused", "causing", "induces", "results"),
    "Dispersed_by": ("dispersed", "spread", "spreads", "via", "through"),
    "Found_on": ("found", "detected", "observed", "present", "collected"),
    "Located_in": ("in", "inside", "within", "from", "region"),
    "Occurs_on": ("during", "in", "on", "season", "year"),
    "Transmits": ("transmit", "transmitted", "vector", "carry", "carried"),
}


class RelationSchema:
    def __init__(self, allowed_pairs: Dict[str, set[Tuple[str, str]]]):
        self.allowed_pairs = allowed_pairs

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> "RelationSchema":
        del documents
        return cls.hardcoded()

    @classmethod
    def hardcoded(cls) -> "RelationSchema":
        return cls({relation: set(pairs) for relation, pairs in HARDCODED_RELATION_SCHEMA.items()})

    def compatible_relations(self, subject_type: str, object_type: str) -> List[str]:
        return [
            relation
            for relation, pairs in self.allowed_pairs.items()
            if (subject_type, object_type) in pairs
        ]

    def is_valid_pair(self, relation: str, subject_type: str, object_type: str) -> bool:
        return (subject_type, object_type) in self.allowed_pairs.get(relation, set())

    def to_serializable(self) -> Dict[str, List[Dict[str, str]]]:
        serialized = {}
        for relation, pairs in self.allowed_pairs.items():
            serialized[relation] = [
                {"subject_type": subject_type, "object_type": object_type}
                for subject_type, object_type in sorted(pairs)
            ]
        return serialized


class FeatureVectorizer:
    def __init__(self, feature_cap: int):
        self.feature_cap = feature_cap
        self.feature_to_index: Dict[str, int] = {}

    def fit(self, feature_dicts: Sequence[Dict[str, float]]) -> None:
        counts = Counter()
        for feature_dict in feature_dicts:
            counts.update(feature_dict.keys())
        most_common = counts.most_common(self.feature_cap)
        self.feature_to_index = {name: index for index, (name, _) in enumerate(most_common)}

    def transform(self, feature_dicts: Sequence[Dict[str, float]]) -> np.ndarray:
        matrix = np.zeros((len(feature_dicts), len(self.feature_to_index)), dtype=np.float32)
        for row_index, feature_dict in enumerate(feature_dicts):
            for name, value in feature_dict.items():
                index = self.feature_to_index.get(name)
                if index is not None:
                    matrix[row_index, index] = value
        return matrix


def generate_relation_examples(
    documents: Sequence[Document],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> List[Dict[str, object]]:
    examples = []
    for document in documents:
        edge_lookup = {(edge.subject, edge.object): set() for edge in document.gold_relation_edges}
        for edge in document.gold_relation_edges:
            edge_lookup[(edge.subject, edge.object)].add(edge.predicate)

        for subject, obj in enumerate_candidate_entity_pairs(document, document.canonical_entities, schema, config):
            features = extract_pair_features(document, subject, obj)
            examples.append(
                {
                    "doc_id": document.doc_id,
                    "subject": subject.canonical_form,
                    "object": obj.canonical_form,
                    "subject_type": subject.entity_type,
                    "object_type": obj.entity_type,
                    "features": features,
                    "labels": edge_lookup.get((subject.canonical_form, obj.canonical_form), set()),
                }
            )
    return examples


def should_consider_pair(
    subject: CanonicalEntity,
    obj: CanonicalEntity,
    schema: RelationSchema,
    config: ExperimentConfig,
) -> bool:
    if not schema.compatible_relations(subject.entity_type, obj.entity_type):
        return False
    sentence_gap = min_distance(subject.sentence_indices, obj.sentence_indices)
    layout_gap = min_distance(subject.layout_indices, obj.layout_indices)
    if sentence_gap > config.max_sentence_distance or layout_gap > config.max_layout_distance:
        return False
    return _passes_pair_pruning_profile(subject, obj, sentence_gap, layout_gap, config)


def enumerate_candidate_entity_pairs(
    document: Document,
    canonical_entities: Sequence[CanonicalEntity],
    schema: RelationSchema,
    config: ExperimentConfig,
) -> List[Tuple[CanonicalEntity, CanonicalEntity]]:
    del document
    pairs: List[Tuple[CanonicalEntity, CanonicalEntity]] = []
    entities = sorted(canonical_entities, key=lambda entity: entity.earliest_start)
    for subject in entities:
        for obj in entities:
            if subject.entity_id == obj.entity_id:
                continue
            if should_consider_pair(subject, obj, schema, config):
                pairs.append((subject, obj))
    return pairs


def min_distance(left: Iterable[int], right: Iterable[int]) -> int:
    left_values = list(left)
    right_values = list(right)
    return min(abs(a - b) for a in left_values for b in right_values)


def _passes_pair_pruning_profile(
    subject: CanonicalEntity,
    obj: CanonicalEntity,
    sentence_gap: int,
    layout_gap: int,
    config: ExperimentConfig,
) -> bool:
    if config.model_name != "modernbert_staged":
        return True
    profile = str(getattr(config, "relation_pair_pruning_profile", "legacy") or "legacy")
    if profile == "legacy":
        return True
    subject_type = subject.entity_type
    object_type = obj.entity_type
    object_name = obj.canonical_form.strip()

    if profile == "precision_v1":
        if (subject_type, object_type) == ("Disease", "Date") or (subject_type, object_type) == ("Pest", "Date") or (subject_type, object_type) == ("Plant", "Date") or (subject_type, object_type) == ("Vector", "Date"):
            if sentence_gap > 1 or layout_gap > 2:
                return False
            if len(object_name) <= 2:
                return False
        if (subject_type, object_type) == ("Disease", "Location") or (subject_type, object_type) == ("Pest", "Location") or (subject_type, object_type) == ("Plant", "Location") or (subject_type, object_type) == ("Vector", "Location"):
            if sentence_gap > 2 or layout_gap > 4:
                return False
            if not object_name:
                return False
        return True
    return True


def extract_pair_features(document: Document, subject: CanonicalEntity, obj: CanonicalEntity) -> Dict[str, float]:
    sentence_gap = min_distance(subject.sentence_indices, obj.sentence_indices)
    layout_gap = min_distance(subject.layout_indices, obj.layout_indices)
    subject_first = 1.0 if subject.earliest_start <= obj.earliest_start else 0.0
    surface_gap = abs(subject.earliest_start - obj.earliest_start)
    left_mention, right_mention = closest_mention_pair(subject, obj)
    between_tokens = extract_span_tokens(document.text, left_mention.end, right_mention.start, limit=10)
    context_tokens = extract_context_tokens(document.text, left_mention.start, right_mention.end)
    subject_head_tokens = extract_name_tokens(subject.canonical_form)
    object_head_tokens = extract_name_tokens(obj.canonical_form)
    pair_mention_count = len(subject.mentions) * len(obj.mentions)
    features: Dict[str, float] = {
        f"subject_type={subject.entity_type}": 1.0,
        f"object_type={obj.entity_type}": 1.0,
        f"type_pair={subject.entity_type}->{obj.entity_type}": 1.0,
        f"sentence_gap={min(sentence_gap, 5)}": 1.0,
        f"layout_gap={min(layout_gap, 8)}": 1.0,
        f"subject_first={int(subject_first)}": 1.0,
        f"subject_mentions={min(len(subject.mentions), 4)}": 1.0,
        f"object_mentions={min(len(obj.mentions), 4)}": 1.0,
        f"same_sentence={int(sentence_gap == 0)}": 1.0,
        f"same_layout={int(layout_gap == 0)}": 1.0,
        f"surface_bucket={bucketize_distance(surface_gap)}": 1.0,
        f"between_bucket={bucketize_distance(max(right_mention.start - left_mention.end, 0))}": 1.0,
        f"pair_mentions={bucketize_count(pair_mention_count)}": 1.0,
        f"subject_name_len={bucketize_count(len(subject.canonical_form.split()))}": 1.0,
        f"object_name_len={bucketize_count(len(obj.canonical_form.split()))}": 1.0,
        f"subject_name={subject.canonical_form.casefold()}": 1.0,
        f"object_name={obj.canonical_form.casefold()}": 1.0,
    }
    for token in context_tokens:
        features[f"context={token}"] = 1.0
    for token in between_tokens:
        features[f"between={token}"] = 1.0
    for token in subject_head_tokens:
        features[f"subject_token={token}"] = 1.0
    for token in object_head_tokens:
        features[f"object_token={token}"] = 1.0
    for relation, keywords in RELATION_TRIGGER_KEYWORDS.items():
        if any(token in context_tokens or token in between_tokens for token in keywords):
            features[f"trigger_hint={relation}"] = 1.0
    return features


def bucketize_distance(distance: int) -> str:
    if distance < 64:
        return "<64"
    if distance < 256:
        return "<256"
    if distance < 1024:
        return "<1024"
    return ">=1024"


def bucketize_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    if value <= 4:
        return "3_4"
    return "5+"


def closest_mention_pair(subject: CanonicalEntity, obj: CanonicalEntity) -> Tuple[Mention, Mention]:
    best_pair = (subject.mentions[0], obj.mentions[0])
    best_distance = float("inf")
    for subject_mention in subject.mentions:
        for object_mention in obj.mentions:
            distance = abs(subject_mention.start - object_mention.start)
            if distance < best_distance:
                best_distance = distance
                if subject_mention.start <= object_mention.start:
                    best_pair = (subject_mention, object_mention)
                else:
                    best_pair = (object_mention, subject_mention)
    return best_pair


def extract_context_tokens(text: str, first_start: int, second_end: int, window: int = 100) -> List[str]:
    left = min(first_start, second_end)
    right = max(first_start, second_end)
    start = max(left - window, 0)
    end = min(right + window, len(text))
    tokens = tokenize_text(text[start:end])
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(8)]


def extract_span_tokens(text: str, start: int, end: int, limit: int = 10) -> List[str]:
    if end <= start:
        return []
    counts = Counter(tokenize_text(text[start:end]))
    return [token for token, _ in counts.most_common(limit)]


def extract_name_tokens(value: str) -> List[str]:
    return tokenize_text(value)[:4]


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.casefold())
