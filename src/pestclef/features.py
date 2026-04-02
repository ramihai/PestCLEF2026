from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import ExperimentConfig
from .schema import CanonicalEntity, Document


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

        entities = sorted(document.canonical_entities, key=lambda entity: entity.earliest_start)
        for subject in entities:
            for obj in entities:
                if subject.entity_id == obj.entity_id:
                    continue
                if not should_consider_pair(subject, obj, schema, config):
                    continue
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
    return sentence_gap <= config.max_sentence_distance and layout_gap <= config.max_layout_distance


def min_distance(left: Iterable[int], right: Iterable[int]) -> int:
    left_values = list(left)
    right_values = list(right)
    return min(abs(a - b) for a in left_values for b in right_values)


def extract_pair_features(document: Document, subject: CanonicalEntity, obj: CanonicalEntity) -> Dict[str, float]:
    sentence_gap = min_distance(subject.sentence_indices, obj.sentence_indices)
    layout_gap = min_distance(subject.layout_indices, obj.layout_indices)
    subject_first = 1.0 if subject.earliest_start <= obj.earliest_start else 0.0
    surface_gap = abs(subject.earliest_start - obj.earliest_start)
    context_tokens = extract_context_tokens(document.text, subject.earliest_start, obj.earliest_start)
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
        f"subject_name={subject.canonical_form.casefold()}": 1.0,
        f"object_name={obj.canonical_form.casefold()}": 1.0,
    }
    for token in context_tokens:
        features[f"context={token}"] = 1.0
    return features


def bucketize_distance(distance: int) -> str:
    if distance < 64:
        return "<64"
    if distance < 256:
        return "<256"
    if distance < 1024:
        return "<1024"
    return ">=1024"


def extract_context_tokens(text: str, first_start: int, second_start: int, window: int = 100) -> List[str]:
    left = min(first_start, second_start)
    right = max(first_start, second_start)
    start = max(left - window, 0)
    end = min(right + window, len(text))
    snippet = text[start:end].casefold()
    tokens = [token for token in snippet.replace("\n", " ").split() if token.isalpha()]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(8)]
