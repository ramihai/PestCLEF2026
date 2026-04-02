from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Mention:
    mention_id: str
    entity_type: str
    form: str
    offsets: List[Tuple[int, int]]
    normalizations: List[Dict[str, str]]
    sentence_index: int
    layout_index: int

    @property
    def start(self) -> int:
        return self.offsets[0][0]

    @property
    def end(self) -> int:
        return self.offsets[-1][1]


@dataclass
class CanonicalEntity:
    entity_id: str
    entity_type: str
    mentions: List[Mention]
    canonical_form: str
    normalization_name: Optional[str]
    alias_forms: Set[str]

    @property
    def earliest_start(self) -> int:
        return min(mention.start for mention in self.mentions)

    @property
    def sentence_indices(self) -> List[int]:
        return sorted({mention.sentence_index for mention in self.mentions})

    @property
    def layout_indices(self) -> List[int]:
        return sorted({mention.layout_index for mention in self.mentions})


@dataclass(frozen=True)
class RelationEdge:
    subject: str
    predicate: str
    object: str


@dataclass
class Document:
    doc_id: str
    split: str
    text: str
    layout: List[Dict[str, object]]
    mentions: List[Mention] = field(default_factory=list)
    canonical_entities: List[CanonicalEntity] = field(default_factory=list)
    gold_relation_edges: List[RelationEdge] = field(default_factory=list)
    gold_alias_edges: List[Dict[str, object]] = field(default_factory=list)
    unresolved_gold_edges: List[Dict[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
