from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import ExperimentConfig
from .schema import CanonicalEntity, Document, Mention, RelationEdge


TOKEN_NORMALIZER = re.compile(r"\s+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_text(value: str) -> str:
    cleaned = TOKEN_NORMALIZER.sub(" ", value.strip())
    return cleaned.casefold()


def load_documents(split: str, config: ExperimentConfig) -> List[Document]:
    path = config.json_dir / f"PestCLEF-2026_{split}.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    documents: List[Document] = []
    for item in payload:
        text_path = config.docs_dir / split / f"{item['doc_id']}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        document = Document(
            doc_id=str(item["doc_id"]),
            split=split,
            text=text,
            layout=item.get("layout", []),
        )

        if "text_bound_annotations" in item and text:
            document.mentions = build_mentions(item["text_bound_annotations"], text, document.layout)
            document.canonical_entities = build_canonical_entities(document.mentions, item["text_bound_annotations"])
        if "knowledge_graph" in item:
            document.gold_alias_edges = item["knowledge_graph"]
            if document.canonical_entities:
                aligned_edges, unresolved = align_gold_edges(item["knowledge_graph"], document.canonical_entities)
                document.gold_relation_edges = aligned_edges
                document.unresolved_gold_edges = unresolved
            else:
                document.gold_relation_edges = [
                    RelationEdge(
                        subject=edge["subject"][0] if isinstance(edge["subject"], list) else edge["subject"],
                        predicate=edge["predicate"],
                        object=edge["object"][0] if isinstance(edge["object"], list) else edge["object"],
                    )
                    for edge in item["knowledge_graph"]
                ]
        documents.append(document)
    return documents


def build_mentions(annotation_block: Dict[str, object], text: str, layout: Sequence[Dict[str, object]]) -> List[Mention]:
    sentence_spans = split_sentences(text)
    mentions: List[Mention] = []
    for entity in annotation_block.get("entities", []):
        start = entity["offsets"][0][0]
        end = entity["offsets"][-1][1]
        mention_text = " ".join(text[a:b] for a, b in entity["offsets"])
        sentence_index = locate_span(start, end, sentence_spans)
        layout_index = locate_span(start, end, [tuple(block["offsets"]) for block in layout]) if layout else 0
        mentions.append(
            Mention(
                mention_id=entity["id"],
                entity_type=entity["type"],
                form=mention_text or entity["form"],
                offsets=[tuple(offset) for offset in entity["offsets"]],
                normalizations=entity.get("normalizations", []),
                sentence_index=sentence_index,
                layout_index=layout_index,
            )
        )
    mentions.sort(key=lambda mention: (mention.start, mention.end))
    return mentions


def build_canonical_entities(mentions: Sequence[Mention], annotation_block: Dict[str, object]) -> List[CanonicalEntity]:
    parent = {mention.mention_id: mention.mention_id for mention in mentions}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for cluster in annotation_block.get("identity_coreferences", []):
        if not cluster:
            continue
        root = cluster[0]
        for mention_id in cluster[1:]:
            union(root, mention_id)

    grouped: Dict[str, List[Mention]] = defaultdict(list)
    for mention in mentions:
        grouped[find(mention.mention_id)].append(mention)

    entities: List[CanonicalEntity] = []
    for index, cluster in enumerate(sorted(grouped.values(), key=lambda group: min(item.start for item in group))):
        entity_type = Counter(item.entity_type for item in cluster).most_common(1)[0][0]
        canonical_form, normalization_name, alias_forms = choose_canonical_name(cluster)
        entities.append(
            CanonicalEntity(
                entity_id=f"C{index}",
                entity_type=entity_type,
                mentions=sorted(cluster, key=lambda item: (item.start, item.end)),
                canonical_form=canonical_form,
                normalization_name=normalization_name,
                alias_forms=alias_forms,
            )
        )
    return entities


def choose_canonical_name(mentions: Sequence[Mention]) -> Tuple[str, Optional[str], set[str]]:
    usable_forms = [mention.form.strip() for mention in mentions if mention.form.strip()]
    counts = Counter(form for form in usable_forms if form)
    earliest_by_form = {}
    for mention in mentions:
        if mention.form.strip() and mention.form not in earliest_by_form:
            earliest_by_form[mention.form.strip()] = mention.start
    if counts:
        canonical_form = sorted(
            counts.keys(),
            key=lambda form: (-counts[form], earliest_by_form.get(form, 10**9), len(form)),
        )[0]
    else:
        canonical_form = ""

    normalization_name = None
    for mention in mentions:
        if mention.normalizations:
            normalization_name = f"{mention.normalizations[0]['resource']}:{mention.normalizations[0]['reference']}"
            break

    alias_forms = {normalize_text(mention.form) for mention in mentions if mention.form.strip()}
    return canonical_form, normalization_name, alias_forms


def align_gold_edges(
    knowledge_graph: Sequence[Dict[str, object]],
    canonical_entities: Sequence[CanonicalEntity],
) -> Tuple[List[RelationEdge], List[Dict[str, object]]]:
    alias_index: Dict[str, List[CanonicalEntity]] = defaultdict(list)
    for entity in canonical_entities:
        for alias in entity.alias_forms:
            alias_index[alias].append(entity)
        alias_index[normalize_text(entity.canonical_form)].append(entity)

    aligned: List[RelationEdge] = []
    unresolved: List[Dict[str, object]] = []
    for edge in knowledge_graph:
        subject_entity = resolve_alias_group(edge["subject"], alias_index)
        object_entity = resolve_alias_group(edge["object"], alias_index)
        if subject_entity is None or object_entity is None:
            unresolved.append(edge)
            continue
        aligned.append(
            RelationEdge(
                subject=subject_entity.canonical_form,
                predicate=edge["predicate"],
                object=object_entity.canonical_form,
            )
        )
    return deduplicate_edges(aligned), unresolved


def resolve_alias_group(
    alias_group: Sequence[str] | str,
    alias_index: Dict[str, List[CanonicalEntity]],
) -> Optional[CanonicalEntity]:
    aliases = alias_group if isinstance(alias_group, list) else [alias_group]
    candidates: Counter[str] = Counter()
    lookup: Dict[str, CanonicalEntity] = {}
    for alias in aliases:
        norm = normalize_text(alias)
        for entity in alias_index.get(norm, []):
            candidates[entity.entity_id] += 1
            lookup[entity.entity_id] = entity
    if not candidates:
        return None
    best = sorted(
        candidates.keys(),
        key=lambda entity_id: (
            -candidates[entity_id],
            lookup[entity_id].earliest_start,
            lookup[entity_id].canonical_form,
        ),
    )[0]
    return lookup[best]


def deduplicate_edges(edges: Iterable[RelationEdge]) -> List[RelationEdge]:
    seen = set()
    deduped = []
    for edge in edges:
        key = (edge.subject, edge.predicate, edge.object)
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    return deduped


def split_sentences(text: str) -> List[Tuple[int, int]]:
    if not text:
        return [(0, 0)]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for chunk in SENTENCE_SPLIT.split(text):
        if not chunk:
            continue
        start = text.find(chunk, cursor)
        end = start + len(chunk)
        spans.append((start, end))
        cursor = end
    return spans or [(0, len(text))]


def locate_span(start: int, end: int, spans: Sequence[Tuple[int, int]]) -> int:
    for index, (span_start, span_end) in enumerate(spans):
        if start >= span_start and end <= span_end:
            return index
    closest_index = 0
    closest_distance = float("inf")
    for index, (span_start, span_end) in enumerate(spans):
        distance = min(abs(start - span_start), abs(end - span_end))
        if distance < closest_distance:
            closest_index = index
            closest_distance = distance
    return closest_index


def export_documents(documents: Sequence[Document]) -> List[Dict[str, object]]:
    exported = []
    for document in documents:
        exported.append(
            {
                "doc_id": document.doc_id,
                "split": document.split,
                "text": document.text,
                "layout": document.layout,
                "mentions": [asdict(mention) for mention in document.mentions],
                "canonical_entities": [
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "canonical_form": entity.canonical_form,
                        "normalization_name": entity.normalization_name,
                        "alias_forms": sorted(entity.alias_forms),
                    }
                    for entity in document.canonical_entities
                ],
                "gold_relation_edges": [asdict(edge) for edge in document.gold_relation_edges],
                "unresolved_gold_edges": document.unresolved_gold_edges,
            }
        )
    return exported
