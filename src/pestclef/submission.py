from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .schema import RelationEdge


def serialize_knowledge_graph(edges: Sequence[RelationEdge]) -> str:
    payload = [
        {"subject": edge.subject, "predicate": edge.predicate, "object": edge.object}
        for edge in edges
    ]
    return json.dumps(payload, ensure_ascii=False)


def validate_submission_rows(rows: Sequence[Dict[str, str]]) -> List[str]:
    errors = []
    for row_index, row in enumerate(rows):
        if sorted(row.keys()) != ["doc_id", "knowledge_graph"]:
            errors.append(f"row {row_index}: invalid columns {sorted(row.keys())}")
            continue
        try:
            payload = json.loads(row["knowledge_graph"])
        except json.JSONDecodeError as exc:
            errors.append(f"row {row_index}: invalid JSON ({exc})")
            continue
        if not isinstance(payload, list):
            errors.append(f"row {row_index}: knowledge_graph must be a list")
            continue
        for edge_index, edge in enumerate(payload):
            if sorted(edge.keys()) != ["object", "predicate", "subject"]:
                errors.append(f"row {row_index} edge {edge_index}: invalid keys {sorted(edge.keys())}")
                continue
            for field in ("subject", "predicate", "object"):
                if not isinstance(edge[field], str):
                    errors.append(f"row {row_index} edge {edge_index}: {field} must be a string")
    return errors


def write_submission(
    predicted_edges_by_doc: Dict[str, List[RelationEdge]],
    doc_ids: Iterable[str],
    output_path: Path,
) -> None:
    rows = []
    for doc_id in doc_ids:
        rows.append(
            {
                "doc_id": str(doc_id),
                "knowledge_graph": serialize_knowledge_graph(predicted_edges_by_doc.get(str(doc_id), [])),
            }
        )
    errors = validate_submission_rows(rows)
    if errors:
        raise ValueError("Submission validation failed: " + "; ".join(errors[:10]))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "knowledge_graph"])
        writer.writeheader()
        writer.writerows(rows)
