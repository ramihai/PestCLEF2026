from __future__ import annotations

import csv
import html
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


RELATION_COLORS: Dict[str, str] = {
    "Located_in": "#2E6F40",
    "Found_on": "#C06C2B",
    "Occurs_on": "#7A5CFA",
    "Affects": "#C44536",
    "Causes": "#A23B72",
    "Dispersed_by": "#267C8C",
    "Transmits": "#0E4B8B",
}

NODE_FILL = "#F6F1E8"
NODE_STROKE = "#29413A"
BACKGROUND = "#FBF7F0"


@dataclass(frozen=True)
class SubmissionEdge:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class SubmissionDocument:
    doc_id: str
    edges: List[SubmissionEdge]


def load_submission_csv(path: str | Path) -> List[SubmissionDocument]:
    csv_path = Path(path)
    documents: List[SubmissionDocument] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            payload = json.loads(row["knowledge_graph"])
            edges = [
                SubmissionEdge(
                    subject=str(edge["subject"]),
                    predicate=str(edge["predicate"]),
                    object=str(edge["object"]),
                )
                for edge in payload
            ]
            documents.append(SubmissionDocument(doc_id=str(row["doc_id"]), edges=edges))
    return documents


def summarize_submission(documents: Sequence[SubmissionDocument]) -> Dict[str, object]:
    relation_counts: Dict[str, int] = {}
    edge_counts = [len(document.edges) for document in documents]
    for document in documents:
        for edge in document.edges:
            relation_counts[edge.predicate] = relation_counts.get(edge.predicate, 0) + 1
    densest = sorted(documents, key=lambda document: (-len(document.edges), document.doc_id))[:5]
    return {
        "documents": len(documents),
        "total_edges": sum(edge_counts),
        "average_edges": round(sum(edge_counts) / len(edge_counts), 2) if edge_counts else 0.0,
        "max_edges": max(edge_counts) if edge_counts else 0,
        "min_edges": min(edge_counts) if edge_counts else 0,
        "relation_counts": relation_counts,
        "densest_docs": [(document.doc_id, len(document.edges)) for document in densest],
    }


def filter_edges(
    document: SubmissionDocument,
    selected_relations: Sequence[str] | None = None,
    query: str = "",
) -> List[SubmissionEdge]:
    relation_filter = set(selected_relations or [])
    query_value = query.casefold().strip()
    filtered = []
    for edge in document.edges:
        if relation_filter and edge.predicate not in relation_filter:
            continue
        if query_value:
            blob = f"{edge.subject} {edge.predicate} {edge.object}".casefold()
            if query_value not in blob:
                continue
        filtered.append(edge)
    return filtered


def document_choices(documents: Sequence[SubmissionDocument]) -> List[str]:
    return [document.doc_id for document in documents]


def relation_choices(documents: Sequence[SubmissionDocument]) -> List[str]:
    labels = sorted({edge.predicate for document in documents for edge in document.edges})
    return labels


def build_doc_summary(document: SubmissionDocument, edges: Sequence[SubmissionEdge]) -> str:
    relation_counts: Dict[str, int] = {}
    nodes = set()
    for edge in edges:
        relation_counts[edge.predicate] = relation_counts.get(edge.predicate, 0) + 1
        nodes.add(edge.subject)
        nodes.add(edge.object)
    counts_markup = " ".join(
        f"<span class='pill'><span class='dot' style='background:{RELATION_COLORS.get(label, '#666')}'></span>{html.escape(label)}: {count}</span>"
        for label, count in sorted(relation_counts.items())
    ) or "<span class='muted'>No relations in current view.</span>"
    return (
        f"<div class='summary-card'>"
        f"<div class='summary-head'>Document {html.escape(document.doc_id)}</div>"
        f"<div class='summary-stats'>"
        f"<span><strong>{len(edges)}</strong> edges</span>"
        f"<span><strong>{len(nodes)}</strong> entities</span>"
        f"<span><strong>{len(document.edges)}</strong> total in source</span>"
        f"</div>"
        f"<div class='pill-row'>{counts_markup}</div>"
        f"</div>"
    )


def build_edge_table(edges: Sequence[SubmissionEdge]) -> str:
    if not edges:
        return "<div class='empty-table'>No edges match the current filters.</div>"
    rows = []
    for edge in edges:
        color = RELATION_COLORS.get(edge.predicate, "#666666")
        rows.append(
            "<tr>"
            f"<td>{html.escape(edge.subject)}</td>"
            f"<td><span class='predicate-tag' style='background:{color}'>{html.escape(edge.predicate)}</span></td>"
            f"<td>{html.escape(edge.object)}</td>"
            "</tr>"
        )
    return (
        "<table class='edge-table'>"
        "<thead><tr><th>Subject</th><th>Predicate</th><th>Object</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_overview_html(summary: Dict[str, object]) -> str:
    relation_counts = summary["relation_counts"]
    relation_markup = "".join(
        f"<div class='overview-row'><span>{html.escape(label)}</span><strong>{count}</strong></div>"
        for label, count in sorted(relation_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    densest_markup = "".join(
        f"<div class='overview-row'><span>Doc {html.escape(doc_id)}</span><strong>{count}</strong></div>"
        for doc_id, count in summary["densest_docs"]
    )
    return (
        "<div class='overview-grid'>"
        f"<div class='overview-card'><div class='overview-kicker'>Submission</div><div class='overview-big'>{summary['documents']}</div><div class='overview-sub'>documents loaded</div></div>"
        f"<div class='overview-card'><div class='overview-kicker'>Edges</div><div class='overview-big'>{summary['total_edges']}</div><div class='overview-sub'>triples across the file</div></div>"
        f"<div class='overview-card'><div class='overview-kicker'>Average</div><div class='overview-big'>{summary['average_edges']}</div><div class='overview-sub'>edges per document</div></div>"
        f"<div class='overview-card'><div class='overview-kicker'>Range</div><div class='overview-big'>{summary['min_edges']}–{summary['max_edges']}</div><div class='overview-sub'>min to max edges</div></div>"
        f"<div class='overview-panel'><div class='panel-title'>Relation Totals</div>{relation_markup}</div>"
        f"<div class='overview-panel'><div class='panel-title'>Densest Documents</div>{densest_markup}</div>"
        "</div>"
    )


def render_graph_html(document: SubmissionDocument, edges: Sequence[SubmissionEdge]) -> str:
    if not edges:
        return (
            "<div class='graph-shell empty-graph'>"
            "<div class='empty-title'>No relations to render</div>"
            "<div class='empty-sub'>Try clearing filters or selecting another document.</div>"
            "</div>"
        )

    nodes = sorted({edge.subject for edge in edges} | {edge.object for edge in edges})
    positions = compute_force_layout(nodes, edges)
    width = 1120
    height = 760
    margin = 90
    node_radius = 28.0
    svg_edges = []
    edge_counts: Dict[tuple[str, str], int] = {}
    seen_per_pair: Dict[tuple[str, str], int] = {}

    for edge in edges:
        pair = (edge.subject, edge.object)
        edge_counts[pair] = edge_counts.get(pair, 0) + 1

    for edge in edges:
        pair = (edge.subject, edge.object)
        index = seen_per_pair.get(pair, 0)
        seen_per_pair[pair] = index + 1
        x1, y1 = project_point(positions[edge.subject], width, height, margin)
        x2, y2 = project_point(positions[edge.object], width, height, margin)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        start_x = x1 + dx / length * (node_radius - 2.0)
        start_y = y1 + dy / length * (node_radius - 2.0)
        end_x = x2 - dx / length * (node_radius + 12.0)
        end_y = y2 - dy / length * (node_radius + 12.0)
        nx = -dy / length
        ny = dx / length
        fanout = edge_counts[pair]
        offset = (index - (fanout - 1) / 2.0) * 26.0
        cx = (start_x + end_x) / 2.0 + nx * offset
        cy = (start_y + end_y) / 2.0 + ny * offset
        color = RELATION_COLORS.get(edge.predicate, "#666666")
        label_x = 0.25 * start_x + 0.5 * cx + 0.25 * end_x
        label_y = 0.25 * start_y + 0.5 * cy + 0.25 * end_y - 8
        svg_edges.append(
            f"<path d='M {start_x:.1f} {start_y:.1f} Q {cx:.1f} {cy:.1f} {end_x:.1f} {end_y:.1f}' "
            f"stroke='{color}' stroke-width='2.8' stroke-linecap='round' fill='none' marker-end='url(#arrow-{html.escape(edge.predicate)})' opacity='0.92' />"
        )
        svg_edges.append(
            f"<text x='{label_x:.1f}' y='{label_y:.1f}' class='edge-label' fill='{color}'>{html.escape(edge.predicate)}</text>"
        )

    svg_nodes = []
    for node in nodes:
        x, y = project_point(positions[node], width, height, margin)
        svg_nodes.append(
            f"<g class='node-group'>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{node_radius:.1f}' fill='{NODE_FILL}' stroke='{NODE_STROKE}' stroke-width='2.2' />"
            f"<text x='{x:.1f}' y='{y + 48:.1f}' class='node-label'>{html.escape(node)}</text>"
            f"</g>"
        )

    markers = []
    for label, color in RELATION_COLORS.items():
        markers.append(
            f"<marker id='arrow-{html.escape(label)}' viewBox='0 0 14 14' refX='12' refY='7' markerWidth='9' markerHeight='9' orient='auto'>"
            f"<path d='M 0 0 L 14 7 L 0 14 z' fill='{color}' /></marker>"
        )
    legend_markup = "".join(
        f"<span class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{html.escape(label)}</span>"
        for label, color in RELATION_COLORS.items()
        if any(edge.predicate == label for edge in edges)
    )

    return (
        "<div class='graph-shell'>"
        f"<svg class='graph-svg' viewBox='0 0 {width} {height}' role='img' aria-label='Knowledge graph'>"
        f"<defs>{''.join(markers)}</defs>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='28' fill='{BACKGROUND}' />"
        f"{''.join(svg_edges)}"
        f"{''.join(svg_nodes)}"
        "</svg>"
        "<div class='legend-row'><span class='legend-item'><strong>Direction:</strong> subject <span aria-hidden='true'>→</span> object</span></div>"
        f"<div class='legend-row'>{legend_markup}</div>"
        "</div>"
    )


def compute_force_layout(nodes: Sequence[str], edges: Sequence[SubmissionEdge]) -> Dict[str, tuple[float, float]]:
    if len(nodes) == 1:
        return {nodes[0]: (0.5, 0.5)}

    rng = random.Random(13)
    positions = {node: (rng.random(), rng.random()) for node in nodes}
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for edge in edges:
        adjacency[edge.subject].append(edge.object)
        adjacency[edge.object].append(edge.subject)

    area = 1.0
    k = math.sqrt(area / max(len(nodes), 1))
    temperature = 0.18
    for _ in range(140):
        displacements = {node: [0.0, 0.0] for node in nodes}
        for i, left in enumerate(nodes):
            x1, y1 = positions[left]
            for right in nodes[i + 1 :]:
                x2, y2 = positions[right]
                dx = x1 - x2
                dy = y1 - y2
                distance = math.hypot(dx, dy) + 1e-4
                force = (k * k) / distance
                displacements[left][0] += dx / distance * force
                displacements[left][1] += dy / distance * force
                displacements[right][0] -= dx / distance * force
                displacements[right][1] -= dy / distance * force
        for node, neighbors in adjacency.items():
            x1, y1 = positions[node]
            for neighbor in neighbors:
                x2, y2 = positions[neighbor]
                dx = x1 - x2
                dy = y1 - y2
                distance = math.hypot(dx, dy) + 1e-4
                force = (distance * distance) / k
                displacements[node][0] -= dx / distance * force * 0.5
                displacements[node][1] -= dy / distance * force * 0.5
        for node in nodes:
            dx, dy = displacements[node]
            distance = math.hypot(dx, dy) + 1e-6
            x, y = positions[node]
            x = min(0.95, max(0.05, x + dx / distance * min(distance, temperature)))
            y = min(0.95, max(0.05, y + dy / distance * min(distance, temperature)))
            positions[node] = (x, y)
        temperature *= 0.97
    return positions


def project_point(point: tuple[float, float], width: int, height: int, margin: int) -> tuple[float, float]:
    x, y = point
    return (
        margin + x * (width - 2 * margin),
        margin + y * (height - 2 * margin),
    )
