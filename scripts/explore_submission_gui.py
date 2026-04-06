#!/usr/bin/env python
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.submission_viewer import (  # noqa: E402
    SubmissionDocument,
    build_doc_summary,
    build_edge_table,
    build_overview_html,
    document_choices,
    filter_edges,
    load_submission_csv,
    relation_choices,
    render_graph_html,
    summarize_submission,
)


DEFAULT_SUBMISSION = Path(__file__).resolve().parents[1] / "submission.csv"
APP_CSS = """
:root {
  --bg: #f4efe6;
  --panel: #fffaf2;
  --ink: #22352f;
  --muted: #6e7a74;
  --line: #d9cfbf;
  --accent: #2e6f40;
  --accent-soft: #e5f0e8;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(194, 152, 67, 0.12), transparent 30%),
    radial-gradient(circle at bottom right, rgba(46, 111, 64, 0.12), transparent 32%),
    var(--bg);
  color: var(--ink);
}

.app-shell { max-width: 1500px; margin: 0 auto; }
.hero {
  background: linear-gradient(135deg, rgba(34,53,47,0.97), rgba(63,94,78,0.94));
  color: #f8f4ea;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 28px;
  padding: 28px 30px;
  box-shadow: 0 18px 40px rgba(34,53,47,0.18);
}
.hero h1 { margin: 0 0 10px; font-size: 2.2rem; line-height: 1.05; }
.hero p { margin: 0; max-width: 900px; color: rgba(248,244,234,0.84); }
.panel {
  background: rgba(255,250,242,0.88);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: 0 12px 30px rgba(72,59,41,0.08);
}
.summary-card, .graph-shell, .overview-card, .overview-panel, .empty-table {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 22px;
}
.summary-card { padding: 18px 20px; }
.summary-head { font-size: 1.1rem; font-weight: 700; margin-bottom: 10px; color: var(--ink); }
.summary-stats { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); margin-bottom: 12px; }
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 11px;
  border-radius: 999px;
  background: #f3ecdf;
  border: 1px solid #e0d5c1;
  font-size: 0.92rem;
}
.dot, .legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}
.legend-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  padding: 14px 18px 18px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--ink);
  background: #f3ecdf;
  border: 1px solid #e0d5c1;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.9rem;
}
.graph-shell { padding: 14px; overflow: hidden; }
.graph-svg { width: 100%; height: auto; display: block; }
.node-label {
  font-size: 14px;
  font-weight: 600;
  fill: var(--ink);
  text-anchor: middle;
}
.edge-label {
  font-size: 12px;
  font-weight: 700;
  text-anchor: middle;
}
.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.overview-card, .overview-panel { padding: 18px; }
.overview-kicker {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 8px;
}
.overview-big { font-size: 2rem; font-weight: 800; color: var(--ink); }
.overview-sub { color: var(--muted); font-size: 0.94rem; }
.panel-title { font-size: 1rem; font-weight: 700; margin-bottom: 10px; }
.overview-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 0;
  border-top: 1px solid #efe4d2;
}
.overview-row:first-of-type { border-top: none; padding-top: 0; }
.edge-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  background: var(--panel);
}
.edge-table th, .edge-table td {
  padding: 12px 14px;
  border-top: 1px solid #eee4d4;
  text-align: left;
  vertical-align: top;
}
.edge-table thead th {
  border-top: none;
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.predicate-tag {
  display: inline-block;
  border-radius: 999px;
  color: white;
  padding: 5px 10px;
  font-size: 0.82rem;
  font-weight: 700;
}
.empty-table, .empty-graph {
  padding: 24px;
  color: var(--muted);
}
.empty-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 8px;
}
.muted { color: var(--muted); }

@media (max-width: 1100px) {
  .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 700px) {
  .overview-grid { grid-template-columns: 1fr; }
  .hero h1 { font-size: 1.7rem; }
}
"""


def resolve_submission_path(uploaded_file: str | None, path_value: str) -> Path:
    if uploaded_file:
        return Path(uploaded_file)
    if path_value.strip():
        return Path(path_value.strip()).expanduser()
    raise ValueError("Provide a CSV path or upload a submission file.")


def pretty_json(document: SubmissionDocument, edges: list) -> str:
    payload = [
        {"subject": edge.subject, "predicate": edge.predicate, "object": edge.object}
        for edge in edges
    ]
    return json.dumps({"doc_id": document.doc_id, "knowledge_graph": payload}, indent=2, ensure_ascii=False)


def empty_state(message: str):
    return (
        [],
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=[]),
        f"### {message}",
        "<div class='muted'>Load a submission to begin exploring.</div>",
        "<div class='muted'>Document-level details will appear here.</div>",
        "<div class='empty-graph'><div class='empty-title'>Graph unavailable</div><div class='empty-sub'>No submission is loaded yet.</div></div>",
        "<div class='empty-table'>No rows to display.</div>",
        "",
    )


def load_submission(uploaded_file: str | None, path_value: str):
    try:
        path = resolve_submission_path(uploaded_file, path_value)
        documents = load_submission_csv(path)
        if not documents:
            return empty_state("The selected CSV contains no rows.")
        doc_ids = document_choices(documents)
        relations = relation_choices(documents)
        summary_html = build_overview_html(summarize_submission(documents))
        first_doc = documents[0]
        visible_edges = filter_edges(first_doc, relations, "")
        return (
            documents,
            gr.update(choices=doc_ids, value=first_doc.doc_id),
            gr.update(choices=relations, value=relations),
            f"### Loaded `{path.name}`\n\n{len(documents)} documents are ready to inspect.",
            summary_html,
            build_doc_summary(first_doc, visible_edges),
            render_graph_html(first_doc, visible_edges),
            build_edge_table(visible_edges),
            pretty_json(first_doc, visible_edges),
        )
    except Exception as exc:
        return empty_state(f"Could not load submission: {exc}")


def render_document_view(
    documents: list[SubmissionDocument],
    doc_id: str,
    selected_relations: list[str],
    query: str,
):
    if not documents:
        return (
            "<div class='muted'>Load a submission to see document details.</div>",
            "<div class='empty-graph'><div class='empty-title'>Graph unavailable</div><div class='empty-sub'>No submission is loaded yet.</div></div>",
            "<div class='empty-table'>No rows to display.</div>",
            "",
        )
    document = next((item for item in documents if item.doc_id == doc_id), documents[0])
    edges = filter_edges(document, selected_relations, query)
    return (
        build_doc_summary(document, edges),
        render_graph_html(document, edges),
        build_edge_table(edges),
        pretty_json(document, edges),
    )


def step_document(documents: list[SubmissionDocument], current_doc_id: str, step: int):
    if not documents:
        return gr.update()
    doc_ids = document_choices(documents)
    try:
        index = doc_ids.index(current_doc_id)
    except ValueError:
        index = 0
    next_index = max(0, min(len(doc_ids) - 1, index + step))
    return gr.update(value=doc_ids[next_index])


def previous_document(documents: list[SubmissionDocument], current_doc_id: str):
    return step_document(documents, current_doc_id, -1)


def next_document(documents: list[SubmissionDocument], current_doc_id: str):
    return step_document(documents, current_doc_id, 1)


def load_default_submission():
    return load_submission(None, str(DEFAULT_SUBMISSION))


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


APP_THEME = gr.themes.Soft(primary_hue="green", secondary_hue="amber")


with gr.Blocks() as demo:
    submission_state = gr.State([])

    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <section class="hero">
              <h1>PestCLEF Submission Atlas</h1>
              <p>
                Load any PestCLEF submission CSV, skim the overall graph density, and inspect each document as a
                directed knowledge graph with live relation filters and searchable triples.
              </p>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=3, elem_classes=["panel"]):
                path_input = gr.Textbox(
                    label="CSV path",
                    value=str(DEFAULT_SUBMISSION),
                    placeholder="submission.csv",
                )
            with gr.Column(scale=2, elem_classes=["panel"]):
                upload_input = gr.File(label="Or upload a CSV", file_types=[".csv"], type="filepath")
            with gr.Column(scale=1, elem_classes=["panel"]):
                load_button = gr.Button("Load Submission", variant="primary")

        load_status = gr.Markdown()
        overview_html = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                doc_dropdown = gr.Dropdown(label="Document", choices=[], value=None, filterable=True)
                relation_filter = gr.CheckboxGroup(label="Relations", choices=[], value=[])
                query_box = gr.Textbox(label="Find in triples", placeholder="Search subject, predicate, or object")
                with gr.Row():
                    prev_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
            with gr.Column(scale=3):
                doc_summary_html = gr.HTML()
                graph_html = gr.HTML()

        edge_table_html = gr.HTML()
        raw_json = gr.Code(label="Filtered JSON", language="json", lines=18)

    load_outputs = [
        submission_state,
        doc_dropdown,
        relation_filter,
        load_status,
        overview_html,
        doc_summary_html,
        graph_html,
        edge_table_html,
        raw_json,
    ]
    load_inputs = [upload_input, path_input]
    load_button.click(load_submission, inputs=load_inputs, outputs=load_outputs)
    upload_input.upload(load_submission, inputs=load_inputs, outputs=load_outputs)

    view_inputs = [submission_state, doc_dropdown, relation_filter, query_box]
    view_outputs = [doc_summary_html, graph_html, edge_table_html, raw_json]
    doc_dropdown.change(render_document_view, inputs=view_inputs, outputs=view_outputs)
    relation_filter.change(render_document_view, inputs=view_inputs, outputs=view_outputs)
    query_box.change(render_document_view, inputs=view_inputs, outputs=view_outputs)
    query_box.submit(render_document_view, inputs=view_inputs, outputs=view_outputs)

    prev_button.click(previous_document, inputs=[submission_state, doc_dropdown], outputs=doc_dropdown).then(
        render_document_view,
        inputs=view_inputs,
        outputs=view_outputs,
    )
    next_button.click(next_document, inputs=[submission_state, doc_dropdown], outputs=doc_dropdown).then(
        render_document_view,
        inputs=view_inputs,
        outputs=view_outputs,
    )

    if DEFAULT_SUBMISSION.exists():
        demo.load(load_default_submission, outputs=load_outputs)


def main() -> None:
    demo.launch(
        css=APP_CSS,
        theme=APP_THEME,
        server_name="127.0.0.1",
        server_port=find_open_port(),
    )


if __name__ == "__main__":
    main()
