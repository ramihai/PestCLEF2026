from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.submission_viewer import load_submission_csv, render_graph_html  # noqa: E402


class SubmissionViewerTests(unittest.TestCase):
    def test_load_submission_csv_parses_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "submission.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["doc_id", "knowledge_graph"])
                writer.writeheader()
                writer.writerow(
                    {
                        "doc_id": "42",
                        "knowledge_graph": json.dumps(
                            [
                                {"subject": "aphid", "predicate": "Found_on", "object": "wheat"},
                                {"subject": "aphid", "predicate": "Located_in", "object": "Romania"},
                            ]
                        ),
                    }
                )

            documents = load_submission_csv(csv_path)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].doc_id, "42")
            self.assertEqual(len(documents[0].edges), 2)

    def test_render_graph_html_returns_svg(self) -> None:
        documents = load_submission_csv(Path(__file__).resolve().parents[1] / "submission.csv")
        sample = next(document for document in documents if document.edges)
        html = render_graph_html(sample, sample.edges[: min(5, len(sample.edges))])
        self.assertIn("<svg", html)
        self.assertIn("Knowledge graph", html)


if __name__ == "__main__":
    unittest.main()
