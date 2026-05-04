#!/usr/bin/env python
"""Plan A — k-of-N voting ensemble across multiple submissions.

Sidesteps the broken logit-averaging in multi_seed_ensemble.py by ensembling
at the *prediction* level. For each (doc_id, subject, predicate, object)
triple, count how many input submissions predict it. Keep if at least k of N
agree.

This works around the candidate-set divergence problem: only 18% of test
candidate pairs were detected by all 5 seeds, so logit averaging silently
treated 47% of pairs (single-seed-only) as "averaged" predictions, letting
through their noise. Voting on final triples instead avoids that pitfall —
a triple needs k seeds to actually predict it (which means the candidate had
to survive each seed's mention pipeline AND classifier).

Inputs accept either:
  * submission CSVs (cols: doc_id, knowledge_graph)
  * raw test_predictions.json files ({doc_id: [{subject, predicate, object}]})

Usage:
  python scripts/vote_ensemble.py \\
    --inputs artifacts/modernbert_e2e_v21d_seed42/test_predictions.json \\
             artifacts/modernbert_e2e_v21d_seed13/test_predictions.json \\
             artifacts/modernbert_e2e_v21d_seed1337/test_predictions.json \\
             artifacts/modernbert_e2e_v21d_seed7/test_predictions.json \\
             artifacts/modernbert_e2e_v21d_seed2025/test_predictions.json \\
    --min-votes 3 \\
    --output submission_v21d_5seed_vote3.csv

Submit several --min-votes values to Kaggle to find the right operating point.
Higher k = more precision; lower k = more recall.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.submission import validate_submission_rows


def load_predictions(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load predictions from either a CSV submission or a test_predictions.json."""
    raw = path.read_text(encoding="utf-8")
    # Heuristic: JSON files start with `{`, CSVs start with `doc_id`
    if raw.lstrip().startswith("{"):
        data = json.loads(raw)
        # Normalize edges to plain dicts
        out: Dict[str, List[Dict[str, str]]] = {}
        for doc_id, edges in data.items():
            normalized: List[Dict[str, str]] = []
            for edge in edges:
                normalized.append({
                    "subject": str(edge["subject"]),
                    "predicate": str(edge["predicate"]),
                    "object": str(edge["object"]),
                })
            out[str(doc_id)] = normalized
        return out
    # CSV path
    out = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = str(row["doc_id"])
            edges_raw = json.loads(row["knowledge_graph"])
            edges = []
            for edge in edges_raw:
                edges.append({
                    "subject": str(edge["subject"]),
                    "predicate": str(edge["predicate"]),
                    "object": str(edge["object"]),
                })
            out[doc_id] = edges
    return out


def vote_ensemble(
    submissions: List[Dict[str, List[Dict[str, str]]]],
    min_votes: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Keep triples that appear in at least `min_votes` of the input submissions."""
    n = len(submissions)
    if min_votes < 1 or min_votes > n:
        raise ValueError(f"min_votes={min_votes} out of range [1, {n}]")
    all_doc_ids = set()
    for sub in submissions:
        all_doc_ids.update(sub.keys())

    output: Dict[str, List[Dict[str, str]]] = {}
    for doc_id in sorted(all_doc_ids):
        triple_counts: Counter[Tuple[str, str, str]] = Counter()
        # Preserve the most common surface form of each triple (any seed's edge dict
        # works since predicate/subject/object strings are the keys themselves).
        triple_repr: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for sub in submissions:
            seen_in_this_sub: set = set()
            for edge in sub.get(doc_id, []):
                key = (edge["subject"], edge["predicate"], edge["object"])
                if key in seen_in_this_sub:
                    continue  # avoid double-counting within a single submission
                seen_in_this_sub.add(key)
                triple_counts[key] += 1
                triple_repr.setdefault(key, edge)
        kept = [triple_repr[key] for key, count in triple_counts.items() if count >= min_votes]
        # Sort for determinism
        kept.sort(key=lambda e: (e["subject"], e["predicate"], e["object"]))
        output[doc_id] = kept
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Per-seed submission CSVs or test_predictions.json files")
    parser.add_argument("--min-votes", type=int, required=True, help=f"k in 'k-of-N': minimum agreeing seeds to keep a triple")
    parser.add_argument("--output", required=True, help="Output submission CSV path")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"ERROR: input not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"[vote] N={len(paths)}, k={args.min_votes}", flush=True)
    submissions = []
    for p in paths:
        sub = load_predictions(p)
        n_triples = sum(len(v) for v in sub.values())
        print(f"[vote]   {p}: {len(sub)} docs, {n_triples} triples", flush=True)
        submissions.append(sub)

    voted = vote_ensemble(submissions, args.min_votes)
    n_kept = sum(len(v) for v in voted.values())
    print(f"\n[vote] Kept {n_kept} triples after k={args.min_votes} of {len(paths)} voting", flush=True)

    # Compose CSV rows
    rows = []
    for doc_id in sorted(voted.keys()):
        rows.append({
            "doc_id": doc_id,
            "knowledge_graph": json.dumps(voted[doc_id], ensure_ascii=False),
        })

    errors = validate_submission_rows(rows)
    if errors:
        print("ERROR: submission validation failed:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "knowledge_graph"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[vote] Written → {out_path}", flush=True)

    # Per-relation breakdown for sanity
    per_rel: Counter[str] = Counter()
    for edges in voted.values():
        for e in edges:
            per_rel[e["predicate"]] += 1
    print("\n[vote] Per-relation triple counts:")
    for rel, count in sorted(per_rel.items()):
        print(f"  {rel:<22} {count}")


if __name__ == "__main__":
    main()
