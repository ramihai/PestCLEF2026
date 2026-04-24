#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.submission import serialize_knowledge_graph, validate_submission_rows


def load_submission(path: Path) -> Dict[str, List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        data = {}
        for row in reader:
            data[row["doc_id"]] = json.loads(row["knowledge_graph"])
        return data


def get_edge_key(edge: Dict[str, str]) -> Tuple[str, str, str]:
    return (edge["subject"], edge["predicate"], edge["object"])


def ensemble_submissions(
    sub1: Dict[str, List[Dict[str, str]]],
    sub2: Dict[str, List[Dict[str, str]]],
    strategy: str,
) -> Dict[str, List[Dict[str, str]]]:
    doc_ids = sorted(set(sub1.keys()) | set(sub2.keys()))
    ensembled = {}
    
    for doc_id in doc_ids:
        edges1 = {get_edge_key(e): e for e in sub1.get(doc_id, [])}
        edges2 = {get_edge_key(e): e for e in sub2.get(doc_id, [])}
        
        merged_keys: Set[Tuple[str, str, str]] = set()
        
        if strategy == "union":
            merged_keys = set(edges1.keys()) | set(edges2.keys())
        elif strategy == "intersection":
            merged_keys = set(edges1.keys()) & set(edges2.keys())
        elif strategy == "heuristic":
            # sub1 is v9, sub2 is v14
            for key in set(edges1.keys()) | set(edges2.keys()):
                subj, pred, obj = key
                in_sub1 = key in edges1
                in_sub2 = key in edges2
                
                if in_sub1 and in_sub2:
                    merged_keys.add(key)
                elif pred in ("Affects", "Located_in"):
                    if in_sub2:  # Prefer v14
                        merged_keys.add(key)
                elif pred in ("Found_on", "Occurs_on"):
                    if in_sub1:  # Prefer v9
                        merged_keys.add(key)
                elif pred in ("Causes", "Transmits", "Dispersed_by"):
                    if in_sub1 or in_sub2:  # Union for minority
                        merged_keys.add(key)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        ensembled_edges = []
        for key in sorted(merged_keys):
            # Prefer edge formatting from sub2 if available, else sub1
            ensembled_edges.append(edges2.get(key) or edges1[key])
            
        ensembled[doc_id] = ensembled_edges
        
    return ensembled


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble two PestCLEF submissions.")
    parser.add_argument("sub1", help="Path to first submission CSV (v9)")
    parser.add_argument("sub2", help="Path to second submission CSV (v14)")
    parser.add_argument("output", help="Path for ensembled output CSV")
    parser.add_argument("--strategy", choices=["union", "intersection", "heuristic"], default="heuristic")
    args = parser.parse_args()

    sub1_path = Path(args.sub1)
    sub2_path = Path(args.sub2)
    output_path = Path(args.output)
    
    if not sub1_path.exists():
        print(f"Error: {sub1_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not sub2_path.exists():
        print(f"Error: {sub2_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {sub1_path}...")
    sub1_data = load_submission(sub1_path)
    
    print(f"Loading {sub2_path}...")
    sub2_data = load_submission(sub2_path)
    
    print(f"Ensembling using strategy: {args.strategy}")
    ensembled_data = ensemble_submissions(sub1_data, sub2_data, args.strategy)
    
    rows = []
    for doc_id, edges in ensembled_data.items():
        rows.append({
            "doc_id": doc_id,
            "knowledge_graph": json.dumps(edges, ensure_ascii=False)
        })
        
    print("Validating rows...")
    errors = validate_submission_rows(rows)
    if errors:
        print("Validation errors found:", file=sys.stderr)
        for error in errors[:10]:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
        
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "knowledge_graph"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Wrote ensembled submission to {output_path}")
    
    # Calculate some stats
    total_edges1 = sum(len(edges) for edges in sub1_data.values())
    total_edges2 = sum(len(edges) for edges in sub2_data.values())
    total_ensembled = sum(len(edges) for edges in ensembled_data.values())
    
    print(f"Stats:")
    print(f"  Total edges in Sub 1: {total_edges1}")
    print(f"  Total edges in Sub 2: {total_edges2}")
    print(f"  Total edges Ensembled: {total_ensembled}")


if __name__ == "__main__":
    main()
