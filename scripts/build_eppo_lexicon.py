"""Convert EPPO Global Database CSV exports into the seed lexicon JSON.

EPPO Global Database (https://gd.eppo.int/) lets you export per-category lists
("Pests of plants", "Plant species", "Plant diseases") as CSV files. Each row
typically has a preferred name, EPPO code, and one or more synonyms. This
script normalises any number of those CSVs into the simple

  {entity_type: [alias, alias, ...]}

shape consumed by ``src/pestclef/mention_detection.py``'s lexicon loader.

Usage::

    python scripts/build_eppo_lexicon.py \
        --pest data/lexicons/raw/eppo_pests.csv \
        --plant data/lexicons/raw/eppo_plants.csv \
        --disease data/lexicons/raw/eppo_diseases.csv \
        --output data/lexicons/eppo_pest_plant_disease.json

CSVs are read with the standard library and tolerate either a header row
containing names like ``Preferred name``/``Other names``/``Synonyms`` (case
insensitive) or a generic two-column ``code,name`` format.

The script intentionally has no external deps so it can run on the M4 Pro Mac
without setup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional


def _read_csv_aliases(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    aliases: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect=dialect)
        rows = list(reader)
    if not rows:
        return aliases
    header_lookup = {column.strip().lower(): index for index, column in enumerate(rows[0])}
    has_header = any(column.replace(" ", "_") in {
        "preferred_name", "preferred", "name", "other_names", "synonyms",
        "common_name", "scientific_name",
    } for column in header_lookup)
    data_rows = rows[1:] if has_header else rows
    name_columns = []
    if has_header:
        for key in (
            "preferred name",
            "preferred",
            "scientific name",
            "name",
            "common name",
            "other names",
            "synonyms",
        ):
            if key in header_lookup:
                name_columns.append(header_lookup[key])
    if not name_columns:
        # generic fallback: take all non-numeric columns
        name_columns = list(range(len(rows[0])))
    for row in data_rows:
        for index in name_columns:
            if index >= len(row):
                continue
            value = row[index].strip()
            if not value:
                continue
            for token in value.split(";"):
                normalized = token.strip().strip('"').strip("'")
                if not normalized:
                    continue
                if normalized.isdigit():
                    continue
                aliases.append(normalized)
    return aliases


def _dedupe_preserving_order(aliases: Iterable[str]) -> List[str]:
    seen: set = set()
    ordered: List[str] = []
    for alias in aliases:
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(alias)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pest", type=Path, default=None, help="EPPO pest CSV")
    parser.add_argument("--plant", type=Path, default=None, help="EPPO plant CSV")
    parser.add_argument("--disease", type=Path, default=None, help="EPPO disease CSV")
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("data/lexicons/eppo_pest_plant_disease.json"),
        help="Existing seed lexicon to merge into (default: data/lexicons/eppo_pest_plant_disease.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lexicons/eppo_pest_plant_disease.json"),
        help="Where to write the merged lexicon JSON",
    )
    args = parser.parse_args()

    base: dict = {"Pest": [], "Plant": [], "Disease": []}
    if args.seed and args.seed.exists():
        seed = json.loads(args.seed.read_text(encoding="utf-8"))
        for key in ("Pest", "Plant", "Disease"):
            base[key] = list(seed.get(key, []))
        meta = seed.get("_meta", {})
    else:
        meta = {}

    base["Pest"].extend(_read_csv_aliases(args.pest))
    base["Plant"].extend(_read_csv_aliases(args.plant))
    base["Disease"].extend(_read_csv_aliases(args.disease))

    out = {
        "_meta": {
            **meta,
            "rebuilt_from": {
                "pest_csv": str(args.pest) if args.pest else None,
                "plant_csv": str(args.plant) if args.plant else None,
                "disease_csv": str(args.disease) if args.disease else None,
            },
        },
        "Pest": _dedupe_preserving_order(base["Pest"]),
        "Plant": _dedupe_preserving_order(base["Plant"]),
        "Disease": _dedupe_preserving_order(base["Disease"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output} | Pest={len(out['Pest'])} Plant={len(out['Plant'])} Disease={len(out['Disease'])}"
    )


if __name__ == "__main__":
    main()
