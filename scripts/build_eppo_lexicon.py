"""Build the EPPO pest/plant/disease lexicon from the Bayer flat-format files.

The Bayer flat format is distributed by EPPO (https://data.eppo.int/) as a set
of UTF-8-BOM CSV files.  The three NAME files we use are:

  data/bayer/gainame.txt   – animals (insects, nematodes, mites, rodents …)
  data/bayer/gafname.txt   – micro-organisms, viruses, fungi, bacteria, …
  data/bayer/pflname.txt   – plants

Each file has a DictReader-compatible header row with these 13 fields:

  identifier, datatype, code, lang, langno, preferred, status,
  creation, modification, country, fullname, authority, shortname

Filtering applied:
  • status == 'A'          (active entries only; 'N' = non-standard / deprecated)
  • lang   in ('en','la')  (English common names + Latin scientific names)
  • len(fullname) >= min_alias_length (default 4)

Mapping to our entity types:
  gainame.txt  →  Pest     (GAI animals; overwhelmingly pest insects/nematodes
                             in an EPOP context)
  gafname.txt  →  Disease  (pathogenic fungi, bacteria, viruses, oomycetes)
  pflname.txt  →  Plant

Output JSON keeps the same ``{Pest: [...], Plant: [...], Disease: [...], _meta: {...}}``
shape consumed by ``src/pestclef/mention_detection.py``.

Usage (rebuild from Bayer files, merging into existing seed lexicon)::

    python scripts/build_eppo_lexicon.py \\
        --bayer-dir data/bayer \\
        --output data/lexicons/eppo_pest_plant_disease.json

Backward-compatible with old CSV interface::

    python scripts/build_eppo_lexicon.py \\
        --pest data/lexicons/raw/eppo_pests.csv \\
        --plant data/lexicons/raw/eppo_plants.csv \\
        --output data/lexicons/eppo_pest_plant_disease.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Bayer flat-format reader
# ---------------------------------------------------------------------------

def _read_bayer_name_file(
    path: Path,
    min_alias_length: int = 4,
) -> List[str]:
    """Read a Bayer NAME file and return deduplicated en+la fullnames.

    Only rows with ``status == 'A'`` and ``lang in ('en', 'la')`` are kept.
    The ``fullname`` field is used as the primary alias; ``shortname`` is added
    only when it differs from fullname by more than capitalisation alone.
    """
    if not path.exists():
        raise FileNotFoundError(f"Bayer file not found: {path}")
    seen: set = set()
    aliases: List[str] = []

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status") != "A":
                continue
            if row.get("lang") not in ("en", "la"):
                continue

            for field in ("fullname", "shortname"):
                value = (row.get(field) or "").strip()
                if not value or len(value) < min_alias_length:
                    continue
                if value.isdigit():
                    continue
                key = value.casefold()
                if key in seen:
                    continue
                seen.add(key)
                aliases.append(value)

    return aliases


# ---------------------------------------------------------------------------
# Legacy CSV reader (backward-compat with old --pest/--plant/--disease args)
# ---------------------------------------------------------------------------

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
    has_header = any(
        column.replace(" ", "_") in {
            "preferred_name", "preferred", "name", "other_names", "synonyms",
            "common_name", "scientific_name",
        }
        for column in header_lookup
    )
    data_rows = rows[1:] if has_header else rows
    name_columns = []
    if has_header:
        for key in ("preferred name", "preferred", "scientific name", "name",
                    "common name", "other names", "synonyms"):
            if key in header_lookup:
                name_columns.append(header_lookup[key])
    if not name_columns:
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
                if not normalized or normalized.isdigit():
                    continue
                aliases.append(normalized)
    return aliases


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # Bayer flat-format (primary)
    parser.add_argument(
        "--bayer-dir",
        type=Path,
        default=None,
        help="Directory containing gainame.txt, gafname.txt, pflname.txt "
             "(default: data/bayer if it exists)",
    )
    parser.add_argument(
        "--min-alias-length",
        type=int,
        default=4,
        help="Minimum alias length to include (default: 4)",
    )

    # Legacy CSV interface
    parser.add_argument("--pest",    type=Path, default=None, help="EPPO pest CSV (legacy)")
    parser.add_argument("--plant",   type=Path, default=None, help="EPPO plant CSV (legacy)")
    parser.add_argument("--disease", type=Path, default=None, help="EPPO disease CSV (legacy)")

    # Seed merging
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("data/lexicons/eppo_pest_plant_disease.json"),
        help="Existing seed lexicon to merge into",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Start fresh (do not merge existing seed lexicon)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lexicons/eppo_pest_plant_disease.json"),
        help="Where to write the merged lexicon JSON",
    )
    args = parser.parse_args()

    # Resolve Bayer dir
    bayer_dir: Optional[Path] = args.bayer_dir
    if bayer_dir is None:
        default_bayer = Path("data/bayer")
        if default_bayer.is_dir():
            bayer_dir = default_bayer

    # Load seed
    base: Dict[str, List[str]] = {"Pest": [], "Plant": [], "Disease": []}
    meta: dict = {}
    if not args.no_seed and args.seed and args.seed.exists():
        seed = json.loads(args.seed.read_text(encoding="utf-8"))
        for key in ("Pest", "Plant", "Disease"):
            base[key] = list(seed.get(key, []))
        meta = seed.get("_meta", {})

    # Bayer flat-format
    bayer_stats: dict = {}
    if bayer_dir is not None:
        for label, filename, entity_key in [
            ("gainame (Pest)",    "gainame.txt", "Pest"),
            ("gafname (Disease)", "gafname.txt", "Disease"),
            ("pflname (Plant)",   "pflname.txt", "Plant"),
        ]:
            path = bayer_dir / filename
            if path.exists():
                aliases = _read_bayer_name_file(path, min_alias_length=args.min_alias_length)
                base[entity_key].extend(aliases)
                bayer_stats[label] = len(aliases)
                print(f"  {label}: {len(aliases):,} aliases loaded")
            else:
                print(f"  WARNING: {path} not found, skipping")
    else:
        print("  No Bayer directory found; using CSV inputs only")

    # Legacy CSV (supplemental)
    base["Pest"].extend(_read_csv_aliases(args.pest))
    base["Plant"].extend(_read_csv_aliases(args.plant))
    base["Disease"].extend(_read_csv_aliases(args.disease))

    # Deduplicate and write
    out = {
        "_meta": {
            **meta,
            "rebuilt_from": {
                "bayer_dir": str(bayer_dir) if bayer_dir else None,
                "bayer_stats": bayer_stats,
                "pest_csv":    str(args.pest)    if args.pest    else None,
                "plant_csv":   str(args.plant)   if args.plant   else None,
                "disease_csv": str(args.disease) if args.disease else None,
            },
        },
        "Pest":    _dedupe_preserving_order(base["Pest"]),
        "Plant":   _dedupe_preserving_order(base["Plant"]),
        "Disease": _dedupe_preserving_order(base["Disease"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} | "
        f"Pest={len(out['Pest']):,}  Plant={len(out['Plant']):,}  Disease={len(out['Disease']):,}"
    )


if __name__ == "__main__":
    main()
