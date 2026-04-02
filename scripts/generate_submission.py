#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.config import ExperimentConfig
from pestclef.pipeline import run_test_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PestCLEF submission.csv file.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--train-on-dev", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = ExperimentConfig(
        artifacts_dir=Path(args.artifacts_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    result = run_test_submission(config, train_on_dev=args.train_on_dev)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
