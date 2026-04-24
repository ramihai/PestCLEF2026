#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pestclef.config import ExperimentConfig
from pestclef.pipeline import run_dev_evaluation


def main() -> None:
    learning_rates = [1e-5, 2e-5, 3e-5]
    warmup_ratios = [0.0, 0.06, 0.1]
    
    results_summary = []
    best_f1 = -1.0
    best_config_name = ""

    print("Starting V18 Hyperparameter Sweep (Interactive)")
    print(f"Grid: LR {learning_rates} x Warmup {warmup_ratios}")
    print("-" * 50)

    for lr in learning_rates:
        for warmup in warmup_ratios:
            config_name = f"lr{lr}_w{warmup}"
            artifacts_dir = f"artifacts/modernbert_sweep_{config_name}"
            
            print(f"\nNext run: {config_name}")
            print(f"  Learning Rate: {lr}")
            print(f"  Warmup Ratio: {warmup}")
            print(f"  Output Dir: {artifacts_dir}")
            
            try:
                input("Press [Enter] to start this run, or [Ctrl+C] to abort sweep... ")
            except KeyboardInterrupt:
                print("\nSweep aborted by user.")
                sys.exit(0)
                
            config = ExperimentConfig(
                artifacts_dir=Path(artifacts_dir),
                model_name="modernbert_staged",
                learning_rate_encoder=lr,
                warmup_ratio=warmup,
                epochs=3,
                batch_size=4,
                train_batch_size=4,
                eval_batch_size=8,
            )
            
            print(f"\nRunning evaluation for {config_name}...")
            result = run_dev_evaluation(config)
            metrics = result["metrics"]
            
            micro_f1 = metrics.get("micro", {}).get("f1", 0.0)
            
            results_summary.append({
                "config": config_name,
                "learning_rate_encoder": lr,
                "warmup_ratio": warmup,
                "micro_f1": micro_f1,
                "macro_f1": metrics.get("macro", {}).get("f1", 0.0),
            })
            
            if micro_f1 > best_f1:
                best_f1 = micro_f1
                best_config_name = config_name
                
            print(f"Run {config_name} complete. Micro F1: {micro_f1:.4f}")
            print("-" * 50)

    print("\n" + "=" * 50)
    print("Sweep Complete! Summary:")
    
    # Sort by micro F1 descending
    results_summary.sort(key=lambda x: x["micro_f1"], reverse=True)
    
    for res in results_summary:
        print(f"  {res['config']:<15} | Micro F1: {res['micro_f1']:<6.4f} | Macro F1: {res['macro_f1']:<6.4f}")
        
    print(f"\nBest Config: {best_config_name} (Micro F1: {best_f1:.4f})")
    
    # Save summary
    summary_path = Path("artifacts/modernbert_sweep_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
