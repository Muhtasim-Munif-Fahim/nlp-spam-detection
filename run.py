#!/usr/bin/env python3
"""Run spam detection pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import run_pipeline


def main() -> int:
    print("=" * 50)
    print("NLP Spam Detection Pipeline")
    print("=" * 50)

    results = run_pipeline(output_dir="output", n_samples=500)

    print(f"\nDataset: {results['n_samples']} messages, {results['n_features']} features\n")
    print(f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 62)
    for model, metrics in results["results"].items():
        print(f"{model:<22} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

    print(f"\nResults saved to output/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
