"""Comparison and visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path


def format_results_table(results: dict) -> str:
    lines = [f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}"]
    lines.append("-" * 62)
    for model, metrics in results.items():
        lines.append(f"{model:<22} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    return "\n".join(lines)


def load_results(path: str | Path = "output/results.json") -> dict:
    return json.loads(Path(path).read_text())


def find_best_model(results: dict, metric: str = "f1") -> str:
    return max(results, key=lambda m: results[m].get(metric, 0))
