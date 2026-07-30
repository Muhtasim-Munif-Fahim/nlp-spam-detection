"""Tests for analysis module."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import format_results_table, find_best_model


class TestAnalysis:
    def test_format_results(self):
        results = {"nb": {"accuracy": 0.95, "f1": 0.94}}
        table = format_results_table(results)
        assert "nb" in table
        assert "0.95" in table

    def test_find_best(self):
        results = {"a": {"f1": 0.8}, "b": {"f1": 0.9}}
        assert find_best_model(results) == "b"
