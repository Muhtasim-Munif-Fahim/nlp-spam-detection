"""Tests for data generation and preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_spam_dataset
from preprocessing import clean_text, preprocess_corpus


class TestDataGeneration:
    def test_creates_correct_size(self):
        df = generate_spam_dataset(n_ham=50, n_spam=50)
        assert len(df) == 100

    def test_has_text_and_label(self):
        df = generate_spam_dataset(n_ham=20, n_spam=20)
        assert "text" in df.columns
        assert "label" in df.columns
        assert set(df["label"].unique()) == {0, 1}


class TestPreprocessing:
    def test_clean_text_removes_urls(self):
        result = clean_text("Check http://bit.ly/test for details")
        assert "http" not in result
        assert "url" in result

    def test_clean_text_lowercases(self):
        result = clean_text("HELLO World")
        assert result == "hello world"

    def test_preprocess_adds_features(self):
        df = generate_spam_dataset(n_ham=10, n_spam=10)
        df = preprocess_corpus(df)
        assert "cleaned" in df.columns
        assert "word_count" in df.columns
        assert "has_url" in df.columns
