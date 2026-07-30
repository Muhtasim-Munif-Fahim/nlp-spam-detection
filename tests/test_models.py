"""Tests for model training and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_spam_dataset
from preprocessing import preprocess_corpus, extract_features
from models import get_models, evaluate_model, train_and_evaluate_all
from sklearn.model_selection import train_test_split


class TestModels:
    def test_all_models_are_callable(self):
        models = get_models()
        assert len(models) >= 3
        for name, model in models.items():
            assert hasattr(model, "fit")

    def test_evaluate_returns_metrics(self):
        df = generate_spam_dataset(n_ham=50, n_spam=50)
        df = preprocess_corpus(df)
        X, _ = extract_features(df["cleaned"], max_features=100)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        results = train_and_evaluate_all(X_train, y_train, X_test, y_test)
        assert "naive_bayes" in results
        assert "accuracy" in results["naive_bayes"]
