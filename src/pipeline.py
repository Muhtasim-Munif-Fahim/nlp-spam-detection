"""Main pipeline: end-to-end spam classification."""

from __future__ import annotations

import json
from pathlib import Path

from data_generation import generate_spam_dataset
from preprocessing import preprocess_corpus, extract_features
from models import train_and_evaluate_all
from sklearn.model_selection import train_test_split


def run_pipeline(output_dir: str | Path = "output", n_samples: int = 500) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = generate_spam_dataset(n_ham=n_samples // 2, n_spam=n_samples // 2)
    df = preprocess_corpus(df)

    X, vectorizer = extract_features(df["cleaned"], method="tfidf", max_features=3000)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    results = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    summary = {"n_samples": len(df), "n_features": X.shape[1], "results": results}
    (output / "results.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(output / "spam_data.csv", index=False)

    return summary
