"""Text classification models."""

from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_models():
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "svm": LinearSVC(random_state=42, dual="auto"),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    }


def train_and_evaluate_all(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
    return results
