"""Text preprocessing: cleaning, tokenization, vectorization."""

from __future__ import annotations

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["cleaned"] = df[text_col].apply(clean_text)
    df["word_count"] = df["cleaned"].str.split().str.len()
    df["has_url"] = df[text_col].str.contains(r"http|bit\.ly", case=False).astype(int)
    df["has_numbers"] = df[text_col].str.contains(r"\d", regex=True).astype(int)
    df["all_caps_ratio"] = (
        df[text_col].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    )
    return df


def extract_features(
    texts: pd.Series,
    method: str = "tfidf",
    max_features: int = 5000,
) -> tuple:
    if method == "count":
        vec = CountVectorizer(max_features=max_features, stop_words="english")
    else:
        vec = TfidfVectorizer(max_features=max_features, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts)
    return X, vec
