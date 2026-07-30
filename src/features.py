"""Additional features: character n-grams and metadata features."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def extract_char_ngrams(texts: pd.Series, ngram_range: tuple = (2, 4), max_features: int = 500) -> tuple:
    vec = CountVectorizer(analyzer="char", ngram_range=ngram_range, max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec


def compute_text_stats(texts: pd.Series) -> pd.DataFrame:
    stats = pd.DataFrame({
        "char_count": texts.str.len(),
        "avg_word_length": texts.str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0),
        "capital_ratio": texts.str.findall(r"[A-Z]").str.len() / texts.str.len().clip(lower=1),
        "punctuation_count": texts.str.findall(r"[!?.$,:;]").str.len(),
        "exclamation_count": texts.str.count("!"),
    })
    return stats
