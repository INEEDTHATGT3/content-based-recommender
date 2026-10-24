import os
import re
import warnings
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# NLTK stopwords (same as the notebook); fall back to scikit-learn's list
# if the NLTK download is not possible (offline / restricted server).
try:
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    _STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Clean text by lowercasing, stripping punctuation/symbols, and removing stopwords.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove stopwords and single characters
    tokens = [w for w in text.split() if len(w) > 1 and w not in _STOP_WORDS]
    return " ".join(tokens)


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load US companies dataset with resilient path fallbacks.

    Args:
        filepath: Optional path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    candidate_paths = [
        filepath,
        "data/us_companies.csv",
        "content-based-recommender/data/us_companies.csv",
        "../content-based-recommender/data/us_companies.csv",
        "../data/us_companies.csv",
        "us_companies.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "us_companies.csv"),
    ]

    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        raise FileNotFoundError(
            "Could not locate us_companies.csv in candidate paths."
        )

    df = pd.read_csv(resolved_path)
    return df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess company text features for content-based recommendation.

    Handles missing values, combines relevant text columns, and creates clean_text.

    Args:
        df: Input DataFrame containing companies.

    Returns:
        DataFrame with added 'clean_text' column.
    """
    df = df.copy()

    # Step 1: Handle missing values in text columns
    text_cols = ["description", "company_category", "description_short"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    # Step 2: Combine text features to capture rich context
    df["combined_text"] = (
        df["description"]
        + " "
        + df["company_category"]
        + " "
        + df["description_short"]
    )

    # Step 3: Apply text cleaning
    df["clean_text"] = df["combined_text"].apply(clean_text)

    return df


def build_tfidf_matrix(
    df: pd.DataFrame,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Convert clean_text into TF-IDF vector matrix.

    Args:
        df: DataFrame containing 'clean_text'.
        max_features: Maximum number of features to retain.
        ngram_range: Lower and upper boundary for n-grams.

    Returns:
        Tuple of (fitted TfidfVectorizer, TF-IDF matrix).
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )
    matrix = tfidf.fit_transform(df["clean_text"])
    return tfidf, matrix


def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """Compute pairwise cosine similarity between all item vectors.

    Args:
        tfidf_matrix: Sparse matrix of TF-IDF vectors.

    Returns:
        Dense square numpy array of similarity scores.
    """
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(
    item_name: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Find top N similar companies for a given company name.

    Args:
        item_name: Target company name.
        df: DataFrame with company details.
        similarity_matrix: Precomputed cosine similarity matrix.
        top_n: Number of recommendations to return.

    Returns:
        DataFrame containing recommended companies, category, similarity score,
        and short description.
    """
    # Step 1: Find index of selected item (case-insensitive)
    matches = df[df["company_name"].astype(str).str.lower() == item_name.strip().lower()]

    if matches.empty:
        # Partial match fallback
        matches = df[
            df["company_name"].astype(str).str.lower().str.contains(item_name.strip().lower(), na=False)
        ]
        if matches.empty:
            return pd.DataFrame()

    idx = matches.index[0]

    # Step 2: Compute/extract similarity scores for this company
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Step 3: Sort descending by score, exclude self (same index)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    # Step 4: Build output DataFrame
    results = []
    for rank, (i, score) in enumerate(sim_scores, start=1):
        desc = str(df["description_short"].iloc[i]) if "description_short" in df.columns else ""
        if not desc or desc.lower() == "nan":
            desc = str(df["description"].iloc[i]) if "description" in df.columns else ""

        results.append({
            "Rank": rank,
            "Company": df["company_name"].iloc[i],
            "Category": df["company_category"].iloc[i] if "company_category" in df.columns else "N/A",
            "Similarity": round(float(score), 4),
            "Similarity (%)": f"{round(float(score) * 100, 1)}%",
            "Description": (desc[:130] + "...") if len(desc) > 130 else desc,
        })

    return pd.DataFrame(results)
