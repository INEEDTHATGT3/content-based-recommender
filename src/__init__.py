"""Content-based recommendation package.
"""
from .recommender import (
    load_data,
    preprocess_text,
    build_tfidf_matrix,
    compute_similarity_matrix,
    recommend,
)

__all__ = [
    "load_data",
    "preprocess_text",
    "build_tfidf_matrix",
    "compute_similarity_matrix",
    "recommend",
]
