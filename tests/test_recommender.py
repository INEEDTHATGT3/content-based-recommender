"""Quick sanity checks for the recommendation pipeline.  Run: python -m pytest -q"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender import (  # noqa: E402
    build_tfidf_matrix,
    clean_text,
    compute_similarity_matrix,
    load_data,
    preprocess_text,
    recommend,
)


def test_clean_text():
    assert clean_text("The Weather, Forecasts & 24/7 DATA!") == "weather forecasts data"
    assert clean_text(None) == ""


def test_pipeline_end_to_end():
    df = preprocess_text(load_data())
    assert len(df) == 529
    assert df["clean_text"].isna().sum() == 0

    _, matrix = build_tfidf_matrix(df)
    assert matrix.shape == (529, 5000)

    sim = compute_similarity_matrix(matrix)
    assert sim.shape == (529, 529)

    for name in ["AccuWeather", "Accenture", "48 Factoring Inc."]:
        recs = recommend(name, df, sim, top_n=5)
        assert len(recs) == 5
        assert name not in recs["Company"].values          # never recommends itself
        assert list(recs["Similarity"]) == sorted(recs["Similarity"], reverse=True)


def test_consulting_peers_for_accenture():
    df = preprocess_text(load_data())
    _, matrix = build_tfidf_matrix(df)
    recs = recommend("Accenture", df, compute_similarity_matrix(matrix), top_n=5)
    assert {"Booz Allen Hamilton", "Deloitte"} & set(recs["Company"])
