"""
Assignment 20 - Content-Based Company Recommender (Streamlit app)
=================================================================

What this app does
------------------
You pick a company -> the app finds other companies whose *descriptions*
talk about similar things. No user ratings, no login data, only text.
That is what "content-based" filtering means.

The pipeline is exactly the same as in the notebook:

    STEP 1  Load the CSV                      (Task 1)
    STEP 2  Clean the text -> clean_text      (Task 2)
    STEP 3  Turn text into TF-IDF vectors     (Task 3)
    STEP 4  Cosine similarity between all     (Task 4)
    STEP 5  recommend(item_name, top_n)       (Task 5)
    STEP 6  Streamlit UI                      (Task 6)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

On Render the start command is:
    streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Settings - change these and see how the recommendations change
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "us_companies.csv"
ITEM_COL = "company_name"                                       # item identifier
TEXT_COLS = ["description", "company_category", "description_short"]  # text features
MAX_FEATURES = 5000          # keep only the 5000 most useful words / phrases
NGRAM_RANGE = (1, 2)         # single words + 2-word phrases ("open data")

KAGGLE_URL = "https://www.kaggle.com/datasets/thegovlab/open-data-500-companies"
GITHUB_URL = "https://github.com/INEEDTHATGT3/content-based-recommender"

st.set_page_config(page_title="Company Recommender", page_icon="🏢", layout="centered")


# ---------------------------------------------------------------------------
# Stopwords: NLTK list (same as the notebook). If NLTK can't download its
# list (e.g. no internet on the server), fall back to scikit-learn's list
# so the app never crashes.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_stopwords() -> set:
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return set(ENGLISH_STOP_WORDS)


# ---------------------------------------------------------------------------
# STEP 2 - text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str, stop_words: set) -> str:
    """lowercase -> keep only letters -> drop stopwords and 1-letter words"""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove punctuation, numbers, symbols
    words = [w for w in text.split() if len(w) > 1 and w not in stop_words]
    return " ".join(words)


# ---------------------------------------------------------------------------
# STEP 1 to 4 - build the model once, then Streamlit caches it.
# Without the cache, every click would re-read the CSV and redo TF-IDF.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the TF-IDF model (first load only)...")
def build_model():
    # STEP 1: load data
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=ITEM_COL).reset_index(drop=True)

    # STEP 2: handle missing values, combine text columns, clean
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)
    stop_words = get_stopwords()
    df["clean_text"] = df["combined_text"].apply(lambda t: clean_text(t, stop_words))

    # STEP 3: TF-IDF vectors (one row per company, one column per word/phrase)
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
                            stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    # STEP 4: cosine similarity between every pair of companies
    similarity = cosine_similarity(tfidf_matrix)

    return df, tfidf, tfidf_matrix, similarity


# ---------------------------------------------------------------------------
# STEP 5 - the recommend() function
# ---------------------------------------------------------------------------
def recommend(item_name: str, top_n: int = 5) -> pd.DataFrame:
    """Return the top_n companies most similar to item_name."""
    # 1. find the index of the selected company (case-insensitive)
    names = df[ITEM_COL].str.lower()
    matches = df.index[names == item_name.strip().lower()]
    if len(matches) == 0:
        return pd.DataFrame()
    idx = matches[0]

    # 2. similarity of this company with every other company
    scores = similarity[idx]

    # 3. sort high -> low, skip the company itself, keep top_n
    ranked = [i for i in np.argsort(scores)[::-1] if i != idx][:top_n]

    rows = []
    for rank, i in enumerate(ranked, start=1):
        rows.append({
            "Rank": rank,
            "Company": df.at[i, ITEM_COL],
            "Category": df.at[i, "company_category"] or "N/A",
            "Similarity": round(float(scores[i]), 4),
            "Shared keywords": ", ".join(shared_keywords(idx, i)),
            "Description": short_desc(i),
            "Website": df.at[i, "url"] if "url" in df.columns else "",
        })
    return pd.DataFrame(rows)


def shared_keywords(a: int, b: int, k: int = 5) -> list:
    """Words that push the similarity up = words with high TF-IDF in BOTH companies.
    Cosine similarity is a sum over words of (weight_in_a * weight_in_b),
    so the biggest products are the biggest reasons for the match."""
    product = tfidf_matrix[a].multiply(tfidf_matrix[b]).toarray().ravel()
    top = [vocab[j] for j in np.argsort(product)[::-1][:15] if product[j] > 0]
    # drop a single word if a 2-word phrase containing it is already in the list
    # ("united" + "united states" -> keep only "united states")
    phrases = [t for t in top if " " in t]
    words = [t for t in top if not any(t in p.split() for p in phrases)]
    return words[:k]


def short_desc(i: int, limit: int = 220) -> str:
    text = str(df.at[i, "description_short"] or df.at[i, "description"]).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def label_for(score: float) -> str:
    """Plain-English label. TF-IDF scores are small; 0.2+ is already a strong match."""
    score = round(score, 2)          # judge the same number the user sees on screen
    if score >= 0.20:
        return "Strong match"
    if score >= 0.10:
        return "Good match"
    return "Weak match"


df, tfidf, tfidf_matrix, similarity = build_model()
vocab = tfidf.get_feature_names_out()

# ---------------------------------------------------------------------------
# STEP 6 - Streamlit UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 About the model")
    st.metric("Companies in dataset", f"{len(df):,}")
    st.metric("TF-IDF matrix", f"{tfidf_matrix.shape[0]} × {tfidf_matrix.shape[1]}")
    st.write(f"**Text used:** {', '.join(TEXT_COLS)}")
    st.write(f"**N-grams:** {NGRAM_RANGE}  ·  **Max features:** {MAX_FEATURES}")
    st.write("**Similarity:** cosine")
    st.markdown("---")
    st.markdown(f"📁 [Dataset on Kaggle]({KAGGLE_URL})")
    st.markdown(f"💻 [Source code on GitHub]({GITHUB_URL})")

st.title("🏢 Company Recommender")
st.write("Pick a company and get other companies that do **similar work**, "
         "based only on how their descriptions are written.")

with st.expander("🤔 How does this work? (2-minute explanation)"):
    st.markdown(
        """
1. **Clean the text** - lowercase it, remove punctuation and common words like
   *the, and, is* (stopwords).
2. **TF-IDF** - give every word a score. A word scores **high** if it appears a
   lot in *this* company's description but is **rare** across all companies
   (e.g. *weather*, *mortgage*). Common words like *services* score low.
3. **Cosine similarity** - each company is now a vector of word scores.
   We measure the **angle** between two vectors:
   same direction = similar topics (score near 1), unrelated = score near 0.
   Angle (not distance) is used so a short and a long description about the
   same topic still match.
4. **Recommend** - sort all companies by their score with the one you picked
   and show the top results.
        """
    )
    st.latex(r"\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}")

companies = sorted(df[ITEM_COL].tolist(), key=str.lower)
default = companies.index("AccuWeather") if "AccuWeather" in companies else 0

selected = st.selectbox("1️⃣ Choose a company (type to search)", companies, index=default)
top_n = st.slider("2️⃣ How many recommendations?", min_value=1, max_value=10, value=5)

# Small info card about the selected company
row = df[df[ITEM_COL] == selected].iloc[0]
with st.container(border=True):
    st.markdown(f"**{selected}**")
    details = [row.get("company_category"), f"{row.get('city', '')}, {row.get('state', '')}".strip(", ")]
    if pd.notna(row.get("year_founded")):
        details.append(f"founded {int(row['year_founded'])}")
    st.caption("  ·  ".join(str(d).strip() for d in details if d and str(d).strip()))
    st.write(short_desc(row.name, limit=400))

if st.button("3️⃣ Get recommendations", type="primary"):
    recs = recommend(selected, top_n)

    if recs.empty:
        st.warning("Couldn't find that company. Try another one.")
    else:
        st.subheader(f"Companies similar to {selected}")

        for _, r in recs.iterrows():
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"**#{r['Rank']}  {r['Company']}**")
                left.caption(r["Category"])
                right.markdown(f"**{r['Similarity']:.2f}**  \n{label_for(r['Similarity'])}")
                st.progress(min(r["Similarity"] / max(recs["Similarity"].max(), 1e-9), 1.0))
                st.write(r["Description"])
                if r["Shared keywords"]:
                    st.caption(f"🔑 Why: both talk about **{r['Shared keywords']}**")

        with st.expander("📋 See results as a table / download"):
            table = recs.drop(columns=["Website"])
            st.dataframe(table, hide_index=True)
            st.download_button("Download CSV", table.to_csv(index=False),
                               file_name=f"recommendations_{selected[:30]}.csv", mime="text/csv")

        st.info("💡 Scores look small (0.1 - 0.3) because each company uses only a few of the "
                f"{tfidf_matrix.shape[1]:,} TF-IDF words. What matters is the **ranking**, "
                "not the absolute number.")
else:
    st.caption("Tip: try AccuWeather, Accenture or 48 Factoring Inc. - the three companies tested in the notebook.")
