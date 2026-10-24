import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Company Recommender", page_icon="🏢")

# using sklearn stopwords here, nltk download was not working on render
stop_words = ENGLISH_STOP_WORDS


def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join([w for w in text.split() if len(w) > 1 and w not in stop_words])


# cache so it doesnt run tfidf again on every click
@st.cache_resource
def load_model():
    df = pd.read_csv("data/us_companies.csv")
    for col in ["description", "company_category", "description_short"]:
        df[col] = df[col].fillna("")
    df["combined_text"] = df["description"] + " " + df["company_category"] + " " + df["description_short"]
    df["clean_text"] = df["combined_text"].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["clean_text"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, similarity


df, similarity = load_model()


def recommend(item_name, top_n=5):
    idx = df[df["company_name"] == item_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]  # remove the company itself
    return scores


st.title("🏢 Company Recommender")
st.write("Select a company and get similar companies based on their description (TF-IDF + cosine similarity).")

with st.sidebar:
    st.write("Total companies:", len(df))
    st.write("Dataset: [Open Data 500 (Kaggle)](https://www.kaggle.com/datasets/thegovlab/open-data-500-companies)")

company = st.selectbox("Select a company", sorted(df["company_name"].tolist()))
top_n = st.slider("Number of recommendations", 1, 10, 5)

row = df[df["company_name"] == company].iloc[0]
st.caption(f"{row['company_category']} | {row['city']}, {row['state']}")
st.write(row["description_short"] or row["description"])

if st.button("Recommend"):
    st.subheader("Similar companies:")
    selected_words = set(row["clean_text"].split())

    for rank, (i, score) in enumerate(recommend(company, top_n), start=1):
        rec = df.iloc[i]
        category = rec["company_category"] or "N/A"
        st.markdown(f"**{rank}. {rec['company_name']}**  ({category})")
        st.progress(float(min(score * 3, 1.0)))  # scores are small so scaled up a bit for the bar
        st.write(f"Similarity: {score:.3f}")
        st.write(rec["description_short"] or rec["description"])

        # common words between both companies
        common = [w for w in rec["clean_text"].split() if w in selected_words]
        common = list(dict.fromkeys(common))[:5]
        if common:
            st.caption("common words: " + ", ".join(common))
        st.divider()
