# Assignment 20: Building & Deploying a Content-Based Recommendation System

## Problem Statement

You are working as an AI / ML Engineer. Your task is to build a content-based recommendation system, starting from raw data preprocessing, performing text vectorization, generating recommendations, and finally deploying the application on Render by connecting it to GitHub using Git.

This assignment simulates a real-world end-to-end ML + deployment workflow.

---

## Dataset Requirement

* **Selected Dataset**: Open Data 500 US Companies Dataset (`us_companies.csv`)
* **Kaggle Link**: [https://www.kaggle.com/datasets/thegovlab/open-data-500-companies](https://www.kaggle.com/datasets/thegovlab/open-data-500-companies)
* **Dataset Contains**:
  - Item Identifier: `company_name` (e.g., AccuWeather, Accenture, 48 Factoring Inc.)
  - Text-based Features: `description` (detailed summary of operations), `company_category` (industry domain), and `description_short` (concise elevator pitch)
  - Metadata: `city`, `state`, `year_founded`, `url`, `business_model`

### Restrictions
* Use Python, Pandas, scikit-learn, and Streamlit only.
* No collaborative filtering required (content-based only).
* Keep UI simple, interactive, and user-friendly.

---

## PART 1 — Data Preprocessing

### Task 1: Load & Understand Dataset
1. Load the dataset using Pandas with resilient multi-path fallbacks.
2. Print:
   - Dataset shape (rows x columns)
   - Column names
   - First 5 rows
   - Essential details: missing values count, data types, duplicate check, unique company counts, and category distribution.
3. Identify text column(s) used for recommendations (`description`, enriched with `company_category` and `description_short`).

### Task 2: Text Preprocessing for Recommendation
Apply the following steps on the text column:
1. Handle missing values: replace NaNs with empty string (`""`).
2. Combine descriptive features into a single composite representation.
3. Convert text to lowercase.
4. Remove punctuation and special characters using regular expressions (`[^a-z\s]`).
5. Remove stopwords using NLTK's English stopword lexicon.
6. Store cleaned text in a new column: `clean_text`.

---

## PART 2 — Text Vectorization

### Task 3: Vectorization using TF-IDF
1. Use `TfidfVectorizer()` from scikit-learn to convert text into numerical feature vectors.
2. Set reasonable parameters:
   - `max_features=5000`: Caps vocabulary to the most informative unigrams and bigrams, mitigating noisy rare words.
   - `ngram_range=(1, 2)`: Captures multi-word phrases such as "machine learning" or "data analytics".
   - `stop_words='english'`: Secondary safeguard against lingering noise words.
3. Display:
   - Shape of TF-IDF matrix (529 documents x 5000 features).
   - Sparsity percentage.
   - Sample unigram and bigram features.

### Task 4: Similarity Computation
1. Compute pairwise cosine similarity between all item vectors using scikit-learn's `cosine_similarity`.
2. Store the similarity matrix.
3. Explain why cosine similarity is used:
   - Cosine similarity measures the angle between two document vectors regardless of their length/magnitude.
   - Two company descriptions of differing lengths discussing the same subject produce a near-zero angle (similarity near 1.0).
   - Euclidean distance penalizes differing text lengths and is sensitive to document verbosity.

---

## PART 3 — Recommendation Logic

### Task 5: Build Recommendation Function
Create a function:
```python
def recommend(item_name: str, top_n: int = 5) -> pd.DataFrame:
    # returns top N similar items
```
The function:
1. Finds the index of the selected item (with case-insensitive matching and partial fallback).
2. Extracts pairwise cosine similarity scores for that index against all items.
3. Sorts scores descending and excludes self-match.
4. Returns the top N most similar companies formatted with Rank, Company Name, Category, Similarity Score, and Short Description.
5. Tested with 3 distinct companies:
   - `AccuWeather` (Weather & Environment)
   - `Accenture` (Technology & Consulting)
   - `48 Factoring Inc.` (Finance & Investment)

---

## PART 4 — Simple App Interface

### Task 6: Build UI using Streamlit
1. Create an intuitive, modern Streamlit interface (`app.py`):
   - Dropdown (`st.selectbox`) to select any indexed company.
   - Slider (`st.slider`) to customize the number of recommendations (1–10).
   - Selected company inspection expander (category, location, founding year, description).
   - Action button (`st.button`) to generate recommendations.
   - Results displayed as cards with similarity score, match label, progress bar, description and the shared TF-IDF keywords behind each match.
2. Leverage `@st.cache_resource` to ensure dataset loading and TF-IDF computation occur only once at startup.

---

## PART 5 — Version Control with Git & GitHub

### Task 7: Git & GitHub Setup
1. Initialize a Git repository (`git init`).
2. Add all core assets: `app.py`, `requirements.txt`, `README.md`, `data/us_companies.csv`.
3. Commit and push project code to GitHub repository.

---

## PART 6 — Deployment on Render

### Task 8: Deploy Application on Render
1. Create a Render account at [render.com](https://render.com).
2. Connect Render to the GitHub repository.
3. Configure Web Service:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy the application.

### Task 9: Final Validation
1. Verify live deployment URL.
2. Confirm dropdown, buttons, caching, and recommendation generation work without exceptions.
3. Include live URL in assignment submission documentation.
