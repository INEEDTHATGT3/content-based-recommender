# Assignment 20: Content-Based Company Recommendation System

**Live app:** https://content-based-recommender-7aei.onrender.com (free tier: first load after a quiet spell takes ~1 min to wake up)  
**GitHub repo:** https://github.com/INEEDTHATGT3/content-based-recommender

done with assignment 19 earlier (word2vec custom embeddings) and continuing with assignment 20 now to build an end-to-end content-based recommendation engine and deploy it on Render via GitHub.
also testing this locally before pushing, makes it way easier to debug Streamlit interactions without waiting on cloud build queues.

This assignment had a few practical challenges at first. I had to figure out how to combine multiple text columns so the TF-IDF vectors didn't just latch onto generic corporate fluff ("solutions", "services"), and I had to make sure the Streamlit app cached the similarity matrix properly so it didn't freeze on every button click. Keeping this simple, clear, and authentic.

---

## What I actually did

I selected the **Open Data 500 US Companies dataset** (`us_companies.csv`), which contains 529 US companies (the CSV has 1,041 *lines* because some descriptions span multiple lines - pandas reads it as 529 rows) with their descriptions, categories, and business summaries.

1. **Preprocessing**: Handled missing values, combined `description`, `company_category`, and `description_short`, lowercased the text, stripped punctuation with regex, and filtered out stopwords using NLTK.
2. **TF-IDF Vectorization**: Fitted a `TfidfVectorizer` with `max_features=5000` and `ngram_range=(1, 2)` to capture domain phrases like "machine learning" and "data analytics".
3. **Cosine Similarity**: Computed a `529 x 529` pairwise similarity matrix using scikit-learn's `cosine_similarity`.
4. **Recommendation Logic**: Wrote a `recommend()` function that looks up any target company, sorts its similarity scores descending, filters out self-match, and returns the top N peer recommendations. Tested with 3 different companies: `AccuWeather`, `Accenture`, and `48 Factoring Inc.`.
5. **Interactive UI**: Built a Streamlit interface (`app.py`) with a searchable company dropdown, a top-N slider and a recommend button. Each result shows a similarity score, a plain-English label (strong / good / weak match) and the **shared keywords** that caused the match, so you can actually see *why* TF-IDF picked it. There's also a "How does this work?" section and a CSV download.
6. **Deployment Pipeline**: Packaged dependencies in `requirements.txt`, documented Render configuration (`--server.port $PORT --server.address 0.0.0.0`), and set up the repository for version control.

---

## Tasks covered

- **Task 1: Load & Understand Dataset** (shape, columns, null check, category counts)
- **Task 2: Text Preprocessing for Recommendation** (lowercasing, punctuation stripping, stopwords removal, null handling -> `clean_text`)
- **Task 3: Vectorization using TF-IDF** (`max_features=5000`, `ngram_range=(1, 2)`, vocabulary inspection)
- **Task 4: Similarity Computation** (cosine similarity matrix, explanation of cosine vs euclidean)
- **Task 5: Build Recommendation Function** (`recommend(item_name, top_n=5)`, verified with 3 test items)
- **Task 6: Build UI using Streamlit** (interactive dropdown, button, progress bars, responsive layout)
- **Task 7: Git & GitHub Setup** (`app.py`, `requirements.txt`, `README.md`, `data/us_companies.csv`)
- **Task 8: Deploy Application on Render** (build & start commands, dynamic port binding)
- **Task 9: Final Validation** (live testing, error handling, sanity checks)

---

## My learning notes & gotchas

- **Combining features vs single column**: At first I only vectorized the `description` column, but many descriptions use generic words like "platform", "leading provider", "innovative". By concatenating `description` + `company_category` + `description_short`, the TF-IDF matrix gained domain-specific anchor words that drastically improved recommendation quality.
- **Why cosine similarity over Euclidean**: Euclidean distance measures straight-line distance, which punishes short descriptions simply because they have fewer words (lower vector magnitude). Cosine similarity measures the angle between vectors, so a concise 15-word summary and a verbose 200-word paragraph about weather forecasting still achieve a very high similarity score.
- **The importance of caching**: Without caching in Streamlit, every time a user changes the slider or clicks a button, the entire dataset is re-read from disk and the TF-IDF matrix is recomputed. Wrapping the model build in `@st.cache_resource` made the app instant after the first load (`cache_resource` instead of `cache_data` because the sparse matrix and vectorizer don't need to be copied on every rerun).
- **NLTK download on the server**: the app first tries NLTK's stopword list (same as the notebook) and falls back to scikit-learn's built-in list if the download fails, so a network hiccup on Render can't crash the app.
- **`Readme.md` vs `README.md`**: Windows doesn't care about filename case but GitHub/Linux does - had to rename it so GitHub shows it on the repo page.
- **Render dynamic port binding**: Render assigns a dynamic port via the `$PORT` environment variable. Running standard `streamlit run app.py` binds to port `8501`, causing Render's port probe to fail and kill the deployment. The start command must be: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Handling empty or missing values**: Some companies in the CSV lacked short descriptions. Replacing `NaN` values with empty strings before concatenation prevented `TypeError` exceptions during text cleaning.

---

## Dataset Details & Kaggle Link

- **Dataset Name**: Open Data 500 US Companies Dataset (`us_companies.csv`)
- **Kaggle Link**: [https://www.kaggle.com/datasets/thegovlab/open-data-500-companies](https://www.kaggle.com/datasets/thegovlab/open-data-500-companies)
- **Total Records**: 529 companies across diverse US industries (Data/Technology, Finance & Investment, Healthcare, Governance, Environment, etc.)

---

## How to run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the Streamlit application:

```bash
streamlit run app.py
```

Or run the Jupyter notebook:
jupyter notebook "Assignment_20_(content_based_recommender).ipynb"

## Deployment on Render

1. Push this repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Content-based recommendation system"
git branch -M main
git remote add origin https://github.com/INEEDTHATGT3/content-based-recommender.git
git push -u origin main
```

2. Log in to [Render](https://render.com) and click **New → Web Service**.
3. Connect your GitHub repository.
4. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service** and wait for the build to complete.

(There's also a `render.yaml` in the repo, so **New → Blueprint** on Render picks up the same settings automatically. `.python-version` pins Python 3.13.)

## Project structure

```
content-based-recommender/
├── app.py                 # Streamlit app (Tasks 1-6 in one readable file)
├── Assignment_20_(content_based_recommender).ipynb   # full notebook walkthrough
├── data/us_companies.csv  # Open Data 500 dataset
├── src/recommender.py     # same pipeline as reusable functions (used by tests)
├── tests/test_recommender.py
├── requirements.txt
├── render.yaml            # Render blueprint (build + start commands)
└── .python-version
```
