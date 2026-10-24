# Assignment 20 - Content Based Recommendation System

Live app: https://content-based-recommender-7aei.onrender.com
(free plan on Render, so first load can take around 1 min)

## Dataset

- Name: Open Data 500 US Companies (`us_companies.csv`)
- Kaggle: https://www.kaggle.com/datasets/thegovlab/open-data-500-companies
- 529 companies, 22 columns
- item name: `company_name`
- text used: `description` + `company_category` + `description_short`

## What I did

1. loaded the csv with pandas and checked shape, columns, nulls
2. cleaned text: lowercase, removed punctuation, removed stopwords (nltk), filled NaN with "" -> saved in `clean_text`
3. TF-IDF with `max_features=5000`, `ngram_range=(1,2)` -> matrix shape (529, 5000)
4. cosine similarity between all companies -> (529, 529)
5. `recommend(item_name, top_n=5)` function, tested on AccuWeather, Accenture and 48 Factoring Inc.
6. streamlit app (`app.py`) with dropdown, slider and button
7. pushed to github and deployed on render

## My learning notes

- first I used only the description column but results were bad because many companies use same words like "platform", "services". adding category and short description helped.
- cosine is better than euclidean here because it doesn't care about text length.
- without caching the app was recomputing tf-idf on every click, `@st.cache_resource` fixed it.
- render gives its own port so I had to add `--server.port $PORT --server.address 0.0.0.0` in start command, otherwise deploy fails.
- AccuWeather gives United Mayflower as #1 just because both say "united states" and "local". tf-idf only matches words, not meaning.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render settings

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Instance: Free
