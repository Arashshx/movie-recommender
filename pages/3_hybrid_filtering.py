import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Hybrid Recommender", layout="wide")
st.title("🎭 Hybrid Recommender System (Content + Collaborative)")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    tags = pd.read_csv("data/tags.csv")
    ratings = pd.read_csv("data/ratings.csv")
    links = pd.read_csv("data/links.csv")
    return movies, tags, ratings, links

df_movies, df_tags, df_ratings, df_links = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
df_movies = df_movies[['movieId', 'title', 'genres']].dropna()
df_tags = df_tags[['movieId', 'tag']].dropna()
df_tags['tag'] = df_tags['tag'].str.lower()
df_tags = df_tags[df_tags['tag'].str.contains('[a-zA-Z]', regex=True)]

df_movies['genre_text'] = df_movies['genres'].str.replace('|', ' ', regex=False)
tags_grouped = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
df_movies = pd.merge(df_movies, tags_grouped, on='movieId', how='left')
df_movies['all_tags'] = df_movies['tag'].fillna('')
df_movies['combined_text'] = df_movies['genre_text'] + ' ' + df_movies['all_tags']
df_movies = pd.merge(df_movies, df_links[['movieId', 'tmdbId']], on='movieId', how='left')

# -------------------------------
# Content-Based model
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['combined_text'])
similarity_cb = cosine_similarity(tfidf_matrix)

movieId_to_index_cb = {row['movieId']: i for i, row in df_movies.iterrows()}
index_to_movieId_cb = {i: row['movieId'] for i, row in df_movies.iterrows()}

# -------------------------------
# Collaborative model (item-based)
# -------------------------------
user_movie_matrix = df_ratings.pivot_table(index='userId', columns='movieId', values='rating')
movie_user_matrix = user_movie_matrix.T.fillna(0)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(movie_user_matrix)

movieId_to_index_cf = {movie_id: idx for idx, movie_id in enumerate(movie_user_matrix.index)}
index_to_movieId_cf = {idx: movie_id for movie_id, idx in movieId_to_index_cf.items()}

# -------------------------------
# TMDB poster fetcher
# -------------------------------
TMDB_API_KEY = st.secrets["tmdb"]["key"]

def get_poster_url(tmdb_id):
    if np.isnan(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            poster_path = response.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w185{poster_path}"
    except:
        return None
    return None

# -------------------------------
# Hybrid recommender function
# -------------------------------
def recommend_hybrid(title, alpha=0.5, n_recommendations=5):
    if title not in df_movies['title'].values:
        return []

    idx_cb = df_movies[df_movies['title'] == title].index[0]
    movie_id = df_movies.iloc[idx_cb]['movieId']
    sim_cb = similarity_cb[idx_cb]

    if movie_id not in movieId_to_index_cf:
        return []

    idx_cf = movieId_to_index_cf[movie_id]
    distances, indices = knn_model.kneighbors(movie_user_matrix.iloc[idx_cf].values.reshape(1, -1), n_neighbors=n_recommendations + 10)
    sim_cf = np.zeros(len(df_movies))
    for i in indices.flatten()[1:]:
        mid = index_to_movieId_cf[i]
        if mid in movieId_to_index_cb:
            sim_cf[movieId_to_index_cb[mid]] = 1

    hybrid_score = alpha * sim_cb + (1 - alpha) * sim_cf
    sorted_indices = np.argsort(hybrid_score)[::-1]
    top_indices = [i for i in sorted_indices if i != idx_cb][:n_recommendations]

    return df_movies.iloc[top_indices]

# -------------------------------
# UI - Movie selector + alpha slider
# -------------------------------
selected_title = st.selectbox("🎥 Select a movie:", options=df_movies['title'].sort_values(), index=None)

# 🔄 Add slider to control alpha (hybrid weight)
alpha = st.slider("⚖️ Adjust hybrid weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if selected_title:
    st.subheader(f"Top 5 similar movies (α = {alpha:.2f})")

    recommended = recommend_hybrid(selected_title, alpha=alpha)

    # Display in rows of 3 columns
    rows = [recommended.iloc[i:i+3] for i in range(0, len(recommended), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                poster_url = get_poster_url(movie['tmdbId'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown("📷 No Poster")
                st.markdown(f"**{movie['title']}**")
