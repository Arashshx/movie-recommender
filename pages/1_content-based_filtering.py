import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page configuration
st.set_page_config(page_title="Content-Based Recommender", layout="wide")
st.title("🎬 Content-Based Movie Recommender with Posters")

# Load API key from secrets
TMDB_API_KEY = st.secrets["tmdb"]["key"]

# Load datasets
@st.cache_data
def load_data():
    df_movies = pd.read_csv("data/movies.csv")
    df_tags = pd.read_csv("data/tags.csv")
    df_links = pd.read_csv("data/links.csv")

    df_movies = df_movies.dropna(subset=['genres'])
    df_tags = df_tags.dropna(subset=['tag'])

    df_movies = df_movies[['movieId', 'title', 'genres']]
    df_tags = df_tags[['movieId', 'tag']]
    df_links = df_links[['movieId', 'tmdbId']]

    df_tags = df_tags[df_tags['tag'].str.strip() != '']
    df_tags['tag'] = df_tags['tag'].str.lower()
    df_tags = df_tags[df_tags['tag'].str.contains('[a-zA-Z]', regex=True)]

    df_movies['genre_text'] = df_movies['genres'].str.replace('|', ' ', regex=False)
    tags_grouped = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    tags_grouped.columns = ['movieId', 'all_tags']

    df_movies = pd.merge(df_movies, tags_grouped, on='movieId', how='left')
    df_movies = pd.merge(df_movies, df_links, on='movieId', how='left')
    df_movies['all_tags'] = df_movies['all_tags'].fillna('')
    df_movies['combined_text'] = df_movies['genre_text'] + ' ' + df_movies['all_tags']

    return df_movies

df_movies = load_data()

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['combined_text'])

# KNN model fitting
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Get poster from TMDb API
def get_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
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

# Display movie cards in a 3-column grid
def show_movie_cards(indices):
    for i in range(0, len(indices), 3):
        chunk = indices[i:i+3]
        cols = st.columns(3)
        for j, idx in enumerate(chunk):
            with cols[j]:
                movie = df_movies.iloc[idx]
                poster_url = get_poster_url(movie['tmdbId'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.write("No poster available")
                st.markdown(f"**{movie['title']}**")

# User interface
selected_movie = st.selectbox("Select a movie:", options=df_movies['title'].tolist(), index=None)

if selected_movie:
    idx = df_movies[df_movies['title'] == selected_movie].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    st.subheader("Top 5 similar movies:")
    show_movie_cards(indices[0][1:6])
