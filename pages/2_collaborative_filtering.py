import streamlit as st
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors

# Page configuration
st.set_page_config(page_title="Item-Based Collaborative Recommender", layout="wide")
st.title("🤝 Item-Based Collaborative Movie Recommender")

# Load TMDb API key from secrets
TMDB_API_KEY = st.secrets["tmdb"]["key"]

# Load datasets
@st.cache_data
def load_data():
    df_movies = pd.read_csv("data/movies.csv")
    df_ratings = pd.read_csv("data/ratings.csv")
    df_links = pd.read_csv("data/links.csv")

    df_movies = df_movies[['movieId', 'title']]
    df_ratings = df_ratings[['userId', 'movieId', 'rating']]
    df_links = df_links[['movieId', 'tmdbId']]

    df_movies.dropna(subset=['movieId', 'title'], inplace=True)
    df_ratings.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

    df_movies = pd.merge(df_movies, df_links, on='movieId', how='left')
    return df_movies, df_ratings

df_movies, df_ratings = load_data()

# Create user-item matrix
user_movie_matrix = df_ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Transpose for item-based filtering
movie_user_matrix = user_movie_matrix.T.fillna(0)

# Train model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(movie_user_matrix)

# Poster retriever
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

# Show recommended movie cards
def show_movie_cards(df_subset):
    rows = [df_subset.iloc[i:i+3] for i in range(0, len(df_subset), 3)]
    for row in rows:
        cols = st.columns(3)
        for j, (_, movie) in enumerate(row.iterrows()):
            with cols[j]:
                poster_url = get_poster_url(movie['tmdbId'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.write("No poster available")
                st.markdown(f"**{movie['title']}**")

# Recommend similar movies based on item-based KNN
def recommend_similar_movies(title, n_recommendations=5):
    if title not in df_movies['title'].values:
        return pd.DataFrame()

    movie_id = df_movies[df_movies['title'] == title]['movieId'].values[0]
    if movie_id not in movie_user_matrix.index:
        return pd.DataFrame()

    movie_vector = movie_user_matrix.loc[movie_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)
    recommended_ids = movie_user_matrix.index[indices.flatten()[1:]]
    result_df = df_movies[df_movies['movieId'].isin(recommended_ids)]
    return result_df

# User selects a movie
title_list = df_movies['title'].drop_duplicates().sort_values().tolist()
selected_movie = st.selectbox("Select a movie:", options=title_list, index=None)

if selected_movie:
    st.subheader("Top 5 similar movies:")
    recommendations = recommend_similar_movies(selected_movie)
    show_movie_cards(recommendations)
