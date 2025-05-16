import streamlit as st
import pandas as pd

st.set_page_config(page_title="Movie Recommender Home", layout="wide")

st.markdown("""
# 🍿 Movie Recommendation System

---

## 🔍 Project Overview

This project demonstrates a modular, comparative, and interactive recommender system using three mainstream techniques:

1. [**Content-Based Filtering**](./content-based_filtering)  
2. [**Collaborative Filtering**](./collaborative_filtering)  
3. [**Hybrid Filtering**](./hybrid_filtering)

Users can search for a movie title and receive top-5 similar recommendations, displayed with posters fetched from TMDB API. The project is built using Python and Streamlit.

---

## 📦 Dataset

The system uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/latest/), including:

- `movies.csv`: movieId, title, genres  
- `tags.csv`: userId, movieId, tag  
- `ratings.csv`: userId, movieId, rating  
- `links.csv`: movieId to TMDB & IMDb mapping  

> All users have rated at least 20 movies.

---

## 🧠 Recommender Algorithms

### ✅ 1. Content-Based Filtering
- Uses movie metadata (genres + user tags)
- TF-IDF vectorization → cosine similarity
- Top 5 similar movies returned for a given input title

### ✅ 2. Collaborative Filtering (Item-Based)
- Builds a movie-user rating matrix
- Uses `NearestNeighbors` (KNN) with cosine distance
- Suggests items rated similarly by the same users

### ✅ 3. Hybrid Filtering
- Combines content and collaborative scores:
  `HybridScore = α * ContentScore + (1 - α) * CollaborativeScore`
- User can adjust α using a slider in the Streamlit app

---
""")

data = {
    "Model Type": ["Content-Based", "Collaborative", "Hybrid (α = 0.5)"],
    "Precision@5": [0.23, 0.40, 0.44],
    "Recall@5": [0.028, 0.044, 0.058],
    "F1@5": [0.047, 0.073, 0.096]
}
df_eval = pd.DataFrame(data)
st.subheader("📊 Evaluation Summary")
st.dataframe(df_eval, use_container_width=True)

st.markdown("""
> ✅ The Hybrid model provides the best overall recommendation performance.

---

## 📚 References

### 📘 General Theory
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.

### 📗 Content-Based Filtering
- Choi, J.H. & Baek, S.J. (2012). *A movie recommendation algorithm based on genre correlations*.  
- Fernández-Tobías, I. et al. (2012). *Content-based recommendation systems: State of the art and trends*.  
- Said, A. et al. (2011). *Tuning metadata for better movie content-based recommendation systems*.

### 📘 Collaborative Filtering
- Gaurav, B. et al. (2020). *A comprehensive analysis on movie recommendation system employing collaborative filtering*.  
- Zhang, Y. et al. (2016). *An improved collaborative movie recommendation system using computational intelligence*.  
- Patel, M. et al. (2019). *Comparative study of recommender system approaches and movie recommendation using collaborative filtering*.

### 📙 Hybrid Recommender
- Wei, S. et al. (2016). *A hybrid approach for movie recommendation via tags and ratings*.  
- Lekakos, G. & Caravelas, P. (2006). *A hybrid approach for movie recommendation*.

---

## 📫 Contact

For feedback, questions, or collaboration inquiries:  
📧 **Email**: arash.shx@gmail.com
""")
