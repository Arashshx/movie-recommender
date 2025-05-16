
# 🎬 Movie Recommendation System

## 🔎 Project Overview
This project demonstrates a modular, comparative, and interactive recommender system using three mainstream techniques:

1. **Content-Based Filtering**  
2. **Collaborative Filtering**  
3. **Hybrid Filtering**

Users can search for a movie title and receive top-5 similar recommendations, displayed with posters fetched from TMDB API. The project is built using Python and Streamlit.

## 📦 Dataset
The system uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/latest/), including:

- `movies.csv`: movieId, title, genres  
- `tags.csv`: userId, movieId, tag  
- `ratings.csv`: userId, movieId, rating  
- `links.csv`: movieId to TMDB & IMDb mapping  

> All users have rated at least 20 movies.

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

## 📊 Evaluation

100 users with ≥6 rated movies were used to evaluate each model.

| Model Type        | Precision@5 | Recall@5 | F1@5  |
|-------------------|-------------|----------|-------|
| Content-Based     | 0.23        | 0.028    | 0.047 |
| Collaborative (KNN)  | 0.40     | 0.044    | 0.073 |
| Hybrid (α = 0.5)     | **0.44** | **0.058**| **0.096** |

> ✅ The Hybrid model provides the best overall recommendation performance.

## 🧰 Stack & Tools
- Python, NumPy, Pandas  
- scikit-learn (TF-IDF, KNN)  
- Streamlit (UI, interactivity)  
- TMDB API (poster images)  

## 📁 Project Structure
```
├── home.py
├── pages/
│   ├── 1_Content_Based.py
│   ├── 2_Collaborative.py
│   └── 3_Hybrid.py
├── data/
├── .streamlit/
│   └── secrets.toml
├── requirements.txt
└── README.md
```

## 📚 References

### General Theory:
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.

### Content-Based Filtering:
- Choi, J.H. et al. (2012). *A movie recommendation algorithm based on genre correlations*
- Fernández-Tobías, I. et al. (2012). *Content-based recommendation systems*
- Said, A. et al. (2011). *Tuning metadata for better movie recommendations*

### Collaborative Filtering:
- Gaurav, B. et al. (2020). *A comprehensive analysis on collaborative filtering*
- Zhang, Y. et al. (2016). *An improved collaborative recommender using PCA and GA*
- Patel, M. et al. (2019). *Comparative study of collaborative filtering models*

### Hybrid Recommender:
- Wei, S. et al. (2016). *A hybrid approach for movie recommendation via tags and ratings*
- Lekakos, G., & Caravelas, P. (2006). *A hybrid approach for movie recommendation*
