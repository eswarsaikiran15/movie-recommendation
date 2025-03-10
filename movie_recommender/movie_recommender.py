import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define file paths
movies_path = "movies.csv"
ratings_path = "ratings.csv"
tags_path = "tags.csv"
links_path = "links.csv"

# Load datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

# Ensure necessary columns exist
if "movieId" not in movies.columns:
    raise ValueError("movies.csv is missing 'movieId' column")

# Handle missing values
movies["genres"].fillna("", inplace=True)

# Merge links with movies to include external links
movies = movies.merge(links, on="movieId", how="left")

# Ensure IMDB IDs are properly formatted
movies["imdbId"] = movies["imdbId"].apply(lambda x: f"tt{int(x):07d}" if pd.notnull(x) else None)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["genres"])

# Compute similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Preprocess movie titles
def clean_title(title):
    return title.lower().strip().split(" (", 1)[0]

movies["title_clean"] = movies["title"].apply(clean_title)

# Compute average ratings
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movies = movies.merge(avg_ratings, on="movieId", how="left")
movies["rating"].fillna(movies["rating"].mean(), inplace=True)

# Function to get movie recommendations
def get_recommendations(title, movies, cosine_sim, num_recommendations=5, min_rating=0):
    title = clean_title(title)
    idx = movies.index[movies["title_clean"] == title].tolist()
    if not idx:
        return []
    idx = idx[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movies.iloc[movie_indices].copy()
    recommended_movies = recommended_movies[recommended_movies["rating"] >= min_rating]
    recommended_movies["weighted_score"] = recommended_movies["rating"] * 0.7 + np.arange(len(recommended_movies), 0, -1) * 0.3
    return recommended_movies.sort_values(by="weighted_score", ascending=False)[["title", "rating", "imdbId"]].head(num_recommendations).values.tolist()

# Streamlit UI
st.title("🎬 Movie Recommender")

movie_title = st.text_input("Enter a movie title:")
num_recommendations = st.slider("How many recommendations do you want?", 1, 10, 5)
min_rating = st.slider("Enter minimum rating for recommendations:", 0.0, 5.0, 3.0, 0.1)

if st.button("Get Recommendations"):
    if movie_title:
        with st.spinner("Fetching recommendations..."):
            recommendations = get_recommendations(movie_title, movies, cosine_sim, num_recommendations, min_rating)
        
        if recommendations:
            st.subheader(f"Movies similar to '{movie_title}':")
            for rec in recommendations:
                title, rating, imdb_id = rec
                if imdb_id:
                    imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                    st.markdown(f"- **{title}** (⭐ {round(rating, 1)}) - [IMDB Link]({imdb_url})")
                else:
                    st.markdown(f"- **{title}** (⭐ {round(rating, 1)}) - No IMDB Link Available")
        else:
            st.warning(f"No recommendations found for '{movie_title}'. Showing top-rated movies instead.")
            top_movies = movies.sort_values(by="rating", ascending=False).head(num_recommendations)
            for _, row in top_movies.iterrows():
                imdb_url = f"https://www.imdb.com/title/{row['imdbId']}/" if pd.notnull(row['imdbId']) else "No IMDB Link Available"
                st.markdown(f"- **{row['title']}** (⭐ {round(row['rating'], 1)}) - {imdb_url}")
