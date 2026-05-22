import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set app title
st.title("Movie Recommender")

# Load dataset
df = pd.read_csv("imdb_top_1000.csv")

# Create a tags column combining genre and description
df["tags"] = df["Genre"] + " " + df["Overview"]

# Convert text tags into features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
vectors = vectorizer.fit_transform(df["tags"])

# Calculate similarity scores between all movies
similarity = cosine_similarity(vectors)

# User input text box to search for a movie
movie_input = st.text_input("Search movie:")

if movie_input:
    # Find matching movies in the dataset based on user input
    matches = df[df['Series_Title'].str.contains(movie_input, case=False, na=False)]
    
    if len(matches) > 0:
        # Get the index of the first matching movie
        movie_idx = matches.index[0]
        
        # Get similarity scores for this movie and sort them in descending order
        similarity_scores = list(enumerate(similarity[movie_idx]))
        sorted_sims = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Display recommendations
        st.write("Similar movies:")
        
        # Loop through top 5 similar movies (skipping the first one as it is the movie itself)
        for i, score in sorted_sims[1:6]:
            movie_title = df.iloc[i]['Series_Title']
            st.write(f"- {movie_title} ({score:.2%})")
    else:
        st.warning("No matching movies found. Please try another title.")
