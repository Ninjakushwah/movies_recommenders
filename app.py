import streamlit as st
import pickle
import pandas as pd

# Load saved files
@st.cache_resource
def load_data():
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    df = pickle.load(open('movies.pkl', 'rb'))
    return cv, similarity, df

cv, similarity, df = load_data()

# App title
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title(" Movie Recommendation Engine")
st.write("Enter a movie name to get similar recommendations")

# Input
movie_name = st.text_input("Enter movie name:", placeholder="e.g., The Dark Knight")

if movie_name:
    # Search
    movie_list = df[df['Series_Title'].str.contains(movie_name.lower(), case=False)].index
    
    if len(movie_list) == 0:
        st.error(f" Movie '{movie_name}' not found")
    else:
        index = movie_list[0]
        selected_movie = df.iloc[index]['Series_Title']
        st.success(f" Found: {selected_movie}")
        
        # Get recommendations
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        st.subheader("Similar Movies:")
        cols = st.columns(2)
        
        for i, movie_idx in enumerate(distances[1:6]):
            with cols[i % 2]:
                movie_data = df.iloc[movie_idx[0]]
                score = movie_idx[1]
                
                st.write(f"**{i+1}. {movie_data['Series_Title']}**")
                st.write(f"Rating: ⭐ {movie_data['IMDB_Rating']}")
                st.write(f"Similarity Score: {score:.2%}")
                st.write(f"Genre: {movie_data['Genre']}")
                st.write("---")