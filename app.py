import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title(" Movie Recommender")

@st.cache_resource
def get_recommendations():
    df = pd.read_csv("imdb_top_1000.csv")
    df["tags"] = df["Genre"] + " " + df["Overview"]
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    vectors = vectorizer.fit_transform(df["tags"])
    similarity = cosine_similarity(vectors)
    
    return df, similarity

df, similarity = get_recommendations()

movie = st.text_input("Search movie:")
if movie:
    matches = df[df['Series_Title'].str.contains(movie, case=False, na=False)]
    if len(matches) > 0:
        idx = matches.index[0]
        sims = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
        st.write("Similar movies:")
        for i, score in sims[1:6]:
            st.write(f"{df.iloc[i]['Series_Title']} ({score:.2%})")
