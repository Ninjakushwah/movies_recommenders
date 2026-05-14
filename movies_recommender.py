import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
df = pd.read_csv("imdb_top_1000.csv")
print("Data loaded")
print(df.head())
print(df.columns)

# Check Runtime
print(df["Runtime"].head())

# Clean Runtime - Remove this from the min and convert to int
runtime_clean = df["Runtime"].str.replace(" min","").astype(int)
df["Runtime"] = runtime_clean
print("Runtime cleaned")

# clean a year column
df["Released_Year"] = pd.to_numeric(df['Released_Year'], errors="coerce")

# Null values hatao
print("Null values:", df.isnull().sum())
df = df.dropna(subset=["Released_Year"])

# Select columns - use right columns for tags
df = df[['Series_Title', 'Genre', 'IMDB_Rating', 'Overview', 'Star1', 'Star2', 'Star3', 'Star4']]

# combine the Cast  (Star1, Star2, Star3, Star4)
df["Cast"] = df["Star1"] + " " + df["Star2"] + " " + df["Star3"] + " " + df["Star4"]
print("Cast combined")
print(df["Cast"].head(3))

# creates Tags  - Genre + Overview + Cast
df["tags"] = df["Genre"] + " " + df["Overview"] + " " + df["Cast"]
print("Tags banaye")
print(df["tags"].head(3))

# remove Comma 
df["tags"] = df["tags"].str.replace(",", "")

#  covert into Lowercase
df["tags"] = df["tags"].str.lower()
print("Lowercase kiya")

# CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")
vector = cv.fit_transform(df["tags"])
print("Vectorization done")
print("Vector shape:", vector.shape)

# Similarity
similarity = cosine_similarity(vector)
print("Similarity calculated")
print("Similarity shape:", similarity.shape)

# Recommendation function
def recommend(movie):
    # search Movie 
    movie_list = df[df['Series_Title'].str.contains(movie.lower(), case=False)].index
    
    if len(movie_list) == 0:
        return [f"Movie '{movie}' not found"]
    
    # Find the Index
    index = movie_list[0]
    print(f"Found movie at index {index}: {df.iloc[index]['Series_Title']}")
    
    # find Similar movies 
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    # Top 5 
    recommendations = []
    for i in distances[1:6]:
        movie_name = df.iloc[i[0]].Series_Title
        score = i[1]
        recommendations.append(movie_name)
        print(f"  - {movie_name} (score: {score:.2f})")
    
    return recommendations

# Test
print("\n Testing Recommendation :")
result = recommend('The Dark Knight')
print("Recommendations:", result)

# Save files
print("\n Saving Files :")
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
print("Vectorizer saved")

pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Similarity saved")

pickle.dump(df, open('movies.pkl', 'wb'))
print("Movies data saved")

print("\nAll done!")