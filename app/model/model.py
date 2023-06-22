import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Hardcode version of the similarity matrix used
__version__ = '1.0.0'

#Get base directory path
BASE_DIR = Path(__file__).resolve(strict = True).parent

# Load Dataframe of the movies tags
df = pd.read_csv(f'{BASE_DIR}/df_recommendation.csv')

# Define Vectorizer and calculate vectorization
cv = CountVectorizer(max_features = 5000,
                     stop_words = 'english')
vectors = cv.fit_transform(df['tags']).toarray()

# Get recommendation based on vector similarity
def recommend(movie:str):
    index = df[df['title'] == movie].index[0]
    row_vector = np.array(vectors[index]).reshape(1, -1)
    similarity = cosine_similarity(row_vector, vectors)[0]
    similar_movies = sorted(list(enumerate(similarity)),reverse=True,key = lambda x: x[1])[1:6]
    
    recommendations = []

    for i in similar_movies:
        recommendations.append(df.iloc[i[0]].title)
    
    return recommendations