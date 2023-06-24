import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np


#Hardcode version of the similarity matrix used
__version__ = '1.0.0'

#Get base directory path
BASE_DIR = Path(__file__).resolve(strict = True).parent

# Load pickle file with similarity matrix
with open(f"{BASE_DIR}/similar_movies_list.pkl","rb") as f:
    recommendation_list = pickle.load(f)

df = pd.read_csv(f'{BASE_DIR}/df_recommendation.csv')

def recommend(movie):
    index = df[df['title'] == movie].index[0]
    recommended_index = recommendation_list[index]

    recommendation = []
    for movie_index in recommended_index:
        recommendation.append(df.iloc[movie_index].title)

    for i in range(len(recommendation) // 2):
        recommendation[i], recommendation[-1 - i] = recommendation[-1 - i], recommendation[i]
    
    return recommendation