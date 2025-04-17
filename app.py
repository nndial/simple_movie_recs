#import streamlit as st 
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
#from streamlit_lottie 
import json

df = pd.read_csv('netflix_titles.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

netflix_data = df.copy()


tfidf = TfidfVectorizer(stop_words='english')
netflix_data['description'] = netflix_data['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_data['description'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data[['title','description']].iloc[movie_indices]



st.title("Simple Movie Recommendation Application")
movie_list = netflix_data['title'].values
movie_list # list of all movies in our dataset

selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
     recommended_movie_names = get_recommendations(selected_movie)
     recommended_movie_names