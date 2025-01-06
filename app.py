import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from langchain.llms import Ollama

df = pd.read_csv('movie_dataset.csv')

movie_model = SentenceTransformer('all-MiniLM-L6-v2')
general_model = Ollama(model="llama3") 

def embed_text(title, keywords):
    title_embedding = movie_model.encode(str(title))  
    keywords_embedding = movie_model.encode(str(keywords))  
    return np.concatenate((title_embedding, keywords_embedding), axis=0)

embeddings = df.apply(lambda row: embed_text(row['original_title'], row['keywords']), axis=1).tolist()
vote_averages = df['vote_average'].values

combined_embeddings = []
for embedding, vote in zip(embeddings, vote_averages):
    normalized_vote = np.array([vote])  
    combined_embedding = np.concatenate((embedding, normalized_vote), axis=0)
    combined_embeddings.append(combined_embedding)
combined_embeddings = np.array(combined_embeddings).astype('float32')

dimension = combined_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(combined_embeddings)

faiss.write_index(index, 'movie_index.index')

def embed_query(query):
    query_embedding = embed_text(query, "") 
    normalized_query_vote = 0.0  
    query_combined_embedding = np.concatenate((query_embedding, np.array([normalized_query_vote], dtype='float32')))
    return query_combined_embedding.reshape(1, -1).astype('float32')

def search_similar_movies(query, top_k=10):
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, top_k)
    similar_movies = df.iloc[indices[0]]
    return similar_movies

def get_movie_details(query):
    similar_movies = search_similar_movies(query)
    
    movie_details = ""
    for idx, row in similar_movies.iterrows():
        movie_details += f"Movie Name: {row['original_title']}\n"
        movie_details += f"Genres: {row['genres']}\n"  
        movie_details += f"Overview: {row['overview']}\n\n"
    
    if not movie_details:
        movie_details = "No similar movies found."
    
    return movie_details

def is_movie_query(query):
    movie_keywords = ["movie", "film", "actor", "genre", "director"]
    return any(keyword in query.lower() for keyword in movie_keywords)

def process_query(query):
    if is_movie_query(query):
        return get_movie_details(query)
    else:
        return general_model(query)

iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Enter your query", placeholder="e.g., Sci-fi action or What is the capital of France?"),
    outputs=gr.Textbox(label="Response"),
    title="Movie & Knowledge System",
    description="Ask a question about movies or general knowledge, and the system will respond appropriately."
)

iface.launch()
