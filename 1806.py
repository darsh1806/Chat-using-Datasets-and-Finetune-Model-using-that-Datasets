import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import faiss
from datasets import Dataset
from langchain.llms import Ollama

# Load the dataset
df = pd.read_csv('movie_dataset.csv')

# Prepare Q&A data from the dataset
qa_pairs = df[['question', 'answer']].dropna() if 'question' in df.columns and 'answer' in df.columns else pd.DataFrame(columns=['question', 'answer'])

# Function to prepare dataset for fine-tuning
def prepare_qa_dataset(csv_file):
    df = pd.read_csv(csv_file)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns.")
    return Dataset.from_pandas(df[['question', 'answer']])

# Fine-tuning function
def fine_tune_model(dataset, model_name="gpt2", output_dir="./fine_tuned_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the dataset
    def preprocess(example):
        prompt = f"Q: {example['question']} A: {example['answer']}"
        encoding = tokenizer(prompt, padding="max_length", truncation=True, max_length=128)
        encoding["labels"] = encoding["input_ids"]
        return encoding

    tokenized_dataset = dataset.map(preprocess, remove_columns=["question", "answer"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs"
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Fine-tune the model if Q&A data exists
if not qa_pairs.empty:
    qa_dataset = prepare_qa_dataset('movie_dataset.csv')
    fine_tune_model(qa_dataset)

# Load models
movie_model = SentenceTransformer('all-MiniLM-L6-v2')
general_model = pipeline("text-generation", model="./fine_tuned_model") if not qa_pairs.empty else Ollama(model="llama3")

# Function to create embeddings for the dataset
def embed_text(title, keywords):
    title_embedding = movie_model.encode(str(title))
    keywords_embedding = movie_model.encode(str(keywords))
    return np.concatenate((title_embedding, keywords_embedding), axis=0)

# Generate embeddings for the dataset
embeddings = df.apply(lambda row: embed_text(row['original_title'], row['keywords']), axis=1).tolist()
vote_averages = df['vote_average'].values

# Combine embeddings with vote averages
combined_embeddings = []
for embedding, vote in zip(embeddings, vote_averages):
    normalized_vote = np.array([vote])
    combined_embedding = np.concatenate((embedding, normalized_vote), axis=0)
    combined_embeddings.append(combined_embedding)
combined_embeddings = np.array(combined_embeddings).astype('float32')

# Create FAISS index
dimension = combined_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(combined_embeddings)

# Save the index
faiss.write_index(index, 'movie_index.index')

# Query embedding function
def embed_query(query):
    query_embedding = embed_text(query, "")
    normalized_query_vote = 0.0
    query_combined_embedding = np.concatenate((query_embedding, np.array([normalized_query_vote], dtype='float32')))
    return query_combined_embedding.reshape(1, -1).astype('float32')

# Search for similar movies
def search_similar_movies(query, top_k=10):
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, top_k)
    similar_movies = df.iloc[indices[0]]
    return similar_movies

# Get details of similar movies
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

# Handle Q&A from the dataset
def handle_qa_query(query):
    for _, row in qa_pairs.iterrows():
        if query.lower() in row['question'].lower():
            return row['answer']
    return "I couldn't find an answer to that question in the dataset."

# Determine if the query is movie-related
def is_movie_query(query):
    movie_keywords = ["movie", "film", "actor", "genre", "director"]
    return any(keyword in query.lower() for keyword in movie_keywords)

# Process queries
def process_query(query):
    if is_movie_query(query):
        return get_movie_details(query)
    elif not qa_pairs.empty:
        return handle_qa_query(query)
    else:
        return general_model(query, max_length=100)[0]["generated_text"]

# Create Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Enter your query", placeholder="e.g., Sci-fi action or What is the capital of France?"),
    outputs=gr.Textbox(label="Response"),
    title="Movie & Knowledge System",
    description="Ask a question about movies, Q&A based on dataset, or general knowledge."
)

# Launch the application
iface.launch()
