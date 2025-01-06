Movie & Knowledge System
This project is a Gradio-based interface that allows users to query a movie dataset or ask general knowledge questions. The system uses a combination of sentence embeddings, FAISS for similarity search, and the Ollama language model to provide relevant responses.

Features
Movie Search:
Users can search for movies based on titles, genres, or keywords.
The system returns similar movies along with their details (title, genres, and overview).
General Knowledge Queries:
Users can ask general knowledge questions, and the system will respond using the Ollama language model.
Technologies Used
Gradio: For creating the user interface.
Sentence Transformers: For generating embeddings of movie titles and keywords.
FAISS: For efficient similarity search in the movie dataset.
Ollama: For answering general knowledge queries.
Pandas: For handling the movie dataset.
NumPy: For numerical operations and embedding concatenation.


Installation
1.Clone the repository:
git clone <repository-url>
2.Install dependencies
3.Download the movie dataset:
Ensure you have a movie_dataset.csv file in the project directory. This file should contain columns like original_title, keywords, genres, overview, and vote_average.
4.Run the application

Usage
Movie Queries:

Enter a movie title, genre, or keyword in the text box.
Example: "Sci-fi action" or "Inception".
General Knowledge Queries:

Enter any general knowledge question.
Example: "What is the capital of France?".

Code Overview
Key Functions
embed_text(title, keywords):

Generates embeddings for movie titles and keywords using the Sentence Transformer model.
embed_query(query):

Converts a user query into an embedding for similarity search.
search_similar_movies(query, top_k=10):

Searches for movies similar to the query using FAISS.
get_movie_details(query):

Retrieves and formats details of similar movies.
is_movie_query(query):

Determines if the query is related to movies.
process_query(query):

Routes the query to either the movie search or general knowledge system.

Gradio Interface
The interface is created using Gradio, with a single text input and output box.
The title and description provide context for the user.

Dataset
The dataset (movie_dataset.csv) should contain the following columns:
original_title: The title of the movie.
keywords: Keywords associated with the movie.
genres: Genres of the movie.
overview: A brief description of the movie.
vote_average: The average rating of the movie.

Future Improvements
Expand Dataset: Include more movies and additional metadata.
Enhance Query Understanding: Improve the system's ability to understand complex queries.
User Feedback: Add a feedback mechanism to improve the system over time




