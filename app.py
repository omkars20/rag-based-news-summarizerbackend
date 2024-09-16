

import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

# Global variables
api_key = 'your-news-api-key-here'   # go to newsapi website and sign up there , you will get there free api key
# Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a pre-trained summarization model like BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Function to split text into smaller chunks if needed
def split_text_into_chunks(text, max_chunk_size=1024):
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def fetch_articles(query, count=5):
    url = "https://eventregistry.org/api/v1/article/getArticles"
    headers = {"Content-Type": "application/json"}
    payload = {
        "action": "getArticles",
        "keyword": query,
        "articlesCount": count,
        "resultType": "articles",
        "apiKey": api_key,
        "lang": "eng"
    }

    response = requests.post(url, headers=headers, json=payload)
    print("Response status code:", response.status_code)
    print("Response content:", response.content)  # Debug raw content

    if response.status_code == 200:
        response_json = response.json()
        print("Response JSON:", response_json)  # Debug to ensure correct parsing

        if 'articles' in response_json and 'results' in response_json['articles']:
            # Extract both body and URL
            articles = [
                (article.get('body', ''), article.get('url', ''))  # Make sure to fetch URL
                for article in response_json['articles']['results']
            ]
            print("Fetched articles with URLs:", articles)  # Debug to ensure URLs are included
            return articles
    else:
        print(f"Error fetching articles: {response.status_code}")
    return []



# Step 2: Create FAISS Index
def create_faiss_index(documents):
    document_embeddings = model.encode([doc[0] for doc in documents])
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(np.array(document_embeddings))
    return index, document_embeddings

def retrieve_documents(query, documents, index):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)  # Retrieve top 3 documents
    print(f"Indices of Retrieved Docs: {I}")  # Debugging

    # Ensure both the body and URL are being returned correctly
    retrieved_docs = [documents[i] for i in I[0]] if len(I[0]) > 0 else []
    print(f"Retrieved docs with body and URLs: {retrieved_docs}")  # Debugging to ensure the correct documents are retrieved
    return retrieved_docs


# Step 4: Summarize the Retrieved Articles and Include URL
def summarize_news(article_body):
    chunks = split_text_into_chunks(article_body)
    summaries = [summarizer(chunk, max_length=300, min_length=100, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)  # Concatenate summaries

def generate_summary(query, retrieved_docs):
    summaries = []
    for doc in retrieved_docs:
        body, url = doc
        print(f"Summarizing article from: {url}")  # Debug to ensure URL is being processed
        summary = summarize_news(body)
        summaries.append({
            "summary": summary,
            "source": url  # Include the source URL in the response
        })
    print(f"Generated summaries with URLs: {summaries}")  # Debug to ensure URLs are part of the response
    return summaries




def rag_pipeline(query):
    articles = fetch_articles(query, count=5)
    if articles:
        print(f"Articles found: {len(articles)}")  # Debug how many articles were fetched
        index, _ = create_faiss_index(articles)
        retrieved_docs = retrieve_documents(query, articles, index)
        print(f"Retrieved docs: {retrieved_docs}")  # Debug the documents retrieved
        if retrieved_docs:
            return generate_summary(query, retrieved_docs)
        else:
            print("No relevant documents retrieved.")
            return "No relevant articles retrieved. Please try adjusting your search term."
    else:
        print("No articles found.")
        return "No relevant articles found."



# Suggested Queries (this could be more sophisticated)
def suggest_queries(query):
    return [f"news about {query}", f"latest {query} updates", f"{query} highlights"]

# Flask route for API with suggestions
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query not provided"}), 400

        # Fetch articles and generate summaries
        articles = fetch_articles(query, count=5)
        if not articles:
            return jsonify({"response": "No articles found"}), 404

        response = generate_summary(query, articles)
        return jsonify({"response": response})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
