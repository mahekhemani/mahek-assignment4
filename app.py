import os
import ssl
import numpy as np
from flask import Flask, request, render_template
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Configure Matplotlib to use the 'Agg' backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Bypass SSL verification (temporary solution)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply SVD to reduce dimensionality (LSA)
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        if query:
            # Process the query
            similarities, top_indices = process_query(query)
            # Get the top 5 documents and their similarity scores
            top_docs = [(documents[i], similarities[0, i]) for i in top_indices]
            # Generate a bar chart for the top documents' similarity scores
            chart_filename = create_similarity_chart(similarities[0, top_indices], top_indices)
            # Render the results
            return render_template('index.html', results=top_docs, query=query, chart_filename=chart_filename)
    # Render the homepage
    return render_template('index.html')

# Function to process the query and calculate cosine similarity
def process_query(query):
    # Transform the query into the TF-IDF space
    query_vec = vectorizer.transform([query])
    # Project the query vector into the reduced SVD space
    query_reduced = svd.transform(query_vec)
    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_reduced, X_reduced)
    # Get the top 5 most similar documents
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    return similarities, top_indices

# Function to create a bar chart for the top documents' similarity scores
def create_similarity_chart(similarity_scores, top_indices):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 6), similarity_scores, tick_label=[f'Doc {i+1}' for i in range(5)])
    plt.xlabel('Top Documents')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Top 5 Documents')
    chart_filename = 'static/similarity_chart.png'
    plt.savefig(chart_filename)
    plt.close()
    return chart_filename

# Run the Flask app
if __name__ == '__main__':
    # Ensure the static folder exists
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=3000)
