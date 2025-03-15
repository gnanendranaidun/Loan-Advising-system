#!/usr/bin/env python3
import json
import numpy as np
from openai import OpenAI
import argparse

# Initialize OpenAI client
client = OpenAI()

def load_embeddings(file_path="bank_embeddings.json"):
    """Load embeddings from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert list embeddings back to numpy arrays for faster computation
    for item in data:
        item["embedding"] = np.array(item["embedding"])
    
    return data

def get_query_embedding(query):
    """Get embedding for a search query"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_embeddings(query, embeddings, top_k=5):
    """Search for the most similar documents to the query"""
    query_embedding = get_query_embedding(query)
    
    # Calculate similarity scores
    for item in embeddings:
        item["similarity"] = cosine_similarity(query_embedding, item["embedding"])
    
    # Sort by similarity (highest first)
    results = sorted(embeddings, key=lambda x: x["similarity"], reverse=True)
    
    return results[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Search bank documents by semantic similarity")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--file", default="bank_embeddings.json", help="Path to embeddings file")
    
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.file}...")
    embeddings = load_embeddings(args.file)
    
    # Search
    print(f"Searching for: {args.query}")
    results = search_embeddings(args.query, embeddings, top_k=args.top)
    
    # Display results
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['file_name']} (Bank: {result['bank_name']}, Type: {result['loan_type']})")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Content preview: {result['content'][:150]}...")

if __name__ == "__main__":
    main() 