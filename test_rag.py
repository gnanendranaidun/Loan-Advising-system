import requests
import json
import sys
from datetime import datetime
import os

def save_to_file(content, prefix):
    """
    Save content to a file with timestamp.
    
    Args:
        content (str): Content to save
        prefix (str): Prefix for the filename
    
    Returns:
        str: Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{prefix}_{timestamp}.txt"
    
    # Write content to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filename

def test_rag_context(query, top_k=5):
    """
    Test the RAG context endpoint and save results to file.
    
    Args:
        query (str): The search query
        top_k (int): Number of top results to retrieve
    """
    url = "http://localhost:8001/rag-context"
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Format the content
        content = f"Query: {query}\n\n"
        content += "=== RAG Context ===\n"
        content += result["context"]
        
        content += "\n\n=== Sources ===\n"
        for i, source in enumerate(result["sources"], 1):
            content += f"\n[Source {i}]\n"
            content += f"File: {source['file_name']}\n"
            content += f"Bank: {source['bank_name']}\n"
            content += f"Loan Type: {source['loan_type']}\n"
            content += f"Similarity Score: {source['similarity']:.4f}\n"
        
        # Save to file
        filename = save_to_file(content, "rag_context")
        print(f"\nRAG context saved to: {filename}")
        
        # Also print to console
        print("\n" + content)
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        sys.exit(1)

def test_search(query, top_k=5):
    """
    Test the search endpoint and save results to file.
    
    Args:
        query (str): The search query
        top_k (int): Number of top results to retrieve
    """
    url = "http://localhost:8001/search"
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Format the content
        content = f"Query: {query}\n\n"
        content += "=== Search Results ===\n"
        for i, res in enumerate(result["results"], 1):
            content += f"\n[Result {i}]\n"
            content += f"File: {res['file_name']}\n"
            content += f"Bank: {res['bank_name']}\n"
            content += f"Loan Type: {res['loan_type']}\n"
            content += f"Similarity Score: {res['similarity']:.4f}\n"
            content += f"Preview: {res['content_preview']}\n"
        
        content += "\n=== Combined Context ===\n"
        content += result["context"]
        
        # Save to file
        filename = save_to_file(content, "search_results")
        print(f"\nSearch results saved to: {filename}")
        
        # Also print to console
        print("\n" + content)
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rag.py <query> [top_k]")
        sys.exit(1)
        
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("\nTesting RAG Context endpoint...")
    test_rag_context(query, top_k)
    
    print("\nTesting Search endpoint...")
    test_search(query, top_k)