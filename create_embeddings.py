#!/usr/bin/env python3
import os
import json
import glob
from openai import OpenAI
import numpy as np

# Initialize OpenAI client
# Make sure to set OPENAI_API_KEY environment variable or pass it directly
client = OpenAI()

# Function to read markdown file content
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to chunk text into smaller pieces if needed
def chunk_text(text, max_tokens=8000):
    # Simple chunking by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Rough estimate: 1 token ≈ 4 characters
        paragraph_tokens = len(paragraph) // 4
        
        if current_size + paragraph_tokens > max_tokens and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_size = paragraph_tokens
        else:
            current_chunk.append(paragraph)
            current_size += paragraph_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

# Function to get embeddings for a text using OpenAI API
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # You can change to other models if needed
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Function to process a file and get its embeddings
def process_file(file_path):
    try:
        print(f"Processing {file_path}")
        text = read_markdown_file(file_path)
        
        # Get file metadata
        file_name = os.path.basename(file_path)
        directory_path = os.path.dirname(file_path)
        parts = directory_path.split(os.sep)
        
        # Extract bank name and loan type
        bank_name = parts[-2] if parts[-1] != "" else parts[-3]
        loan_type = parts[-1] if parts[-1] != "" else parts[-2]
        
        # Chunk the text if needed
        chunks = chunk_text(text)
        
        # Process each chunk
        results = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                result = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "bank_name": bank_name,
                    "loan_type": loan_type,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "content": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Save a preview
                    "embedding": embedding
                }
                results.append(result)
        
        return results
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

# Main function to process all bank directories
def process_all_banks():
    # List of bank directories
    bank_dirs = [
        "HDFC Bank",
        "State Bank of India (SBI)",
        "Indian Overseas Bank"
    ]
    
    all_embeddings = []
    
    for bank_dir in bank_dirs:
        print(f"Processing bank: {bank_dir}")
        
        # Find all markdown files in the bank directory and its subdirectories
        markdown_files = glob.glob(f"{bank_dir}/**/*.md", recursive=True)
        
        for file_path in markdown_files:
            file_embeddings = process_file(file_path)
            all_embeddings.extend(file_embeddings)
    
    # Save embeddings to a file
    with open("bank_embeddings.json", "w", encoding="utf-8") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = []
        for item in all_embeddings:
            item_copy = item.copy()
            item_copy["embedding"] = list(item["embedding"])
            serializable_embeddings.append(item_copy)
        
        json.dump(serializable_embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"Total embeddings generated: {len(all_embeddings)}")
    print(f"Embeddings saved to bank_embeddings.json")

if __name__ == "__main__":
    process_all_banks() 