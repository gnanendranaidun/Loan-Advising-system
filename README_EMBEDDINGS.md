# Bank Documents Embeddings

This directory contains scripts to convert the bank documents into vector embeddings using OpenAI's Embeddings API and to search through these embeddings.

## Setup

1. Install the required Python packages:
   ```bash
   pip install openai numpy
   ```

2. Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Creating Embeddings

To convert all the bank documents into embeddings, run:

```bash
./create_embeddings.py
```

This script will:
- Traverse all bank directories (HDFC Bank, State Bank of India (SBI), Indian Overseas Bank)
- Read all markdown files
- Chunk the content if necessary
- Convert each chunk to embedding vectors using OpenAI's API
- Save the embeddings to a file named `bank_embeddings.json`

The process might take some time depending on the number and size of the documents, as it needs to make API calls for each chunk.

## Searching Through Embeddings

Once you have generated the embeddings, you can search through them using:

```bash
./search_embeddings.py "your search query here"
```

Additional options:
- `--top N`: Show top N results (default: 5)
- `--file path/to/embeddings.json`: Use a different embeddings file

For example:
```bash
./search_embeddings.py "What are the interest rates for home loans?" --top 7
```

This will find the most semantically similar content to your query across all bank documents.

## How It Works

1. **Embedding Generation**:
   - Documents are read and split into manageable chunks
   - Each chunk is sent to OpenAI's API to generate embeddings
   - These embeddings capture the semantic meaning of the text in high-dimensional space

2. **Semantic Search**:
   - Your search query is also converted to an embedding vector
   - Cosine similarity is calculated between the query embedding and all document embeddings
   - Results are ranked by similarity and returned

This approach allows for more intelligent searching beyond simple keyword matching, as it understands the contextual meaning of the query and documents.

## Notes

- The embedding model used is `text-embedding-3-small`. You can change this in the scripts if needed.
- Each API call to generate embeddings costs a small amount. Check OpenAI's pricing for details.
- The embeddings file can become quite large if you have many documents. 