# Bank Document Search API

This project provides a semantic search API for bank documents using OpenAI embeddings. It allows users to search through bank documents using natural language queries and returns the most relevant results based on semantic similarity.

## Features

- Semantic search using OpenAI embeddings
- RESTful API built with FastAPI
- Support for multiple banks and document types
- Configurable number of search results
- JSON response format with similarity scores

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

1. Start the API server:
```bash
python api_endpoint.py
```

2. The server will start at `http://localhost:8000`

3. API Documentation is available at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

4. Example API call:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the eligibility criteria for gold loans?", "top_k": 3}'
```

## API Endpoints

### POST /search

Search for documents using a natural language query.

Request body:
```json
{
    "query": "your search query",
    "top_k": 5  // optional, defaults to 5
}
```

Response format:
```json
{
    "results": [
        {
            "file_name": "document name",
            "bank_name": "bank name",
            "loan_type": "type of loan",
            "similarity": 0.123,
            "content_preview": "preview of the content..."
        }
    ]
}
```

## Files

- `api_endpoint.py`: FastAPI server implementation
- `search_embeddings.py`: Core search functionality using OpenAI embeddings
- `bank_embeddings.json`: Pre-computed embeddings for bank documents

## Dependencies

- FastAPI
- Uvicorn
- OpenAI
- NumPy
- Pydantic

## License

MIT License 