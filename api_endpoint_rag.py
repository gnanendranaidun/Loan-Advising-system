from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import search_embeddings
import uvicorn

app = FastAPI(title="Bank Document Search API",
             description="API to search bank documents using semantic embeddings")

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    file_name: str
    bank_name: str
    loan_type: str
    similarity: float
    content_preview: str
    full_content: str  # Added full content for context

class SearchResponse(BaseModel):
    results: List[SearchResult]
    context: str  # Combined context for RAG

class RAGResponse(BaseModel):
    context: str
    sources: List[dict]

def format_context(results: List[dict]) -> str:
    """
    Format the search results into a single context string for RAG.
    
    Args:
        results (List[dict]): List of search results
    
    Returns:
        str: Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Document {i} - {result['bank_name']} - {result['loan_type']}]\n"
            f"{result['content']}\n"
        )
    
    return "\n".join(context_parts)

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery):
    try:
        # Load embeddings
        embeddings = search_embeddings.load_embeddings()
        
        # Search using the existing functionality
        results = search_embeddings.search_embeddings(
            search_query.query, 
            embeddings, 
            top_k=search_query.top_k
        )
        
        # Format results and create context
        formatted_results = [
            SearchResult(
                file_name=result["file_name"],
                bank_name=result["bank_name"],
                loan_type=result["loan_type"],
                similarity=result["similarity"],
                content_preview=result["content"][:200] + "...",  # Preview first 200 chars
                full_content=result["content"]
            )
            for result in results
        ]
        
        # Create combined context for RAG
        context = format_context(results)
        
        return SearchResponse(results=formatted_results, context=context)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-context", response_model=RAGResponse)
async def get_rag_context(search_query: SearchQuery):
    """
    Get formatted context for RAG along with source information.
    This endpoint is specifically designed for retrieval augmented generation.
    """
    try:
        # Load embeddings
        embeddings = search_embeddings.load_embeddings()
        
        # Search using the existing functionality
        results = search_embeddings.search_embeddings(
            search_query.query, 
            embeddings, 
            top_k=search_query.top_k
        )
        
        # Create context for RAG
        context = format_context(results)
        
        # Create source information
        sources = [
            {
                "file_name": result["file_name"],
                "bank_name": result["bank_name"],
                "loan_type": result["loan_type"],
                "similarity": result["similarity"]
            }
            for result in results
        ]
        
        return RAGResponse(context=context, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to Bank Document Search API",
        "endpoints": {
            "/search": "Search documents and get results with context",
            "/rag-context": "Get formatted context for RAG with source information"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Note: Using port 8001 