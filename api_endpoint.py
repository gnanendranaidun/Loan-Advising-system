from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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

class SearchResponse(BaseModel):
    results: List[SearchResult]

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
        
        # Format results
        formatted_results = [
            SearchResult(
                file_name=result["file_name"],
                bank_name=result["bank_name"],
                loan_type=result["loan_type"],
                similarity=result["similarity"],
                content_preview=result["content"][:150] + "..."
            )
            for result in results
        ]
        
        return SearchResponse(results=formatted_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Bank Document Search API. Use /search endpoint to search documents."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 