"""FastAPI REST API server for the Corporate LLM system."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from corporate_llm import CorporateLLM
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="Corporate LLM API",
    description="REST API for corporate document question answering",
    version="1.0.0"
)

# Initialize LLM system
llm_system = CorporateLLM()


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_context: bool = True


class Source(BaseModel):
    filename: str
    filepath: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]


class IndexRequest(BaseModel):
    clear_existing: bool = False


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Corporate LLM API",
        "version": "1.0.0",
        "endpoints": [
            "/ask - POST: Ask a question",
            "/index - POST: Index documents",
            "/stats - GET: Get system statistics",
            "/clear - POST: Clear document index"
        ]
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer based on indexed documents.
    
    Args:
        request: QuestionRequest with question and optional parameters
        
    Returns:
        QuestionResponse with answer and sources
    """
    try:
        result = llm_system.ask(
            question=request.question,
            top_k=request.top_k,
            use_context=request.use_context
        )
        
        return QuestionResponse(
            answer=result['answer'],
            sources=[Source(**source) for source in result['sources']]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_documents(request: IndexRequest):
    """Index all documents from the documents directory.
    
    Args:
        request: IndexRequest with clear_existing flag
        
    Returns:
        Status message
    """
    try:
        llm_system.index_documents(clear_existing=request.clear_existing)
        stats = llm_system.get_stats()
        
        return {
            "status": "success",
            "message": "Documents indexed successfully",
            "total_chunks": stats['total_chunks']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics.
    
    Returns:
        StatsResponse with system statistics
    """
    try:
        stats = llm_system.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_index():
    """Clear the document index.
    
    Returns:
        Status message
    """
    try:
        llm_system.clear_index()
        return {
            "status": "success",
            "message": "Document index cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
