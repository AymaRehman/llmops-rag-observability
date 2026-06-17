import time

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.app.guardrails import validate_user_input
from src.app.rag_chain import RAGPipeline

app = FastAPI(title="Production RAG Observability API")

QUERY_COUNTER = Counter("rag_queries_total", "Total RAG queries processed")
LATENCY_HISTOGRAM = Histogram("rag_query_latency_seconds", "Latency of RAG queries in seconds")

rag = RAGPipeline()

class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    clean_prompt = validate_user_input(request.prompt)
    
    start_time = time.time()
    QUERY_COUNTER.inc()
    
    result = rag.query(clean_prompt)
    
    duration = time.time() - start_time
    LATENCY_HISTOGRAM.observe(duration)
    
    return {
        "query": clean_prompt,
        "answer": result["answer"],
        "retrieved_context": result["context"],
        "latency_seconds": round(duration, 3)
    }

Instrumentator().instrument(app).expose(app)
