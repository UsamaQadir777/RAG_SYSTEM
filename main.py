"""
FastAPI application for the Domain-Specific RAG System.
Features:
- /ask endpoint with citation and confidence
- /ask/stream endpoint with streaming responses
- Query history tracking
- Request latency measurement
- Structured logging
- Simple frontend
"""
import os
import time
import logging
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware          # ← ADD THIS

from config import LOG_FILE, LOG_LEVEL
from rag_pipeline import retrieve
from llm_generator import generate_answer, generate_answer_stream

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- App Setup ---
app = FastAPI(
    title="Domain-Specific RAG System",
    description="Mini ChatGPT for company documents",
    version="1.0.0",
)

# --- CORS Middleware (fixes localhost vs 127.0.0.1 mismatch) --- ← ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# --- Query History (in-memory, last 100 queries) ---
query_history = deque(maxlen=100)


# --- Middleware: Latency Measurement ---
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    logger.info(
        "%s %s completed in %.2f ms",
        request.method,
        request.url.path,
        latency_ms,
    )
    return response


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the frontend UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "RAG system running"}


@app.get("/ask")
def ask_question(query: str):
    """
    Answer a question using the RAG pipeline.

    Returns: answer, citations, confidence, latency_ms
    """
    if not query.strip():
        return {"error": "Please provide a non-empty query."}

    start = time.perf_counter()

    # Retrieve relevant chunks
    results = retrieve(query)

    # Generate answer with citations and confidence
    response = generate_answer(query, results)

    latency_ms = (time.perf_counter() - start) * 1000

    # Log to history
    history_entry = {
        "query": query,
        "answer": response["answer"],
        "citations": response["citations"],
        "confidence": response["confidence"],
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_retrieved": len(results),
    }
    query_history.append(history_entry)
    logger.info(
        "Query answered in %.2f ms (confidence=%s, chunks=%d): '%s'",
        latency_ms,
        response["confidence"],
        len(results),
        query[:80],
    )

    return {
        "query": query,
        "answer": response["answer"],
        "citations": response["citations"],
        "confidence": response["confidence"],
        "latency_ms": round(latency_ms, 2),
    }


@app.get("/ask/stream")
def ask_question_stream(query: str):
    """
    Stream the answer token-by-token using Server-Sent Events.
    """
    if not query.strip():
        return {"error": "Please provide a non-empty query."}

    results = retrieve(query)

    def event_stream():
        for token in generate_answer_stream(query, results):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history")
def get_history():
    """Return recent query history."""
    return {"history": list(query_history)}