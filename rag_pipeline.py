from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import threading

from config import (
    DOCS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K,
)

# Module-level state (lazy-initialized)
_model = None
_index = None
_texts = None
_sources = None
_lock = threading.Lock()


def _load_documents(docs_dir):
    documents = []
    if not os.path.isdir(docs_dir):
        return documents

    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    return documents


def _initialize():
    global _model, _index, _texts, _sources

    if _index is not None:
        return

    with _lock:
        if _index is not None:
            return

        documents = _load_documents(DOCS_DIR)
        if not documents:
            raise RuntimeError(f"No documents found in '{DOCS_DIR}'.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        _model = SentenceTransformer(EMBEDDING_MODEL)

        _texts = [chunk.page_content for chunk in chunks]
        _sources = [
            chunk.metadata.get("source", "unknown") for chunk in chunks
        ]

        embeddings = _model.encode(_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        _index = faiss.IndexFlatL2(dimension)
        _index.add(embeddings)


def retrieve(query, top_k=TOP_K):
    """
    Returns a list of DICTS, not strings.
    Each dict has: text, source, distance
    """
    _initialize()

    query_embedding = np.array(
        _model.encode([query], show_progress_bar=False), dtype=np.float32
    )

    distances, indices = _index.search(query_embedding, k=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": _texts[idx],
            "source": _sources[idx],
            "distance": float(dist),
        })

    return results


def reset():
    global _model, _index, _texts, _sources
    _model = None
    _index = None
    _texts = None
    _sources = None