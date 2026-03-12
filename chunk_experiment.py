"""
Experiment: Compare chunk sizes, overlap settings, and embedding models.
Measures retrieval relevance across different configurations.

Usage:
    python chunk_experiment.py
"""
import time
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import DOCS_DIR

# --- Experiment Configurations ---
CHUNK_CONFIGS = [
    {"chunk_size": 300, "chunk_overlap": 0, "label": "300 / no overlap"},
    {"chunk_size": 300, "chunk_overlap": 50, "label": "300 / 50 overlap"},
    {"chunk_size": 500, "chunk_overlap": 50, "label": "500 / 50 overlap"},
    {"chunk_size": 800, "chunk_overlap": 0, "label": "800 / no overlap"},
    {"chunk_size": 800, "chunk_overlap": 100, "label": "800 / 100 overlap"},
]

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]

SAMPLE_QUERIES = [
    "What is the company leave policy?",
    "How do I request time off?",
    "What are the working hours?",
    "What is the dress code?",
    "How is performance evaluated?",
]


def run_experiment():
    """Run retrieval experiments across configurations."""
    import os

    # Load documents once
    documents = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_DIR, f))
            documents.extend(loader.load())

    if not documents:
        print(f"No documents found in {DOCS_DIR}/. Add PDFs to run experiments.")
        return

    print("=" * 70)
    print("CHUNK SIZE & EMBEDDING MODEL EXPERIMENT")
    print("=" * 70)

    for model_name in EMBEDDING_MODELS:
        print(f"\n{'─' * 70}")
        print(f"Embedding Model: {model_name}")
        print(f"{'─' * 70}")

        model = SentenceTransformer(model_name)

        for config in CHUNK_CONFIGS:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
            )
            chunks = splitter.split_documents(documents)
            texts = [c.page_content for c in chunks]

            # Build index
            embeddings = model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            print(f"\n  Config: {config['label']} | Chunks: {len(texts)}")

            total_time = 0
            total_dist = 0

            for query in SAMPLE_QUERIES:
                q_emb = np.array(
                    model.encode([query], show_progress_bar=False),
                    dtype=np.float32,
                )
                start = time.perf_counter()
                D, I = index.search(q_emb, k=3)
                elapsed = (time.perf_counter() - start) * 1000

                valid = [(d, i) for d, i in zip(D[0], I[0]) if i != -1]
                avg_dist = np.mean([d for d, _ in valid]) if valid else float("inf")

                total_time += elapsed
                total_dist += avg_dist

                print(
                    f"    Q: {query[:50]:<50} | "
                    f"Avg Dist: {avg_dist:.4f} | "
                    f"Time: {elapsed:.2f} ms"
                )

            n = len(SAMPLE_QUERIES)
            print(
                f"  ── Average: Dist={total_dist / n:.4f}, "
                f"Time={total_time / n:.2f} ms"
            )

    print(f"\n{'=' * 70}")
    print("Experiment complete.")
    print("Lower average distance = better retrieval relevance.")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()