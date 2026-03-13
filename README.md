# RAG System (Retrieval-Augmented Generation)

This project implements a simple **Retrieval-Augmented Generation (RAG)** system that retrieves relevant information from documents and returns answers based on the retrieved content. The system processes documents, converts them into embeddings, stores them in a vector database, and retrieves the most relevant chunks when a user submits a query.

---

## Project Overview

The system works by reading documents (PDF or TXT), splitting them into smaller chunks, converting those chunks into vector embeddings, and storing them in a FAISS vector database. When a user sends a query, the system generates an embedding for the query and retrieves the most relevant document chunks using similarity search.

---

## Features

- Document loading from **PDF and TXT files**
- Document **chunking** for better retrieval
- **Text embeddings** using sentence-transformers
- **Vector storage and similarity search** using FAISS
- **FastAPI backend** for handling queries
- Local API testing for document-based question answering

---

## Technologies Used

- **FastAPI** – API framework for building the backend
- **Uvicorn** – ASGI server for running the FastAPI app
- **Sentence Transformers** – Generate embeddings from text
- **FAISS** – Vector database for similarity search
- **PyPDF** – Extract text from PDF documents
- **LangChain** – Helps manage the RAG pipeline
- **LangChain Text Splitters** – Split documents into chunks
- **NumPy** – Numerical operations for vectors
- **Jinja2** – HTML template rendering
- **OpenAI** – Optional LLM integration for generating responses

---

## Project Workflow

1. Load documents from the `docs` folder.
2. Extract text from PDF/TXT files.
3. Split documents into smaller chunks.
4. Convert text chunks into embeddings.
5. Store embeddings in a FAISS vector database.
6. Receive user query via FastAPI endpoint.
7. Convert query into embedding.
8. Perform similarity search in FAISS.
9. Retrieve the most relevant document chunks.

---

## Project Structure
RAG_SYSTEM
│
├── docs/ # Input documents (PDF/TXT)
├── main.py # FastAPI application
├── rag_pipeline.py # Retrieval pipeline
├── config.py # Project configuration
├── chunk_experiment.py # Chunk size experiments
├── requirements.txt # Project dependencies

## Installation

Clone the repository:

```bash
git clone <repo-link>
cd RAG_SYSTEM

Install dependencies:

pip install -r requirements.txt
Running the API

Start the FastAPI server:

uvicorn main:app --reload

The API will run locally at:

http://127.0.0.1:8000
