# Semantic Search API

AI-powered document search and retrieval system built with FastAPI, PostgreSQL, pgvector, and Sentence Transformers.

The system allows users to upload PDF documents, extract and intelligently chunk their text, generate vector embeddings, store them in PostgreSQL, and search the documents using semantic similarity.

It also includes hybrid search with keyword matching and CrossEncoder re-ranking to improve retrieval relevance.

---

## Features

- PDF document upload and processing
- PDF text extraction using PyMuPDF
- Intelligent text chunking with overlap
- Semantic embeddings using Sentence Transformers
- Vector storage using PostgreSQL + pgvector
- Semantic similarity search
- Hybrid keyword + semantic search
- CrossEncoder-based result re-ranking
- Document-level filtering
- API key authentication
- Request validation
- Rate limiting with SlowAPI
- In-memory TTL caching
- Centralized application logging
- REST API built with FastAPI
- Automatic Swagger/OpenAPI documentation
- Docker and Docker Compose support
- Automated API tests with pytest

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      Client         │
                    │  Browser / Postman  │
                    │      / cURL          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      REST API       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Authentication    Rate Limiting      Caching
              │                │                │
              └────────────────┼────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Document Ingestion             Search Pipeline
                 │                           │
        ┌────────┴────────┐          ┌───────┴────────┐
        │                 │          │                │
        ▼                 ▼          ▼                ▼
   PDF Extraction    Chunking    Embedding       Keyword Search
        │                 │          │                │
        └────────┬────────┘          └───────┬────────┘
                 │                           │
                 ▼                           ▼
        Sentence Transformer          Hybrid Scoring
                 │                           │
                 ▼                           ▼
          Vector Storage             CrossEncoder
                 │                    Re-ranking
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    PostgreSQL + pgvector
                               │
                               ▼
                       Ranked Search Results