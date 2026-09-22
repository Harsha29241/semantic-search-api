# Semantic Search API

An AI-powered document search and retrieval system built with FastAPI, PostgreSQL, pgvector, and Sentence Transformers.

The system allows users to upload PDF documents, extract and intelligently chunk their content, generate vector embeddings, store them in PostgreSQL, and retrieve relevant information using semantic and hybrid search.

---

## Features

- PDF document upload and text extraction
- Intelligent text chunking with overlap
- Local sentence-transformer embeddings
- Vector similarity search using pgvector
- Semantic document search
- Keyword + semantic hybrid search
- Cross-Encoder re-ranking
- Document-level filtering
- Search result caching with TTL
- API key authentication
- Request rate limiting
- Input validation and error handling
- Pydantic response validation
- PostgreSQL document and chunk metadata
- Automated API and service tests
- Docker and Docker Compose support
- Interactive Swagger API documentation

---

## Architecture

```text
                    Client
                      |
                      v
                FastAPI API
                      |
          +-----------+-----------+
          |                       |
          v                       v
     PDF Upload              Search Query
          |                       |
          v                       v
    Text Extraction        Query Embedding
          |                       |
          v                       v
    Intelligent Chunking    Vector Search
          |                       |
          v                       v
     Text Embeddings       Keyword Search
          |                       |
          v                       v
      PostgreSQL          Hybrid Scoring
       + pgvector               |
          |                     v
          |                Cross-Encoder
          |                 Re-ranking
          |                     |
          +----------+----------+
                     |
                     v
              Ranked Results