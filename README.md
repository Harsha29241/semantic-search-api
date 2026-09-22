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
## Architecture

```mermaid
flowchart TD
    A[PDF Document Upload] --> B[FastAPI]
    B --> C[PDF Text Extraction]
    C --> D[Intelligent Chunking]
    D --> E[Sentence Transformer]
    E --> F[Vector Embeddings]

    F --> G[(PostgreSQL + pgvector)]

    H[User Query] --> I[Query Embedding]
    I --> G

    G --> J[Semantic Search]
    H --> K[Keyword Search]

    J --> L[Hybrid Scoring]
    K --> L

    L --> M[Cross-Encoder Re-ranking]
    M --> N[Ranked Search Results]

    B --> O[API Key Authentication]
    B --> P[Rate Limiting]
    B --> Q[Response Caching]