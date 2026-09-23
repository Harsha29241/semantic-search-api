# Semantic Search API

An AI-powered document search and retrieval system built with FastAPI, PostgreSQL, pgvector, and Sentence Transformers.

The system allows users to upload PDF documents, extract their text, intelligently split the text into chunks, generate vector embeddings, store the embeddings in PostgreSQL, and retrieve relevant document sections using semantic similarity.

It also supports hybrid search using semantic similarity and keyword matching, followed by CrossEncoder re-ranking for improved relevance.

---

## Features

- PDF document upload and processing
- PDF text extraction using PyMuPDF
- Intelligent text chunking with overlap
- Sentence Transformer embeddings
- Vector storage using PostgreSQL + pgvector
- Semantic similarity search
- Keyword-based matching
- Hybrid semantic + keyword search
- CrossEncoder re-ranking
- Document-level search filtering
- API key authentication
- Request validation
- Rate limiting with SlowAPI
- In-memory response caching
- Centralized logging
- Global error handling
- Automated API tests with pytest
- Docker and Docker Compose support
- Interactive API documentation with Swagger UI

---

## Architecture

```text
                    ┌─────────────────────┐
                    │       Client        │
                    │  Swagger / cURL     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │   REST API Layer    │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │ PDF Upload  │   │   Semantic  │   │   Hybrid    │
      │ & Ingestion │   │   Search    │   │   Search    │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │   PyMuPDF   │   │ Sentence    │   │  Keyword +  │
      │ Text Extract │   │ Transformers│   │  Semantic    │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             ▼                 │                 ▼
      ┌─────────────┐           │        ┌─────────────┐
      │  Chunking   │           │        │ CrossEncoder│
      │  + Overlap  │           │        │  Re-ranking │
      └──────┬──────┘           │        └──────┬──────┘
             │                  │               │
             ▼                  ▼               │
          ┌───────────────────────────────┐     │
          │      PostgreSQL + pgvector    │◄────┘
          │                               │
          │ Documents                     │
          │ Document Chunks               │
          │ Vector Embeddings             │
          └───────────────────────────────┘
```

---

## Search Pipeline

### Document ingestion

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Text Cleaning
 │
 ▼
Intelligent Chunking
 │
 ▼
Sentence Transformer
 │
 ▼
Vector Embedding
 │
 ▼
PostgreSQL + pgvector
```

### Semantic search

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
pgvector Similarity Search
    │
    ▼
Top Relevant Chunks
```

### Hybrid search

```text
                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Semantic Search        Keyword Matching
          │                     │
          └──────────┬──────────┘
                     ▼
              Hybrid Scoring
                     │
                     ▼
             Candidate Results
                     │
                     ▼
             CrossEncoder
              Re-ranking
                     │
                     ▼
             Final Results
```

---

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning / NLP

- Sentence Transformers
- `all-MiniLM-L6-v2`
- CrossEncoder
- `ms-marco-MiniLM-L-6-v2`

### Document Processing

- PyMuPDF

### Database

- PostgreSQL
- pgvector
- psycopg2

### Security & API Protection

- API key authentication
- Request validation
- SlowAPI rate limiting

### Performance

- TTL-based in-memory caching
- Vector similarity search

### Testing

- pytest
- FastAPI TestClient
- httpx

### Deployment

- Docker
- Docker Compose

---

## Project Structure

```text
semantic-search-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── cache_service.py
│       ├── chunking_service.py
│       ├── database_service.py
│       ├── document_service.py
│       ├── embedding_service.py
│       ├── ingestion_service.py
│       ├── logging_service.py
│       ├── pdf_service.py
│       └── reranker_service.py
│
├── sample_documents/
│   └── sample.pdf
│
├── tests/
│   └── test_api.py
│
├── uploaded_documents/
│
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── init.sql
├── README.md
└── requirements.txt
```

---

## Database Design

The application uses PostgreSQL with the pgvector extension.

### Documents

Stores information about uploaded documents.

```text
documents
├── id
└── filename
```

### Document Chunks

Stores the individual text chunks and their vector embeddings.

```text
document_chunks
├── id
├── document_id
├── chunk_index
├── content
└── embedding
```

The `embedding` column stores the vector representation generated by the Sentence Transformer model.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harsha29241/semantic-search-api.git
cd semantic-search-api
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
API_KEY=your-api-key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=semantic_search
DB_USER=your-user
DB_PASSWORD=your-password
```

Do not commit `.env` to GitHub.

A template is provided in:

```text
.env.example
```

---

## Running with Docker

The project includes Docker Compose for running both the API and PostgreSQL database.

```bash
docker compose up -d --build
```

Check the containers:

```bash
docker compose ps
```

The API runs on:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Stop the containers:

```bash
docker compose down
```

---

## Running Locally

If PostgreSQL is already configured locally:

```bash
./venv/bin/uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Upload Document

```http
POST /upload
```

Requires the API key.

Example:

```bash
curl -X POST \
  -H "X-API-Key: your-api-key" \
  -F "file=@sample_documents/sample.pdf" \
  http://127.0.0.1:8000/upload
```

The upload pipeline:

```text
PDF
→ Text Extraction
→ Chunking
→ Embedding Generation
→ Vector Storage
```

---

### Semantic Search

```http
GET /search
```

Example:

```bash
curl \
  -H "X-API-Key: your-api-key" \
  "http://127.0.0.1:8000/search?q=machine%20learning"
```

The query is converted into an embedding and compared against stored document embeddings using pgvector.

---

### Hybrid Search

```http
GET /hybrid-search
```

Example:

```bash
curl \
  -H "X-API-Key: your-api-key" \
  "http://127.0.0.1:8000/hybrid-search?q=python%20machine%20learning"
```

Hybrid search combines:

```text
Semantic similarity
        +
Keyword matching
        ↓
Hybrid score
        ↓
CrossEncoder re-ranking
```

---

### List Documents

```http
GET /documents
```

Example:

```bash
curl \
  -H "X-API-Key: your-api-key" \
  http://127.0.0.1:8000/documents
```

---

### Get Document

```http
GET /documents/{document_id}
```

Example:

```bash
curl \
  -H "X-API-Key: your-api-key" \
  http://127.0.0.1:8000/documents/1
```

---

### Cache Statistics

```http
GET /cache-stats
```

Example:

```bash
curl \
  -H "X-API-Key: your-api-key" \
  http://127.0.0.1:8000/cache-stats
```

Example response:

```json
{
  "semantic_cache_size": 1,
  "hybrid_cache_size": 1,
  "max_size": 100,
  "ttl_seconds": 300
}
```

---

## Authentication

Protected endpoints require an API key using the `X-API-Key` header.

Example:

```http
X-API-Key: your-api-key
```

The API key is loaded from the environment rather than being hard-coded into the application.

---

## Rate Limiting

The API uses SlowAPI to limit excessive requests.

Current limits include:

```text
POST /upload
10 requests/minute

GET /search
30 requests/minute

GET /hybrid-search
20 requests/minute

Other protected endpoints
30 requests/minute
```

Excessive requests receive:

```http
429 Too Many Requests
```

---

## Caching

The application uses an in-memory TTL cache for repeated search requests.

Configuration:

```text
Maximum entries: 100
TTL: 300 seconds
```

Separate caches are maintained for semantic and hybrid search results.

This reduces repeated embedding generation and database search work for identical queries during the cache lifetime.

---

## Re-ranking

Hybrid search uses a CrossEncoder after the initial candidate retrieval stage.

The purpose of the re-ranking stage is to evaluate the relationship between:

```text
Query + Candidate Document Chunk
```

and produce a relevance score for ordering the final search results.

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Testing

Run the complete test suite:

```bash
PYTHONPATH=. ./venv/bin/python3 -m pytest tests/ -v
```

The API test suite covers:

- Health endpoint
- Root endpoint
- Authentication
- Invalid API keys
- Document listing
- Document retrieval
- Invalid document IDs
- Semantic search
- Search validation
- Hybrid search
- Cache statistics
- Rate limiting

---

## Docker Health Check

Check the running API:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

Check containers:

```bash
docker compose ps
```

View API logs:

```bash
docker compose logs api --tail=50
```

View database logs:

```bash
docker compose logs db --tail=50
```

---

## Interactive API Documentation

FastAPI automatically provides Swagger UI:

```text
http://localhost:8000/docs
```

and ReDoc:

```text
http://localhost:8000/redoc
```

The Swagger interface can be used to test the API endpoints interactively.

---

## Future Improvements

Potential future improvements include:

- Support for additional document formats
- Improved metadata filtering
- Batch document ingestion
- Persistent distributed caching
- Background document processing
- Authentication using JWT
- Advanced hybrid ranking algorithms
- Production deployment
- Monitoring and metrics

---

## Project Objective

The project demonstrates how modern semantic retrieval systems can combine:

```text
Document Processing
        +
NLP Embeddings
        +
Vector Databases
        +
Semantic Search
        +
Keyword Search
        +
Re-ranking
        +
REST APIs
        +
Caching
        +
Rate Limiting
        +
Containerization
```

to build an end-to-end document retrieval system.

---

## Author

**Harshavardhan Reddy**

B.Tech — Artificial Intelligence & Machine Learning

SRM Institute of Science and Technology