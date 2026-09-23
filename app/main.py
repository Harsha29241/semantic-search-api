import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Header,
    Request,
)

from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.services.ingestion_service import ingest_document
from app.services.embedding_service import generate_embedding
from app.services.logging_service import configure_logging

from app.services.document_service import (
    search_chunks,
    get_all_documents,
    get_document_by_id,
    get_document_by_filename,
)

from app.services.cache_service import (
    make_cache_key,
    get_semantic_cache,
    set_semantic_cache,
    get_hybrid_cache,
    set_hybrid_cache,
    clear_search_cache,
    get_cache_stats,
)

from app.services.reranker_service import rerank


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API_KEY is not configured. "
        "Create a .env file with API_KEY=your-key"
    )


# ============================================================
# RATE LIMITER
# ============================================================

limiter = Limiter(
    key_func=get_remote_address
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Semantic Search API",
    description="AI-powered document search and retrieval system",
    version="1.0.0",
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# ============================================================
# FILE STORAGE
# ============================================================

UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# RESPONSE MODELS
# ============================================================

class UploadResponse(BaseModel):
    message: str
    document_id: int
    filename: str


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    document_id: Optional[int]
    result_count: int
    results: List[SearchResult]


class HybridSearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: float


class HybridSearchResponse(BaseModel):
    query: str
    document_id: Optional[int]
    result_count: int
    results: List[HybridSearchResult]


class DocumentSummary(BaseModel):
    id: int
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    document_count: int
    documents: List[DocumentSummary]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_api_key(
    x_api_key: Optional[str]
):
    """
    Validate the API key supplied through X-API-Key header.
    """

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Semantic Search API is running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    # --------------------------------------------------------
    # Filename validation
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    safe_filename = Path(file.filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    # --------------------------------------------------------
    # Duplicate document check
    # --------------------------------------------------------

    try:
        existing_document_id = get_document_by_filename(
            safe_filename
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to check existing documents: "
                f"{str(error)}"
            ),
        )

    if existing_document_id is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Document already exists",
                "document_id": existing_document_id,
                "filename": safe_filename,
            },
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:
        content = await file.read()

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read uploaded file: "
                f"{str(error)}"
            ),
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # --------------------------------------------------------
    # File size validation
    # --------------------------------------------------------

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB limit",
        )

    # --------------------------------------------------------
    # PDF signature validation
    # --------------------------------------------------------

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF",
        )

    # --------------------------------------------------------
    # Save file
    # --------------------------------------------------------

    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save file: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Ingest document
    # --------------------------------------------------------

    try:
        document_id = ingest_document(
            safe_filename,
            str(file_path),
        )

    except Exception as error:

        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Document ingestion failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Clear search caches
    # --------------------------------------------------------

    clear_search_cache()

    return {
        "message": "Document uploaded and indexed successfully",
        "document_id": document_id,
        "filename": safe_filename,
    }


# ============================================================
# SEMANTIC SEARCH
# ============================================================

@app.get(
    "/search",
    response_model=SearchResponse,
)
@limiter.limit("30/minute")
def search(
    request: Request,
    q: str = Query(
        ...,
        min_length=2,
        description="Search query",
    ),
    limit: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of results",
    ),
    document_id: Optional[int] = Query(
        None,
        ge=1,
        description="Optional document ID filter",
    ),
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    q = q.strip()

    if len(q) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Search query must contain "
                "at least 2 characters"
            ),
        )

    # --------------------------------------------------------
    # Validate document
    # --------------------------------------------------------

    if document_id is not None:

        document = get_document_by_id(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document with ID {document_id} "
                    "not found"
                ),
            )

    # --------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------

    cache_key = make_cache_key(
        query=q,
        limit=limit,
        document_id=document_id,
    )

    cached_results = get_semantic_cache(
        cache_key
    )

    if cached_results is not None:
        return {
            "query": q,
            "document_id": document_id,
            "result_count": len(cached_results),
            "results": cached_results,
        }

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    try:
        query_embedding = generate_embedding(q)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate query embedding: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Vector search
    # --------------------------------------------------------

    try:
        rows = search_chunks(
            query_embedding=query_embedding,
            limit=limit,
            document_id=document_id,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Search failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Format results
    # --------------------------------------------------------

    results = []

    for row in rows:

        results.append(
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "similarity": round(
                    float(row[5]),
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Store in cache
    # --------------------------------------------------------

    set_semantic_cache(
        cache_key,
        results,
    )

    return {
        "query": q,
        "document_id": document_id,
        "result_count": len(results),
        "results": results,
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

@app.get(
    "/hybrid-search",
    response_model=HybridSearchResponse,
)
@limiter.limit("20/minute")
def hybrid_search(
    request: Request,
    q: str = Query(
        ...,
        min_length=2,
        description="Search query",
    ),
    limit: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of results",
    ),
    document_id: Optional[int] = Query(
        None,
        ge=1,
        description="Optional document ID filter",
    ),
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    q = q.strip()

    if len(q) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Search query must contain "
                "at least 2 characters"
            ),
        )

    # --------------------------------------------------------
    # Validate document
    # --------------------------------------------------------

    if document_id is not None:

        document = get_document_by_id(
            document_id
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document with ID {document_id} "
                    "not found"
                ),
            )

    # --------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------

    cache_key = make_cache_key(
        query=q,
        limit=limit,
        document_id=document_id,
    )

    cached_results = get_hybrid_cache(
        cache_key
    )

    if cached_results is not None:
        return {
            "query": q,
            "document_id": document_id,
            "result_count": len(cached_results),
            "results": cached_results,
        }

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    try:
        query_embedding = generate_embedding(q)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate query embedding: "
                f"{str(error)}"
            ),
        )

    embedding_list = query_embedding.tolist()

    # --------------------------------------------------------
    # Retrieve semantic candidates
    # --------------------------------------------------------

    connection = None
    cursor = None

    try:

        from app.services.database_service import get_connection

        connection = get_connection()
        cursor = connection.cursor()

        candidate_limit = max(
            limit * 3,
            10,
        )

        if document_id is not None:

            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.content,
                    1 - (
                        c.embedding <=> %s::vector
                    ) AS semantic_score
                FROM document_chunks c
                JOIN documents d
                    ON c.document_id = d.id
                WHERE c.document_id = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    embedding_list,
                    document_id,
                    embedding_list,
                    candidate_limit,
                ),
            )

        else:

            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.content,
                    1 - (
                        c.embedding <=> %s::vector
                    ) AS semantic_score
                FROM document_chunks c
                JOIN documents d
                    ON c.document_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    embedding_list,
                    embedding_list,
                    candidate_limit,
                ),
            )

        rows = cursor.fetchall()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Hybrid search failed: "
                f"{str(error)}"
            ),
        )

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # --------------------------------------------------------
    # Hybrid scoring
    # --------------------------------------------------------

    query_terms = set(
        q.lower().split()
    )

    candidates = []

    for row in rows:

        content = row[4]
        content_lower = content.lower()

        keyword_matches = sum(
            1
            for term in query_terms
            if term in content_lower
        )

        keyword_score = (
            keyword_matches / len(query_terms)
            if query_terms
            else 0
        )

        semantic_score = float(row[5])

        # 70% semantic + 30% keyword
        hybrid_score = (
            0.7 * semantic_score
            + 0.3 * keyword_score
        )

        candidates.append(
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "content": content,
                "semantic_score": round(
                    semantic_score,
                    4,
                ),
                "keyword_score": round(
                    keyword_score,
                    4,
                ),
                "hybrid_score": round(
                    hybrid_score,
                    4,
                ),
            }
        )

    candidates.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # CrossEncoder re-ranking
    # --------------------------------------------------------

    rerank_candidates = candidates[
        :max(limit * 2, 10)
    ]

    if rerank_candidates:

        documents = [
            item["content"]
            for item in rerank_candidates
        ]

        try:

            reranked = rerank(
                q,
                documents,
            )

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Re-ranking failed: "
                    f"{str(error)}"
                ),
            )

        final_results = []

        for index, score in reranked:

            item = rerank_candidates[
                index
            ].copy()

            item["rerank_score"] = round(
                float(score),
                4,
            )

            final_results.append(item)

        final_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        final_results = final_results[
            :limit
        ]

    else:

        final_results = []

    # --------------------------------------------------------
    # Store hybrid results in cache
    # --------------------------------------------------------

    set_hybrid_cache(
        cache_key,
        final_results,
    )

    return {
        "query": q,
        "document_id": document_id,
        "result_count": len(final_results),
        "results": final_results,
    }


# ============================================================
# CACHE STATISTICS
# ============================================================

@app.get("/cache-stats")
@limiter.limit("30/minute")
def cache_stats(
    request: Request,
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    try:

        return get_cache_stats()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve cache statistics: "
                f"{str(error)}"
            ),
        )


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
@limiter.limit("30/minute")
def list_documents(
    request: Request,
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    try:

        rows = get_all_documents()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve documents: "
                f"{str(error)}"
            ),
        )

    documents = []

    for row in rows:

        documents.append(
            {
                "id": row[0],
                "filename": row[1],
                "chunk_count": row[2],
            }
        )

    return {
        "document_count": len(documents),
        "documents": documents,
    }


# ============================================================
# GET SINGLE DOCUMENT
# ============================================================

@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
)
@limiter.limit("30/minute")
def get_document(
    request: Request,
    document_id: int,
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    if document_id < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Document ID must be "
                "a positive integer"
            ),
        )

    try:

        document = get_document_by_id(
            document_id
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve document: "
                f"{str(error)}"
            ),
        )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "id": document[0],
        "filename": document[1],
        "chunk_count": document[2],
    }


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

configure_logging(app)