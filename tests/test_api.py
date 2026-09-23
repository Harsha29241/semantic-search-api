import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# TEST CONFIGURATION
# ============================================================

API_KEY = os.getenv(
    "API_KEY",
    "semantic-search-dev-key-123",
)

client = TestClient(app)


# ============================================================
# HEALTH CHECK
# ============================================================

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Semantic Search API is running"
    assert data["version"] == "1.0.0"


# ============================================================
# API KEY - MISSING
# ============================================================

def test_documents_without_api_key():
    response = client.get("/documents")

    assert response.status_code == 401
    assert response.json()["detail"] == "API key is required"


# ============================================================
# API KEY - INVALID
# ============================================================

def test_documents_with_invalid_api_key():
    response = client.get(
        "/documents",
        headers={
            "X-API-Key": "wrong-api-key"
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key"


# ============================================================
# API KEY - VALID
# ============================================================

def test_documents_with_valid_api_key():
    response = client.get(
        "/documents",
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "document_count" in data
    assert "documents" in data

    assert isinstance(
        data["document_count"],
        int,
    )

    assert isinstance(
        data["documents"],
        list,
    )


# ============================================================
# GET SINGLE DOCUMENT
# ============================================================

def test_get_document():
    response = client.get(
        "/documents/1",
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert "filename" in data
    assert "chunk_count" in data


# ============================================================
# NON-EXISTENT DOCUMENT
# ============================================================

def test_nonexistent_document():
    response = client.get(
        "/documents/999999",
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


# ============================================================
# INVALID DOCUMENT ID
# ============================================================

def test_invalid_document_id():
    response = client.get(
        "/documents/0",
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Document ID must be a positive integer"
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def test_semantic_search():
    response = client.get(
        "/search",
        params={
            "q": "python programming",
            "limit": 5,
        },
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "python programming"
    assert "result_count" in data
    assert "results" in data

    assert isinstance(
        data["results"],
        list,
    )

    if data["results"]:
        result = data["results"][0]

        assert "chunk_id" in result
        assert "document_id" in result
        assert "filename" in result
        assert "chunk_index" in result
        assert "content" in result
        assert "similarity" in result


# ============================================================
# SEARCH WITHOUT API KEY
# ============================================================

def test_search_without_api_key():
    response = client.get(
        "/search",
        params={
            "q": "python"
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "API key is required"


# ============================================================
# SEARCH WITH INVALID DOCUMENT
# ============================================================

def test_search_invalid_document():
    response = client.get(
        "/search",
        params={
            "q": "python",
            "document_id": 999999,
        },
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 404


# ============================================================
# SEARCH QUERY VALIDATION
# ============================================================

def test_search_query_too_short():
    response = client.get(
        "/search",
        params={
            "q": "a"
        },
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 422


# ============================================================
# SEARCH LIMIT VALIDATION
# ============================================================

def test_search_limit_validation():
    response = client.get(
        "/search",
        params={
            "q": "python",
            "limit": 100,
        },
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 422


# ============================================================
# HYBRID SEARCH
# ============================================================

def test_hybrid_search():
    response = client.get(
        "/hybrid-search",
        params={
            "q": "python programming",
            "limit": 5,
        },
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "python programming"
    assert "result_count" in data
    assert "results" in data

    assert isinstance(
        data["results"],
        list,
    )

    if data["results"]:
        result = data["results"][0]

        assert "semantic_score" in result
        assert "keyword_score" in result
        assert "hybrid_score" in result
        assert "rerank_score" in result


# ============================================================
# HYBRID SEARCH WITHOUT API KEY
# ============================================================

def test_hybrid_search_without_api_key():
    response = client.get(
        "/hybrid-search",
        params={
            "q": "python"
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "API key is required"


# ============================================================
# CACHE STATISTICS
# ============================================================

def test_cache_stats():
    response = client.get(
        "/cache-stats",
        headers={
            "X-API-Key": API_KEY
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


# ============================================================
# CACHE STATS WITHOUT API KEY
# ============================================================

def test_cache_stats_without_api_key():
    response = client.get("/cache-stats")

    assert response.status_code == 401
    assert response.json()["detail"] == "API key is required"


# ============================================================
# RATE LIMITING
# ============================================================

def test_rate_limiting():
    responses = []

    for _ in range(31):
        response = client.get(
            "/documents",
            headers={
                "X-API-Key": API_KEY
            },
        )

        responses.append(response.status_code)

    assert 429 in responses