import os

from fastapi.testclient import TestClient

from app.main import app
from app.services.cache_service import (
    make_cache_key,
    set_semantic_cache,
    get_semantic_cache,
    clear_search_cache,
    get_cache_stats,
)


client = TestClient(app)

API_KEY = os.getenv(
    "API_KEY",
    "semantic-search-dev-key-123"
)

HEADERS = {
    "X-API-Key": API_KEY
}


# ============================================================
# BASIC API TESTS
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Semantic Search API is running"
    assert data["version"] == "1.0.0"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============================================================
# AUTHENTICATION TESTS
# ============================================================

def test_documents_without_api_key():
    response = client.get("/documents")

    assert response.status_code == 401

    assert response.json()["detail"] == "API key is required"


def test_documents_with_invalid_api_key():
    response = client.get(
        "/documents",
        headers={
            "X-API-Key": "wrong-key"
        }
    )

    assert response.status_code == 403

    assert response.json()["detail"] == "Invalid API key"


def test_documents_with_valid_api_key():
    response = client.get(
        "/documents",
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()

    assert "document_count" in data
    assert "documents" in data

    assert isinstance(
        data["document_count"],
        int
    )

    assert isinstance(
        data["documents"],
        list
    )


# ============================================================
# DOCUMENT TESTS
# ============================================================

def test_existing_document():
    response = client.get(
        "/documents/1",
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "filename" in data
    assert "chunk_count" in data

    assert data["id"] == 1


def test_non_existing_document():
    response = client.get(
        "/documents/999999",
        headers=HEADERS
    )

    assert response.status_code == 404

    assert response.json()["detail"] == "Document not found"


def test_invalid_document_id():
    response = client.get(
        "/documents/0",
        headers=HEADERS
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Document ID must be a positive integer"
    )


# ============================================================
# SEARCH VALIDATION TESTS
# ============================================================

def test_search_requires_api_key():
    response = client.get(
        "/search",
        params={
            "q": "python"
        }
    )

    assert response.status_code == 401


def test_search_invalid_document():
    response = client.get(
        "/search",
        params={
            "q": "python",
            "document_id": 999999
        },
        headers=HEADERS
    )

    assert response.status_code == 404

    assert (
        "Document with ID 999999 not found"
        in response.json()["detail"]
    )


def test_search_query_too_short():
    response = client.get(
        "/search",
        params={
            "q": "a"
        },
        headers=HEADERS
    )

    assert response.status_code == 422


# ============================================================
# CACHE SERVICE TESTS
# ============================================================

def test_cache_key_normalization():

    key1 = make_cache_key(
        "  Python   Programming  ",
        5,
        1
    )

    key2 = make_cache_key(
        "python programming",
        5,
        1
    )

    assert key1 == key2


def test_semantic_cache():

    clear_search_cache()

    key = make_cache_key(
        "test query",
        5,
        1
    )

    value = {
        "query": "test query",
        "result_count": 0,
        "results": []
    }

    # Cache should initially be empty.
    assert get_semantic_cache(key) is None

    # Store value.
    set_semantic_cache(
        key,
        value
    )

    # Retrieve value.
    cached_value = get_semantic_cache(
        key
    )

    assert cached_value == value

    # Clean up.
    clear_search_cache()


def test_cache_stats():

    clear_search_cache()

    stats = get_cache_stats()

    assert "semantic_cache_size" in stats
    assert "hybrid_cache_size" in stats
    assert "max_size" in stats
    assert "ttl_seconds" in stats

    assert stats["semantic_cache_size"] == 0
    assert stats["hybrid_cache_size"] == 0

    assert stats["max_size"] == 100
    assert stats["ttl_seconds"] == 300


# ============================================================
# CACHE API ENDPOINT
# ============================================================

def test_cache_stats_endpoint():

    response = client.get(
        "/cache-stats",
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()

    assert "semantic_cache_size" in data
    assert "hybrid_cache_size" in data
    assert "max_size" in data
    assert "ttl_seconds" in data
    # ============================================================
# HYBRID SEARCH API
# ============================================================

def test_hybrid_search_endpoint():

    response = client.get(
        "/hybrid-search",
        params={
            "q": "What programming languages does the candidate know?",
            "limit": 5,
            "document_id": 1
        },
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == (
        "What programming languages does the candidate know?"
    )

    assert data["document_id"] == 1

    assert "result_count" in data
    assert "results" in data

    assert isinstance(data["results"], list)

    if data["results"]:

        result = data["results"][0]

        assert "chunk_id" in result
        assert "document_id" in result
        assert "filename" in result
        assert "chunk_index" in result
        assert "content" in result

        assert "semantic_score" in result
        assert "keyword_score" in result
        assert "hybrid_score" in result
        assert "rerank_score" in result


# ============================================================
# RATE LIMITING
# ============================================================

def test_rate_limit():

    rate_limit_hit = False

    for _ in range(35):

        response = client.get(
            "/documents/1",
            headers=HEADERS
        )

        if response.status_code == 429:

            rate_limit_hit = True

            break

        assert response.status_code == 200

    assert rate_limit_hit is True