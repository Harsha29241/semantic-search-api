from threading import Lock
from cachetools import TTLCache


# Cache up to 100 search results.
# Each entry expires after 5 minutes.
CACHE_TTL_SECONDS = 300
CACHE_MAX_SIZE = 100


semantic_cache = TTLCache(
    maxsize=CACHE_MAX_SIZE,
    ttl=CACHE_TTL_SECONDS
)

hybrid_cache = TTLCache(
    maxsize=CACHE_MAX_SIZE,
    ttl=CACHE_TTL_SECONDS
)

cache_lock = Lock()


def normalize_query(query: str) -> str:
    """
    Normalize a search query so that small formatting
    differences do not create separate cache entries.
    """

    return " ".join(query.lower().strip().split())


def make_cache_key(
    query: str,
    limit: int,
    document_id: int | None
) -> tuple:

    return (
        normalize_query(query),
        limit,
        document_id
    )


def get_semantic_cache(key):
    with cache_lock:
        return semantic_cache.get(key)


def set_semantic_cache(key, value):
    with cache_lock:
        semantic_cache[key] = value


def get_hybrid_cache(key):
    with cache_lock:
        return hybrid_cache.get(key)


def set_hybrid_cache(key, value):
    with cache_lock:
        hybrid_cache[key] = value


def clear_search_cache():
    """
    Clear both semantic and hybrid search caches.
    Called after a new document is indexed.
    """

    with cache_lock:
        semantic_cache.clear()
        hybrid_cache.clear()


def get_cache_stats():
    """
    Return basic cache statistics.
    """

    with cache_lock:
        return {
            "semantic_cache_size": len(semantic_cache),
            "hybrid_cache_size": len(hybrid_cache),
            "max_size": CACHE_MAX_SIZE,
            "ttl_seconds": CACHE_TTL_SECONDS
        }