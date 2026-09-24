import os
from threading import Lock

import torch


# ============================================================
# CONFIGURATION
# ============================================================

# Memory-efficient CrossEncoder.
#
# This keeps the CrossEncoder reranking feature while using
# substantially less memory than MiniLM-L6.
MODEL_NAME = (
    "cross-encoder/ms-marco-TinyBERT-L2-v2"
)


# ============================================================
# CPU / MEMORY CONFIGURATION
# ============================================================

try:
    torch.set_num_threads(
        int(os.getenv("TORCH_NUM_THREADS", "1"))
    )
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# RERANKER ENABLE / DISABLE
# ============================================================

ENABLE_RERANKER = (
    os.getenv(
        "ENABLE_RERANKER",
        "false",
    ).lower()
    in ("true", "1", "yes")
)


# ============================================================
# LAZY-LOADED MODEL
# ============================================================

_reranker_model = None
_reranker_lock = Lock()


# ============================================================
# GET RERANKER MODEL
# ============================================================

def get_reranker_model():

    global _reranker_model

    if not ENABLE_RERANKER:

        raise RuntimeError(
            "CrossEncoder reranker is disabled. "
            "Set ENABLE_RERANKER=true to enable it."
        )

    if _reranker_model is None:

        with _reranker_lock:

            if _reranker_model is None:

                from sentence_transformers import (
                    CrossEncoder
                )

                _reranker_model = CrossEncoder(
                    MODEL_NAME,
                    device="cpu",
                    max_length=256,
                )

                _reranker_model.model.eval()

    return _reranker_model


# ============================================================
# RERANK RESULT DICTIONARIES
# ============================================================

def rerank_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Re-rank search results using CrossEncoder.

    Each result must contain:
        content

    Returns:
        Results sorted by rerank_score.
    """

    if not results:
        return []

    if not ENABLE_RERANKER:
        return results

    pairs = [
        (
            query,
            result["content"],
        )
        for result in results
    ]

    model = get_reranker_model()

    with torch.inference_mode():

        scores = model.predict(
            pairs,
            batch_size=2,
            show_progress_bar=False,
        )

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):

        updated_result = result.copy()

        updated_result[
            "rerank_score"
        ] = round(
            float(score),
            4,
        )

        reranked_results.append(
            updated_result
        )

    reranked_results.sort(
        key=lambda item: item[
            "rerank_score"
        ],
        reverse=True,
    )

    return reranked_results


# ============================================================
# RERANK PLAIN DOCUMENTS
# ============================================================

def rerank(
    query: str,
    documents: list[str],
) -> list[tuple[int, float]]:
    """
    Re-rank plain document strings.

    Returns:

        [
            (original_index, score),
            ...
        ]

    Sorted from highest score to lowest score.
    """

    if not documents:
        return []

    if not ENABLE_RERANKER:

        return [
            (
                index,
                0.0,
            )
            for index in range(
                len(documents)
            )
        ]

    pairs = [
        (
            query,
            document,
        )
        for document in documents
    ]

    model = get_reranker_model()

    with torch.inference_mode():

        scores = model.predict(
            pairs,
            batch_size=2,
            show_progress_bar=False,
        )

    scored_results = [
        (
            index,
            float(score),
        )
        for index, score in enumerate(
            scores
        )
    ]

    scored_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_results


# ============================================================
# CHECK RERANKER STATUS
# ============================================================

def is_reranker_enabled():

    return ENABLE_RERANKER


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "CrossEncoder:",
        MODEL_NAME,
    )

    print(
        "Reranker enabled:",
        ENABLE_RERANKER,
    )

    if not ENABLE_RERANKER:

        print(
            "Reranker is disabled."
        )

    else:

        query = (
            "What programming languages "
            "does the candidate know?"
        )

        documents = [
            (
                "The candidate knows "
                "Python, Java and SQL."
            ),
            (
                "The candidate worked on "
                "an AI Resume Analyzer project."
            ),
            (
                "The candidate has experience "
                "with Flask and REST APIs."
            ),
        ]

        results = rerank(
            query,
            documents,
        )

        print()

        for index, score in results:

            print(
                f"Index: {index}, "
                f"Score: {score:.4f}"
            )