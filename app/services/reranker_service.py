
import os
from threading import Lock


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L2-v2"


# ============================================================
# RERANKER ENABLE / DISABLE
# ============================================================

ENABLE_RERANKER = (
    os.getenv("ENABLE_RERANKER", "false").lower()
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
    """
    Load the CrossEncoder reranker only when it is enabled
    and actually needed.

    This keeps Torch and the reranker model completely out
    of the normal application startup path when disabled.
    """

    global _reranker_model

    if not ENABLE_RERANKER:
        raise RuntimeError(
            "CrossEncoder reranker is disabled. "
            "Set ENABLE_RERANKER=true to enable it."
        )

    if _reranker_model is None:

        with _reranker_lock:

            if _reranker_model is None:

                # Import only when reranker is actually enabled.
                import torch
                from sentence_transformers import CrossEncoder

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
    Re-rank dictionary results using CrossEncoder.

    Each result must contain:
        content

    If reranker is disabled, the original results are returned
    unchanged.
    """

    if not results:
        return []

    # Keep reranker disabled by default.
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

    import torch

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

        updated_result["rerank_score"] = round(
            float(score),
            4,
        )

        reranked_results.append(
            updated_result
        )

    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked_results


# ============================================================
# RERANK PLAIN DOCUMENTS
# =============================================
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

    When reranker is disabled, returns the original indexes
    with a score of 0.0 without loading Torch.
    """

    if not documents:
        return []

    # IMPORTANT:
    # Keep reranker disabled on Render by default.
    if not ENABLE_RERANKER:
        return [
            (index, 0.0)
            for index in range(len(documents))
        ]

    pairs = [
        (query, document)
        for document in documents
    ]

    model = get_reranker_model()

    import torch

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
        for index, score in enumerate(scores)
    ]

    scored_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_results


# ============================================================
# STATUS
# ============================================================

def is_reranker_enabled() -> bool:
    """
    Return whether CrossEncoder reranking is enabled.
    """

    return ENABLE_RERANKER