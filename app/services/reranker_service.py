from threading import Lock

from sentence_transformers import CrossEncoder


# CrossEncoder model used for reranking
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded model
_reranker_model = None
_reranker_lock = Lock()


def get_reranker_model():
    """
    Load the CrossEncoder only when reranking is actually requested.

    Lazy loading reduces startup memory usage, which is important
    for low-memory hosting environments such as Render Free.
    """
    global _reranker_model

    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                _reranker_model = CrossEncoder(
                    MODEL_NAME,
                    device="cpu"
                )

    return _reranker_model


def rerank_results(
    query: str,
    results: list[dict]
) -> list[dict]:
    """
    Re-rank search results using a CrossEncoder model.

    Each result must contain a 'content' field.

    Returns the same results with an additional
    'rerank_score' field.
    """

    if not results:
        return []

    pairs = [
        (query, result["content"])
        for result in results
    ]

    # Load model only when required
    model = get_reranker_model()

    scores = model.predict(
        pairs,
        batch_size=2,
        show_progress_bar=False
    )

    reranked_results = []

    for result, score in zip(results, scores):

        updated_result = result.copy()

        updated_result["rerank_score"] = round(
            float(score),
            4
        )

        reranked_results.append(updated_result)

    # Highest score first
    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results


def rerank(
    query: str,
    documents: list[str]
) -> list[tuple[int, float]]:
    """
    Re-rank plain document strings.

    Returns:
        List of (original_index, score) tuples,
        sorted from highest score to lowest score.

    This function is used by the hybrid-search endpoint.
    """

    if not documents:
        return []

    pairs = [
        (query, document)
        for document in documents
    ]

    # Load model only when required
    model = get_reranker_model()

    scores = model.predict(
        pairs,
        batch_size=2,
        show_progress_bar=False
    )

    scored_results = [
        (index, float(score))
        for index, score in enumerate(scores)
    ]

    # Highest score first
    scored_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scored_results


if __name__ == "__main__":

    query = (
        "What programming languages "
        "does the candidate know?"
    )

    sample_results = [
        {
            "content": (
                "The candidate knows "
                "Python, Java and SQL."
            )
        },
        {
            "content": (
                "The candidate worked on "
                "an AI Resume Analyzer project."
            )
        },
        {
            "content": (
                "The candidate has experience "
                "with Flask and REST APIs."
            )
        }
    ]

    print("\nTesting rerank_results():\n")

    results = rerank_results(
        query,
        sample_results
    )

    for result in results:
        print(result)

    print("\nTesting rerank():\n")

    documents = [
        result["content"]
        for result in sample_results
    ]

    reranked = rerank(
        query,
        documents
    )

    for index, score in reranked:
        print(
            f"Index: {index}, "
            f"Score: {score:.4f}"
        )