import os
from threading import Lock

import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# CPU / MEMORY CONFIGURATION
# ============================================================

# Keep PyTorch from creating too many CPU threads.
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
# LAZY-LOADED MODEL
# ============================================================

_model = None
_model_lock = Lock()


# ============================================================
# GET EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Load the embedding model only when it is actually needed.

    The model is loaded once and reused.

    CPU + low-memory loading is used to reduce memory pressure
    on low-memory hosting environments such as Render.
    """

    global _model

    if _model is None:

        with _model_lock:

            if _model is None:

                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu",
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                    },
                )

                _model.eval()

    return _model


# ============================================================
# GENERATE SINGLE EMBEDDING
# ============================================================

def generate_embedding(text: str):
    """
    Generate an embedding for a single text string.

    Returns:
        numpy.ndarray
    """

    if not text:
        raise ValueError(
            "Text cannot be empty"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Text cannot be empty"
        )

    model = get_embedding_model()

    with torch.inference_mode():

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embedding


# ============================================================
# GENERATE MULTIPLE EMBEDDINGS
# ============================================================

def generate_embeddings(
    texts: list[str],
):
    """
    Generate embeddings for multiple texts.

    This function is useful during document ingestion.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    model = get_embedding_model()

    with torch.inference_mode():

        embeddings = model.encode(
            cleaned_texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embeddings


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    text = (
        "Machine learning models learn "
        "patterns from training data."
    )

    embedding = generate_embedding(text)

    print()
    print("Embedding model:", MODEL_NAME)
    print("Embedding type:", type(embedding))
    print("Embedding shape:", embedding.shape)
    print("First 10 values:")
    print(embedding[:10])