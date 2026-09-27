from fastembed import TextEmbedding


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LAZY-LOADED MODEL
# ============================================================

_model = None


# ============================================================
# GET EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Load the FastEmbed model only when needed.
    The model is reused after the first load.
    """

    global _model

    if _model is None:
        _model = TextEmbedding(
            model_name=MODEL_NAME
        )

    return _model


# ============================================================
# GENERATE SINGLE EMBEDDING
# ============================================================

def generate_embedding(text: str):
    """
    Generate a 384-dimensional embedding
    for a single text string.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    text = text.strip()

    model = get_embedding_model()

    embedding = next(
        model.embed([text])
    )

    return embedding


# ============================================================
# GENERATE MULTIPLE EMBEDDINGS
# ============================================================

def generate_embeddings(texts: list[str]):
    """
    Generate embeddings for multiple text chunks.

    Returns:
        list of numpy arrays
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

    embeddings = list(
        model.embed(
            cleaned_texts
        )
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