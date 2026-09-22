from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str):
    """
    Generate an embedding vector for a piece of text.
    """

    return model.encode(text)


if __name__ == "__main__":
    text = "Machine learning models learn patterns from training data."

    embedding = generate_embedding(text)

    print("Embedding type:", type(embedding))
    print("Embedding shape:", embedding.shape)
    print("First 10 values:", embedding[:10])