from app.services.embedding_service import generate_embedding
from app.services.database_service import create_chunk


def main():
    content = "Machine learning is a field of artificial intelligence."

    embedding = generate_embedding(content)

    chunk_id = create_chunk(
        document_id=1,
        chunk_index=1,
        content=content,
        embedding=embedding
    )

    print("Created chunk ID:", chunk_id)


if __name__ == "__main__":
    main()