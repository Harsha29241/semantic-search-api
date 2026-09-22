from app.services.embedding_service import generate_embedding
from app.services.document_service import search_chunks


def main():
    query = "Python machine learning skills"

    query_embedding = generate_embedding(query)

    results = search_chunks(query_embedding, limit=5)

    print(f"Found {len(results)} results")

    for result in results:
        chunk_id, document_id, chunk_index, content, similarity = result

        print("\n--- Result ---")
        print("Chunk ID:", chunk_id)
        print("Document ID:", document_id)
        print("Chunk index:", chunk_index)
        print("Similarity:", round(similarity, 4))
        print("Content:", content[:300])


if __name__ == "__main__":
    main()