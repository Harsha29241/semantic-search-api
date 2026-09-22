from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embedding


PDF_PATH = "sample_documents/sample.pdf"


def main():
    text = extract_text_from_pdf(PDF_PATH)

    print(f"Extracted characters: {len(text)}")

    chunks = chunk_text(text)

    print(f"Number of chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        embedding = generate_embedding(chunk)

        print(f"\n--- Chunk {index} ---")
        print(f"Characters: {len(chunk)}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()