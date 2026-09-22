from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embedding
from app.services.database_service import get_connection
from app.services.document_service import (
    create_document,
    get_document_by_filename
)


def ingest_document(filename: str, file_path: str):

    # Check whether this document already exists
    existing_document_id = get_document_by_filename(filename)

    if existing_document_id is not None:
        return existing_document_id, True

    # 1. Extract text from PDF
    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise ValueError("PDF contains no readable text")

    # 2. Split extracted text into chunks
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No chunks were created from the PDF")

    # 3. Create document record
    document_id = create_document(filename)

    # 4. Store chunks and embeddings
    connection = get_connection()

    try:
        cursor = connection.cursor()

        for index, chunk in enumerate(chunks):

            embedding = generate_embedding(chunk)

            # Convert NumPy array into pgvector format
            embedding_string = "[" + ",".join(
                str(value) for value in embedding
            ) + "]"

            cursor.execute(
                """
                INSERT INTO document_chunks
                (
                    document_id,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s::vector)
                """,
                (
                    document_id,
                    index,
                    chunk,
                    embedding_string
                )
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()

    return document_id, False