from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embedding
from app.services.database_service import get_connection
from app.services.document_service import create_document


def ingest_document(filename: str, file_path: str) -> int:
    """
    Extract a PDF, create chunks, generate embeddings,
    and store everything in PostgreSQL.

    Returns:
        int: The newly created document ID.
    """

    # --------------------------------------------------------
    # 1. Extract text from PDF
    # --------------------------------------------------------

    text = extract_text_from_pdf(file_path)

    if not text or not text.strip():
        raise ValueError(
            "PDF contains no readable text"
        )

    # --------------------------------------------------------
    # 2. Split text into chunks
    # --------------------------------------------------------

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF"
        )

    # --------------------------------------------------------
    # 3. Create document record
    # --------------------------------------------------------

    document_id = create_document(
        filename
    )

    # --------------------------------------------------------
    # 4. Generate embeddings and store chunks
    # --------------------------------------------------------

    connection = get_connection()

    try:

        cursor = connection.cursor()

        for index, chunk in enumerate(chunks):

            # Generate embedding
            embedding = generate_embedding(
                chunk
            )

            # Convert NumPy array into pgvector format
            embedding_string = (
                "["
                + ",".join(
                    str(value)
                    for value in embedding
                )
                + "]"
            )

            cursor.execute(
                """
                INSERT INTO document_chunks
                (
                    document_id,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s::vector
                )
                """,
                (
                    document_id,
                    index,
                    chunk,
                    embedding_string,
                ),
            )

        # Commit everything
        connection.commit()

    except Exception:

        # Roll back if anything fails
        connection.rollback()

        raise

    finally:

        cursor.close()
        connection.close()

    # --------------------------------------------------------
    # 5. Return document ID
    # --------------------------------------------------------

    return document_id