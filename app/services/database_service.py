import os

from dotenv import load_dotenv
import psycopg2


load_dotenv()


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "semantic_search")
DB_USER = os.getenv("DB_USER", "harsha")
DB_PASSWORD = os.getenv("DB_PASSWORD")


if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD is not configured. "
        "Add DB_PASSWORD to your .env file."
    )


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )


def create_document(filename: str) -> int:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (filename)
                VALUES (%s)
                RETURNING id;
                """,
                (filename,)
            )

            document_id = cursor.fetchone()[0]

        connection.commit()

        return document_id

    finally:
        connection.close()


def create_chunk(
    document_id: int,
    chunk_index: int,
    content: str,
    embedding
):
    connection = get_connection()

    try:
        embedding_string = (
            "["
            + ",".join(str(value) for value in embedding)
            + "]"
        )

        with connection.cursor() as cursor:
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
                RETURNING id;
                """,
                (
                    document_id,
                    chunk_index,
                    content,
                    embedding_string
                )
            )

            chunk_id = cursor.fetchone()[0]

        connection.commit()

        return chunk_id

    finally:
        connection.close()


def search_chunks(
    query_embedding,
    limit: int = 5
):
    connection = get_connection()

    try:
        embedding_string = (
            "["
            + ",".join(str(value) for value in query_embedding)
            + "]"
        )

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    d.filename,
                    dc.chunk_index,
                    dc.content,
                    1 - (
                        dc.embedding <=> %s::vector
                    ) AS similarity
                FROM document_chunks dc
                JOIN documents d
                    ON dc.document_id = d.id
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    embedding_string,
                    embedding_string,
                    limit
                )
            )

            return cursor.fetchall()

    finally:
        connection.close()