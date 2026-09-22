from app.services.database_service import (
    get_connection,
    create_chunk
)


def create_document(filename: str):
    connection = get_connection()

    try:
        cursor = connection.cursor()

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
        cursor.close()
        connection.close()


def get_document_by_filename(filename: str):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM documents
            WHERE filename = %s
            ORDER BY id DESC
            LIMIT 1;
            """,
            (filename,)
        )

        result = cursor.fetchone()

        if result:
            return result[0]

        return None

    finally:
        cursor.close()
        connection.close()


def search_chunks(
    query_embedding,
    limit: int = 5,
    document_id: int | None = None
):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        embedding_string = (
            "["
            + ",".join(str(value) for value in query_embedding)
            + "]"
        )

        if document_id is not None:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.content,
                    1 - (
                        c.embedding <=> %s::vector
                    ) AS similarity
                FROM document_chunks c
                JOIN documents d
                    ON c.document_id = d.id
                WHERE c.document_id = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    embedding_string,
                    document_id,
                    embedding_string,
                    limit
                )
            )

        else:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.content,
                    1 - (
                        c.embedding <=> %s::vector
                    ) AS similarity
                FROM document_chunks c
                JOIN documents d
                    ON c.document_id = d.id
                ORDER BY c.embedding <=> %s::vector
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
        cursor.close()
        connection.close()


def get_all_documents():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                d.id,
                d.filename,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c
                ON d.id = c.document_id
            GROUP BY d.id, d.filename
            ORDER BY d.id DESC;
            """
        )

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()


def get_document_by_id(document_id: int):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                d.id,
                d.filename,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c
                ON d.id = c.document_id
            WHERE d.id = %s
            GROUP BY d.id, d.filename;
            """,
            (document_id,)
        )

        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()