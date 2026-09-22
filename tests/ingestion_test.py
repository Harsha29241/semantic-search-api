from app.services.ingestion_service import ingest_document


PDF_PATH = "sample_documents/sample.pdf"


def main():

    document_id = ingest_document(
        "sample.pdf",
        PDF_PATH
    )

    print("Document ID:", document_id)
    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()