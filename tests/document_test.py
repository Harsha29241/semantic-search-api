from app.services.document_service import create_document


def main():
    document_id = create_document("sample.pdf")

    print("Created document ID:", document_id)


if __name__ == "__main__":
    main()