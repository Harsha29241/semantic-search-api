import fitz


def extract_text_from_pdf(file_path: str) -> str:
    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":
    text = extract_text_from_pdf("sample_documents/sample.pdf")
    print(text[:2000])