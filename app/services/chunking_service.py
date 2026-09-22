import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text before chunking.
    """

    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_long_text(text: str, chunk_size: int) -> list[str]:
    """
    Split text into smaller pieces when a paragraph
    is larger than the target chunk size.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (
                f"{current} {sentence}".strip()
            )
        else:
            if current:
                pieces.append(current)

            if len(sentence) <= chunk_size:
                current = sentence
            else:
                # Final fallback for extremely long sentences
                for start in range(0, len(sentence), chunk_size):
                    pieces.append(
                        sentence[start:start + chunk_size]
                    )

                current = ""

    if current:
        pieces.append(current)

    return pieces


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[str]:
    """
    Split text into overlapping chunks while
    preserving paragraph and sentence boundaries where possible.
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = clean_text(text)

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If the paragraph itself is too large,
        # split it into smaller pieces first.
        if len(paragraph) > chunk_size:
            pieces = split_long_text(
                paragraph,
                chunk_size
            )

            for piece in pieces:

                if (
                    len(current_chunk) + len(piece) + 1
                    <= chunk_size
                ):
                    current_chunk = (
                        f"{current_chunk}\n\n{piece}".strip()
                    )

                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    overlap_text = (
                        current_chunk[-overlap:]
                        if current_chunk
                        else ""
                    )

                    current_chunk = (
                        f"{overlap_text}\n\n{piece}".strip()
                        if overlap_text
                        else piece
                    )

            continue

        # Normal paragraph
        if (
            len(current_chunk) + len(paragraph) + 2
            <= chunk_size
        ):
            current_chunk = (
                f"{current_chunk}\n\n{paragraph}".strip()
            )

        else:
            if current_chunk:
                chunks.append(current_chunk)

            overlap_text = (
                current_chunk[-overlap:]
                if current_chunk
                else ""
            )

            current_chunk = (
                f"{overlap_text}\n\n{paragraph}".strip()
                if overlap_text
                else paragraph
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks