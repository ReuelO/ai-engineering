from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "data" / "documents"


def load_documents() -> list[str]:
    """Load all text documents from the documents directory."""

    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        if text := path.read_text(encoding="utf-8").strip():
            documents.append(text)

    return documents
