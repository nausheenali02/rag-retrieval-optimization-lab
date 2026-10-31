from pathlib import Path

from langchain_core.documents import Document

def load_sample_documents(data_dir: str = "data/sample_docs") -> list[Document]:
    """Load sample text files as LangChain Documents."""
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory.resolve()}")

    documents = []

    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name},
            )
        )

    if not documents:
        raise ValueError(f"No non-empty .txt files found in {directory.resolve()}")

    return documents
