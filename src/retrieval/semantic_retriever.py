from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.retrieval.document_loader import load_sample_documents
from src.retrieval.text_splitter import split_documents
from src.retrieval.vector_store import create_vector_store

def build_retriever() -> Chroma:
    """Load, split, and index sample documents."""
    documents = load_sample_documents()
    chunks = split_documents(documents)
    return create_vector_store(chunks)

def search_documents(
    vector_store: Chroma,
    question: str,
    k: int = 3,
    source: str | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve chunks, optionally filtering by source filename."""

    search_filter = {"source": source} if source else None

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=search_filter,
    )

    return results
