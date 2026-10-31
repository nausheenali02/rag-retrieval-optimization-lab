from langchain_core.documents import Document

from src.retrieval.document_loader import load_sample_documents
from src.retrieval.text_splitter import split_documents
from src.retrieval.vector_store import create_vector_store
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.rank_fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Combine semantic and BM25 retrieval using RRF."""

    def __init__(self):
        # Load and split documents only once.
        documents = load_sample_documents()
        self.chunks = split_documents(documents)

        # Build both retrievers using the same chunks.
        self.vector_store = create_vector_store(self.chunks)
        self.bm25_retriever = BM25Retriever(self.chunks)

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Document, float]]:

        # Semantic retrieval.
        semantic_results = self.vector_store.similarity_search(
            query,
            k=k,
        )

        # BM25 retrieval.
        bm25_results = self.bm25_retriever.search(
            query,
            k=k,
        )

        # Extract BM25 documents.
        bm25_documents = [
            document for document, score in bm25_results
        ]

        # Combine rankings using RRF.
        fused_results = reciprocal_rank_fusion(
            [semantic_results, bm25_documents]
        )

        return fused_results[:k]