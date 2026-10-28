from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Retrive document chunks using keyword-based BM25 ranking."""
    def __init__(self, documents: list[Document]):
        self.documents = documents

        # Convert each chunk into lowercase word tokens.
        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        # Build the BM25 index.
        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        """Return the top-k chunks ranked by BM25 score."""

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            (self.documents[index], float(scores[index]))
            for index in ranked_indices
        ]
