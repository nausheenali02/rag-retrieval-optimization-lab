from src.retrieval.semantic_retriever import search_documents
from src.retrieval.hybrid_retriever import HybridRetriever


# Common query
query = "How does hybrid search improve information retrieval?"

# Initialize both retrievers only once.
hybrid_retriever = HybridRetriever()

vector_store = hybrid_retriever.vector_store
bm25_retriever = hybrid_retriever.bm25_retriever


# 1. Semantic Search
semantic_results = search_documents(
    vector_store,
    query,
    k=3,
)

print("\n========== SEMANTIC SEARCH ==========")

for rank, (document, score) in enumerate(semantic_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Source: {document.metadata['source']}")
    print(f"Distance: {score:.4f}")
    print(f"Content: {document.page_content[:150]}")


# 2. BM25 Search
bm25_results = bm25_retriever.search(query, k=3)

print("\n========== BM25 SEARCH ==========")

for rank, (document, score) in enumerate(bm25_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Source: {document.metadata['source']}")
    print(f"BM25 Score: {score:.4f}")
    print(f"Content: {document.page_content[:150]}")


# 3. Hybrid Search
hybrid_results = hybrid_retriever.search(query, k=3)

print("\n========== HYBRID SEARCH (RRF) ==========")

for rank, (document, score) in enumerate(hybrid_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Source: {document.metadata['source']}")
    print(f"RRF Score: {score:.6f}")
    print(f"Content: {document.page_content[:150]}")