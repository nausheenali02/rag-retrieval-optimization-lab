from src.retrieval.semantic_retriever import (
    build_retriever,
    search_documents,
)

vector_store = build_retriever()

question = "How does hybrid search improve information retrieval?"

# Search across all documents
all_results = search_documents(
    vector_store,
    question,
    k=3,
)

print("\n=== Unfiltered Semantic Search ===")

for index, (document, score) in enumerate(all_results, start=1):
    print(f"\nResult {index}")
    print(f"Source: {document.metadata['source']}")
    print(f"Distance: {score:.4f}")
    print(f"Content: {document.page_content[:120]}...")

# Search only document_2.txt
filtered_results = search_documents(
    vector_store,
    question,
    k=3,
    source="document_2.txt",
)

print("\n=== Filtered Search: document_2.txt ===")

for index, (document, score) in enumerate(filtered_results, start=1):
    print(f"\nResult {index}")
    print(f"Source: {document.metadata['source']}")
    print(f"Distance: {score:.4f}")
    print(f"Content: {document.page_content[:120]}...")
