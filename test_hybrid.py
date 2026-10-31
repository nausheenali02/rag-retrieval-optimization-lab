from src.retrieval.hybrid_retriever import HybridRetriever

# Initialize the hybrid retriever.
retriever = HybridRetriever()

question = "How does hybrid search improve information retrieval?"

# Retrieve the top 3 hybrid results.
results = retriever.search(question, k=3)

print(f"Question: {question}")
print(f"Retrieved chunks: {len(results)}")

for index, (document, score) in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(f"Source: {document.metadata['source']}")
    print(f"RRF Score: {score:.6f}")
    print(f"Content: {document.page_content}")