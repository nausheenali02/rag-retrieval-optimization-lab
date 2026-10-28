
from src.retrieval.document_loader import load_sample_documents
from src.retrieval.text_splitter import split_documents
from src.retrieval.bm25_retriever import BM25Retriever

# Load documents and split them into chunks.
documents = load_sample_documents()
chunks = split_documents(documents)

# Build the BM25 keyword index.
retriever = BM25Retriever(chunks)

question = "How does hybrid search improve information retrieval?"

# Retrieve the top 3 matching chunks.
results = retriever.search(question, k=3)

print(f"Question: {question}")
print(f"Retrieved chunks: {len(results)}")

for index, (document, score) in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(f"Source: {document.metadata['source']}")
    print(f"BM25 score: {score:.4f}")
    print(f"Content: {document.page_content}")
