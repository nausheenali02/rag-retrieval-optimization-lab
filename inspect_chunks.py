from src.retrieval.document_loader import load_sample_documents
from src.retrieval.text_splitter import split_documents

# Load and split documents.
documents = load_sample_documents()
chunks = split_documents(documents)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    source = chunk.metadata["source"]
    start_index = chunk.metadata["start_index"]

    # Create a stable ID for each chunk.
    chunk_id = f"{source}:{start_index}"

    print(f"\nChunk ID: {chunk_id}")
    print(f"Content: {chunk.page_content}")