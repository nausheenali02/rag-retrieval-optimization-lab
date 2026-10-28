
from src.retrieval.document_loader import load_sample_documents
from src.retrieval.text_splitter import split_documents


documents = load_sample_documents()
chunks = split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Total chunks created: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Start index: {chunk.metadata['start_index']}")
    print(f"Chunk length: {len(chunk.page_content)} characters")
    print(f"Content preview: {chunk.page_content[:100]}...")
