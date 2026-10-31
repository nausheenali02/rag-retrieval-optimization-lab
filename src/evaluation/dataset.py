# Ground-truth evaluation dataset.
# Each query is mapped to the IDs of its relevant chunks.

evaluation_dataset = [
    {
        "query": "What is Python used for?",
        "relevant_ids": {
            "document_1.txt:0"
        }
    },
    {
        "query": "What are the common tasks in machine learning?",
        "relevant_ids": {
            "document_1.txt:192",
            "document_1.txt:383"
        }
    },
    {
        "query": "What is supervised learning?",
        "relevant_ids": {
            "document_1.txt:383"
        }
    },
    {
        "query": "What is Retrieval-Augmented Generation and how does it work?",
        "relevant_ids": {
            "document_2.txt:0",
            "document_2.txt:241"
        }
    },
    {
        "query": "How do chunk size and overlap affect retrieval quality?",
        "relevant_ids": {
            "document_2.txt:387"
        }
    },
    {
        "query": "What is BM25 and how does it rank documents?",
        "relevant_ids": {
            "document_3.txt:199"
        }
    },
    {
        "query": "How does hybrid search combine semantic and keyword search?",
        "relevant_ids": {
            "document_3.txt:0",
            "document_3.txt:199"
        }
    },
    {
        "query": "What is reranking in RAG?",
        "relevant_ids": {
            "document_3.txt:377"
        }
    },
    {
        "query": "How can query rewriting and metadata filtering improve retrieval?",
        "relevant_ids": {
            "document_3.txt:579"
        }
    },
    {
        "query": "What is a vector database and what is Chroma used for?",
        "relevant_ids": {
            "document_2.txt:584"
        }
    }
]