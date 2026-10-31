from src.evaluation.dataset import evaluation_dataset
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
)

from src.retrieval.semantic_retriever import search_documents
from src.retrieval.hybrid_retriever import HybridRetriever


K = 5

# Initialize retrievers only once.
hybrid_retriever = HybridRetriever()

vector_store = hybrid_retriever.vector_store
bm25_retriever = hybrid_retriever.bm25_retriever


# Store results for each retrieval method.
all_results = {
    "Semantic": [],
    "BM25": [],
    "Hybrid": [],
}

all_relevant_ids = []


# Evaluate every query.
for item in evaluation_dataset:

    query = item["query"]
    relevant_ids = item["relevant_ids"]

    all_relevant_ids.append(relevant_ids)

    # Semantic search.
    semantic_results = search_documents(
        vector_store, query, k=K
    )

    # BM25 search.
    bm25_results = bm25_retriever.search(
        query, k=K
    )

    # Hybrid search.
    hybrid_results = hybrid_retriever.search(
        query, k=K
    )

    # Convert retrieved documents into stable IDs.
    all_results["Semantic"].append([
        f"{doc.metadata['source']}:{doc.metadata['start_index']}"
        for doc, score in semantic_results
    ])

    all_results["BM25"].append([
        f"{doc.metadata['source']}:{doc.metadata['start_index']}"
        for doc, score in bm25_results
    ])

    all_results["Hybrid"].append([
        f"{doc.metadata['source']}:{doc.metadata['start_index']}"
        for doc, score in hybrid_results
    ])


# Per-query evaluation.
print("\n========== PER-QUERY EVALUATION ==========")

for index, item in enumerate(evaluation_dataset):

    print(f"\n{'=' * 60}")
    print(f"Query {index + 1}: {item['query']}")

    relevant_ids = all_relevant_ids[index]

    for method, query_results in all_results.items():

        retrieved_ids = query_results[index]

        precision = precision_at_k(
            retrieved_ids, relevant_ids, K
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, K
        )

        mrr = mean_reciprocal_rank(
            [retrieved_ids], [relevant_ids]
        )

        print(f"\n{method}:")
        print(f"  Precision@{K}: {precision:.4f}")
        print(f"  Recall@{K}:    {recall:.4f}")
        print(f"  MRR:           {mrr:.4f}")

        print("  Retrieved chunks:")

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            relevance = "Relevant" if chunk_id in relevant_ids else "Not relevant"
            print(f"    {rank}. {chunk_id} [{relevance}]")


# Overall evaluation.
print("\n\n========== OVERALL EVALUATION ==========")

for method, query_results in all_results.items():

    precision_scores = [
        precision_at_k(results, relevant, K)
        for results, relevant in zip(query_results, all_relevant_ids)
    ]

    recall_scores = [
        recall_at_k(results, relevant, K)
        for results, relevant in zip(query_results, all_relevant_ids)
    ]

    mrr_score = mean_reciprocal_rank(
        query_results, all_relevant_ids
    )

    print(f"\n--- {method} ---")
    print(f"Precision@{K}: {sum(precision_scores) / len(precision_scores):.4f}")
    print(f"Recall@{K}:    {sum(recall_scores) / len(recall_scores):.4f}")
    print(f"MRR:           {mrr_score:.4f}")