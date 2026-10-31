def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        1 for document_id in top_k
        if document_id in relevant_ids
    )

    return relevant_retrieved / k
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for document_id in top_k
        if document_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)

def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate Reciprocal Rank for a single query."""

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0

def mean_reciprocal_rank(
    query_results: list[list[str]],
    relevant_documents: list[set[str]],
) -> float:
    """Calculate MRR across multiple queries."""

    if len(query_results) != len(relevant_documents):
        raise ValueError("Results and relevance labels must match.")

    if not query_results:
        return 0.0

    scores = [
        reciprocal_rank(results, relevant)
        for results, relevant in zip(
            query_results,
            relevant_documents,
        )
    ]

    return sum(scores) / len(scores)