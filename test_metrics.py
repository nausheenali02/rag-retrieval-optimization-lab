from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)

# Sample retrieval results.
retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Ground-truth relevant documents.
relevant_ids = {"doc1", "doc3", "doc5", "doc6"}

# Test Precision@K.
precision = precision_at_k(retrieved_ids, relevant_ids, k=5)

# Test Recall@K.
recall = recall_at_k(retrieved_ids, relevant_ids, k=5)

# Test Reciprocal Rank.
rr = reciprocal_rank(retrieved_ids, relevant_ids)

# Test Mean Reciprocal Rank across multiple queries.
query_results = [
    ["doc1", "doc2", "doc3"],
    ["doc4", "doc5", "doc6"],
    ["doc7", "doc8", "doc9"],
]

relevant_documents = [
    {"doc1", "doc3"},
    {"doc5"},
    {"doc9"},
]

mrr = mean_reciprocal_rank(
    query_results,
    relevant_documents,
)

print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")
print(f"Reciprocal Rank: {rr:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")