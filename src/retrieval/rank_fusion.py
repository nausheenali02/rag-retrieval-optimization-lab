from langchain_core.documents import Document

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Combine ranked document lists using Reciprocal Rank Fusion."""

    fused_scores = {}
    documents = {}

    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):

            document_id = (
                f"{document.metadata.get('source', '')}:"
                f"{document.page_content}"
            )

            documents[document_id] = document

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

    sorted_ids = sorted(
        fused_scores,
        key=lambda document_id: fused_scores[document_id],
        reverse=True,
    )

    return [
        (documents[document_id], fused_scores[document_id])
        for document_id in sorted_ids
    ]