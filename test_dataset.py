from src.evaluation.dataset import evaluation_dataset

print(f"Total evaluation queries: {len(evaluation_dataset)}")

for index, item in enumerate(evaluation_dataset, start=1):
    print(f"\nQuery {index}: {item['query']}")
    print(f"Relevant chunks: {item['relevant_ids']}")

    assert item["query"]
    assert item["relevant_ids"]

print("\nDataset validation successful!")