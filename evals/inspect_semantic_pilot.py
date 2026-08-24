from __future__ import annotations

from evals.adapter import load_cases
from core.semantic_index import semantic_search


if __name__ == "__main__":
    cases = load_cases("evals/datasets/retrieval_pilot_k27.jsonl")
    for case in cases:
        hits = semantic_search(case["question"], limit=100, kinds={"concept", "moc", "bridge", "knowledge", "course", "textbook"})
        expected = set(case["reference_titles"])
        rank = next((index for index, hit in enumerate(hits, 1) if hit["title"] in expected), None)
        expected_hit = next((hit for hit in hits if hit["title"] in expected), None)
        top = hits[0] if hits else {}
        print(
            f"{case['id']} expected_rank={rank} expected_score={expected_hit.get('semantic_score') if expected_hit else None} "
            f"top_score={top.get('semantic_score')} top={top.get('title')}"
        )
