"""Backward-compatible wrappers for the original prototype API."""

import os
from typing import Dict, List

from core.engine import extract_concepts, generate_bridge
from core.storage import GardenStore


def extract_concepts_node(input_text: str, llm_api_key: str | None = None) -> List[str]:
    if llm_api_key and not os.getenv("GARDEN_API_KEY"):
        os.environ["GARDEN_API_KEY"] = llm_api_key
    return extract_concepts(input_text)


def generate_bridge_node(concept: str, vectorstore_path: str = "", llm_api_key: str | None = None) -> Dict:
    if llm_api_key and not os.getenv("GARDEN_API_KEY"):
        os.environ["GARDEN_API_KEY"] = llm_api_key
    card = generate_bridge(GardenStore(), concept, "前沿材料", concept)
    refs = "\n---\n".join(ref.get("snippet", "") for ref in card["textbook_refs"])
    return {"concept": concept, "textbook_ref": refs, "bridge_card": card["explanation"]}
