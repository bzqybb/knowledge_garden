from __future__ import annotations

import json
import argparse

from core.config import DB_PATH
from core.semantic_index import build_semantic_index
from core.storage import GardenStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略旧向量并完整重建")
    args = parser.parse_args()
    print(json.dumps(
        build_semantic_index(GardenStore(DB_PATH), force_rebuild=args.force),
        ensure_ascii=False,
        indent=2,
    ))
