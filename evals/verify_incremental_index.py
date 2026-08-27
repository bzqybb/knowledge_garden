from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from core.config import DB_PATH, TEMP_DIR
from core.semantic_index import build_semantic_index
from core.storage import GardenStore


if __name__ == "__main__":
    main_store = GardenStore(DB_PATH)
    with tempfile.TemporaryDirectory(
        prefix="garden-index-incremental-", dir=TEMP_DIR,
    ) as folder:
        copied_db = Path(folder) / "garden.db"
        shutil.copy2(DB_PATH, copied_db)
        changed_store = GardenStore(copied_db)
        content = "增量索引验证页：KCL 基于电荷守恒。本页仅用于本地自动测试。"
        changed_store.upsert_note({
            "path": "textbook/__incremental_index_probe__.md",
            "title": "增量索引验证页",
            "kind": "textbook",
            "content": content,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
        try:
            added = build_semantic_index(changed_store)
        finally:
            restored = build_semantic_index(main_store)
    print(json.dumps({"added": added, "restored": restored}, ensure_ascii=False, indent=2))
