from __future__ import annotations

import argparse
import json
import os
import time

from core.config import DB_PATH
from core.pdf_ocr import discover_scanned_textbooks, ocr_textbook_into_store
from core.retrieval import rebuild_domain_map
from core.storage import GardenStore


def main() -> None:
    os.environ.setdefault("GARDEN_DISABLE_NETWORK", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    parser = argparse.ArgumentParser(description="Offline resumable Windows OCR for scanned textbooks")
    parser.add_argument("--directory", default="")
    parser.add_argument("--books", default="", help="Comma-separated filename fragments")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--rebuild-semantic-index", action="store_true")
    args = parser.parse_args()
    store = GardenStore(DB_PATH)
    directory = args.directory or str(store.setting("textbook_directory", ""))
    includes = tuple(item.strip() for item in args.books.split(",") if item.strip())
    books = discover_scanned_textbooks(store, directory, includes=includes)
    print(json.dumps({
        "event": "started", "books": len(books),
        "remaining_pages": sum(int(book.get("remaining_pages", 0)) for book in books),
    }, ensure_ascii=False), flush=True)
    completed = []
    started = time.monotonic()
    for index, book in enumerate(books, 1):
        print(json.dumps({
            "event": "book_started", "book": index, "books": len(books),
            "title": book["title"], "pages": book.get("pages", 0),
            "resume_from": int(book.get("last_page", 0)) + 1,
        }, ensure_ascii=False), flush=True)

        def on_progress(result: dict) -> None:
            page = int(result.get("page", 0))
            if page % 10 == 0 or page == int(result.get("pages", 0)):
                print(json.dumps({
                    "event": "page_progress", "book": index,
                    "title": result["title"], "page": page,
                    "pages": result["pages"],
                    "indexed": result["indexed_in_run"],
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }, ensure_ascii=False), flush=True)

        outcome = ocr_textbook_into_store(
            store, book["path"], max_pages=max(0, args.max_pages), progress=on_progress,
        )
        completed.append(outcome)
        print(json.dumps({"event": "book_finished", **outcome}, ensure_ascii=False), flush=True)
        try:
            created = rebuild_domain_map(store)
            print(json.dumps({"event": "domain_map_updated", "created": created}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"event": "domain_map_error", "error": str(exc)[:240]}, ensure_ascii=False), flush=True)
    if args.rebuild_semantic_index and any(item.get("indexed") for item in completed):
        from core.semantic_index import build_semantic_index

        print(json.dumps({"event": "semantic_index_started"}, ensure_ascii=False), flush=True)
        print(json.dumps({
            "event": "semantic_index_finished", **build_semantic_index(store),
        }, ensure_ascii=False), flush=True)
    print(json.dumps({
        "event": "finished", "books": len(completed),
        "indexed_pages": sum(int(item.get("indexed", 0)) for item in completed),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
