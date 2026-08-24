from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WIKIPEDIA_API = "https://zh.wikipedia.org/w/api.php"


def search_wikipedia(query: str, limit: int = 3, timeout: int = 15) -> list[dict]:
    """Use Wikipedia as a terminology and orientation source, never as sole proof."""
    params = urlencode({
        "action": "query", "generator": "search", "gsrsearch": query.strip(),
        "gsrlimit": max(1, min(limit, 5)), "prop": "extracts|info",
        "exintro": 1, "explaintext": 1, "inprop": "url", "format": "json", "utf8": 1,
    })
    request = Request(
        f"{WIKIPEDIA_API}?{params}",
        headers={"User-Agent": "KnowledgeGarden/1.0 (local learning assistant)"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = list((payload.get("query") or {}).get("pages", {}).values())
    pages.sort(key=lambda item: int(item.get("index", 999)))
    return [
        {
            "title": str(page.get("title") or ""),
            "url": str(page.get("fullurl") or ""),
            "abstract": str(page.get("extract") or "")[:2200],
            "year": None, "authors": [], "venue": "Wikipedia",
            "source": "Wikipedia", "source_type": "encyclopedia",
        }
        for page in pages if page.get("title") and page.get("extract")
    ]
