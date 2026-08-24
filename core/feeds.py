from __future__ import annotations

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

from core.storage import GardenStore
from core.obsidian import parse_markdown, write_raw_material


def _text(element: ET.Element | None, names: list[str]) -> str:
    if element is None:
        return ""
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def fetch_feed(url: str, limit: int = 8) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "KnowledgeGarden/1.0 (+local learning assistant)"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(4_000_000)
    root = ET.fromstring(raw)
    entries: list[dict[str, Any]] = []
    if root.tag.lower().endswith("feed"):
        ns_match = re.match(r"\{(.+?)\}", root.tag)
        ns = f"{{{ns_match.group(1)}}}" if ns_match else ""
        for item in root.findall(f"{ns}entry")[:limit]:
            link_node = item.find(f"{ns}link")
            entries.append({
                "title": _text(item, [f"{ns}title"]),
                "url": (link_node.attrib.get("href", "") if link_node is not None else ""),
                "summary": _strip_html(_text(item, [f"{ns}summary", f"{ns}content"])),
                "published": _text(item, [f"{ns}published", f"{ns}updated"]),
            })
    else:
        channel = root.find("channel") or root
        for item in channel.findall("item")[:limit]:
            entries.append({
                "title": _text(item, ["title"]),
                "url": _text(item, ["link", "guid"]),
                "summary": _strip_html(_text(item, ["description", "content:encoded"])),
                "published": _text(item, ["pubDate", "date"]),
            })
    return [entry for entry in entries if entry["title"]]


def refresh_feeds(store: GardenStore) -> dict[str, int | list[str]]:
    fetched = added = 0
    errors: list[str] = []
    for feed in store.list_feeds():
        if not feed["enabled"]:
            continue
        try:
            entries = fetch_feed(feed["url"])
            fetched += len(entries)
            for entry in entries:
                key = entry["url"] or entry["title"]
                digest = hashlib.sha256((key + entry["summary"]).encode("utf-8")).hexdigest()
                note_payload = {
                    "path": f"feed::{feed['id']}::{hashlib.sha1(key.encode('utf-8')).hexdigest()}",
                    "title": entry["title"],
                    "kind": "frontier",
                    "content": entry["summary"] or f"来自 {feed['name']} 的新内容，点击来源阅读全文。",
                    "tags": ["前沿", feed["name"]],
                    "source": feed["name"],
                    "source_url": entry["url"],
                    "content_hash": digest,
                }
                vault = store.setting("vault_path", "")
                if vault:
                    try:
                        raw_path = write_raw_material(
                            vault, entry["title"], entry["summary"] or "等待阅读全文后补充笔记。",
                            entry["url"], ["前沿", feed["name"]],
                        )
                        parsed = parse_markdown(raw_path, Path(vault).expanduser().resolve())
                        note_id, changed = store.upsert_note(parsed)
                        store.replace_wikilinks(note_id, parsed["wikilinks"])
                    except OSError:
                        note_id, changed = store.upsert_note(note_payload)
                else:
                    note_id, changed = store.upsert_note(note_payload)
                added += int(changed)
                if changed:
                    store.add_task(
                        f"前沿待嫁接：{entry['title']}", "frontier", entry["title"],
                        {"note_id": note_id, "url": entry["url"], "summary": entry["summary"][:1200]},
                        datetime.now().astimezone().isoformat(timespec="seconds"), 10,
                    )
            store.touch_feed(feed["id"])
        except Exception as exc:
            errors.append(f"{feed['name']}：{exc}")
    store.add_activity("refresh_feeds", f"发现 {added} 篇新内容", min(added * 2, 20))
    return {"fetched": fetched, "added": added, "errors": errors}
