from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.storage import GardenStore


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-\u4e00-\u9fff]+)")
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _scalar(value: str) -> Any:
    value = value.strip().strip('"\'')
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [part.strip().strip('"\'') for part in value[1:-1].split(",") if part.strip()]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    current_list: str | None = None
    for raw in match.group(1).splitlines():
        if current_list and raw.lstrip().startswith("- "):
            meta.setdefault(current_list, []).append(_scalar(raw.lstrip()[2:]))
            continue
        current_list = None
        if ":" not in raw or raw.lstrip().startswith("#"):
            continue
        key, value = raw.split(":", 1)
        key, value = key.strip(), value.strip()
        if value:
            meta[key] = _scalar(value)
        else:
            meta[key] = []
            current_list = key
    return meta, text[match.end():]


def infer_kind(path: Path, metadata: dict[str, Any], tags: list[str]) -> str:
    explicit = str(metadata.get("garden_type") or metadata.get("type") or "").lower()
    mapping = {
        "textbook": "textbook", "教材": "textbook", "course": "course", "课程": "course",
        "frontier": "frontier", "前沿": "frontier", "interest": "interest", "兴趣": "interest",
        "card": "card", "对照卡": "bridge", "降维对照": "bridge",
        "概念底座": "concept", "交叉火花": "spark", "主题索引": "moc",
        "sources": "source", "raw": "raw",
    }
    # The AGENTS.md directory contract is authoritative; a concept page may still
    # carry a “课程” tag without becoming a course note.
    wiki_sections = {
        "01-概念底座": "concept", "02-降维对照": "bridge", "03-交叉火花": "spark",
        "04-主题索引": "moc", "sources": "source", "raw": "raw",
    }
    for part in path.parts:
        if part in wiki_sections:
            return wiki_sections[part]
    for candidate in [explicit, *tags, *[part.lower() for part in path.parts]]:
        for key, value in mapping.items():
            if key in candidate:
                return value
    return "interest"


def parse_markdown(path: Path, vault: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    metadata, body = parse_frontmatter(text)
    heading = HEADING_RE.search(body)
    title = str(metadata.get("title") or (heading.group(1).strip() if heading else path.stem))
    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [part.strip() for part in raw_tags.replace(",", " ").split()]
    tags = sorted({str(tag).lstrip("#") for tag in raw_tags} | set(TAG_RE.findall(body)))
    return {
        "path": str(path.relative_to(vault)).replace("\\", "/"),
        "title": title,
        "kind": infer_kind(path.relative_to(vault), metadata, tags),
        "content": body.strip(),
        "tags": tags,
        "source": "obsidian",
        "source_url": str(metadata.get("source") or metadata.get("url") or ""),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "wikilinks": WIKILINK_RE.findall(body),
    }


def sync_vault(vault_path: str | Path, store: GardenStore) -> dict[str, Any]:
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("Obsidian Vault 路径不存在或不是文件夹")
    scanned = changed = failed = 0
    seen_paths: set[str] = set()
    for path in vault.rglob("*.md"):
        if ".obsidian" in path.parts or any(part.startswith(".") for part in path.relative_to(vault).parts):
            continue
        scanned += 1
        try:
            note = parse_markdown(path, vault)
            seen_paths.add(note["path"])
            note_id, is_changed = store.upsert_note(note)
            store.replace_wikilinks(note_id, note.pop("wikilinks"))
            changed += int(is_changed)
        except (OSError, UnicodeError):
            failed += 1
    store.resolve_links()
    removed = store.prune_obsidian_paths(seen_paths)
    store.set_setting("vault_path", str(vault))
    if changed or removed:
        store.add_activity("sync_vault", f"扫描 {scanned} 篇，更新 {changed} 篇，移除 {removed} 个旧索引", min(changed, 20))
        # Run once per changed concept/MOC signature. The taxonomy module caches
        # its signature, so ordinary background sync does not repeatedly call the model.
        from core.taxonomy import classify_unmounted_concepts, rebuild_concept_hierarchy
        classification = classify_unmounted_concepts(store)
        hierarchy = rebuild_concept_hierarchy(store)
    else:
        classification = {
            "examined": 0,
            "classified": 0,
            "needs_review": len(store.setting("classification_queue_v1", {}) or {}),
            "items": list((store.setting("classification_queue_v1", {}) or {}).values()),
        }
        hierarchy = {"changed": False, "relations": 0, "method": "unchanged", "topics": 0}
    return {
        "scanned": scanned, "changed": changed, "removed": removed, "failed": failed,
        "classification": classification, "hierarchy": hierarchy,
    }


def export_markdown(vault_path: str | Path, folder: str, title: str, body: str, tags: list[str]) -> Path:
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("请先配置有效的 Obsidian Vault")
    safe_title = re.sub(r'[<>:"/\\|?*]', "-", title).strip(". ")[:100] or "知识花园卡片"
    output_dir = vault / "知识花园" / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_title}.md"
    frontmatter = "---\ngarden_generated: true\ntags:\n" + "".join(f"  - {tag}\n" for tag in tags) + "---\n\n"
    path.write_text(frontmatter + f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return path


def write_wiki_asset(vault_path: str | Path, section: str, title: str, body: str, tags: list[str]) -> Path:
    """Write a compiled asset into the AGENTS.md wiki layout used by llm-wiki-lab."""
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("请先配置有效的 Obsidian Vault")
    allowed = {"01-概念底座", "02-降维对照", "03-交叉火花", "04-主题索引", "sources"}
    if section not in allowed:
        raise ValueError("未知 Wiki 分区")
    safe_title = re.sub(r'[<>:"/\\|?*]', "-", title).strip(". ")[:100] or "知识资产"
    output_dir = vault / "wiki" / section
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_title}.md"
    frontmatter = "---\ngarden_generated: true\ntags:\n" + "".join(f"  - {tag}\n" for tag in tags) + "---\n\n"
    path.write_text(frontmatter + f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return path


def append_backlink(path: Path, target_title: str) -> bool:
    """Add a visible reciprocal link without rewriting the authored body."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    link = f"[[{target_title}]]"
    if link in text:
        return False
    if "## 相关链接" in text:
        text = text.replace("## 相关链接", f"## 相关链接\n\n- {link}", 1)
    else:
        text = text.rstrip() + f"\n\n## 相关链接\n\n- {link}\n"
    path.write_text(text, encoding="utf-8")
    return True


def write_raw_material(
    vault_path: str | Path, title: str, content: str, source_url: str = "", tags: list[str] | None = None,
    garden_type: str = "frontier",
) -> Path:
    """Persist material received by the web UI into the Vault's raw inbox."""
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("请先配置有效的 Obsidian Vault")
    safe_title = re.sub(r'[<>:"/\\|?*]', "-", title).strip(". ")[:80] or "网页摘录"
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    raw_dir = vault / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{date_prefix}-{safe_title}.md"
    frontmatter = (
        "---\n"
        f"title: \"{title.replace(chr(34), chr(39))}\"\n"
        f"garden_type: {garden_type}\n"
        f"source: \"{source_url.replace(chr(34), chr(39))}\"\n"
        "tags:\n" + "".join(f"  - {tag}\n" for tag in (tags or ["前沿", "网页输入"])) +
        "---\n\n"
    )
    rendered = frontmatter + f"# {title}\n\n{content.strip()}\n"
    if not path.exists() or path.read_text(encoding="utf-8-sig", errors="replace") != rendered:
        path.write_text(rendered, encoding="utf-8")
    return path
