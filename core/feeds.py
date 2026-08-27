from __future__ import annotations

import hashlib
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
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
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(value)))).strip()


_BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_JS_SCALAR = re.compile(
    r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|true|false|null|void\s+0',
    re.IGNORECASE,
)
_VIDEO_FIELDS = re.compile(
    r'(?:^|,)(bvid|title|description|arcurl|pubdate):("(?:\\.|[^"\\])*"|[^,}\]]+)'
)


def describe_feed(url: str) -> dict[str, Any]:
    clean = str(url or "").strip()
    parsed = urllib.parse.urlsplit(clean)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("请输入完整的 http 或 https 博主主页、RSS 或 Atom 链接")
    if parsed.username or parsed.password:
        raise ValueError("订阅链接不能包含账号或密码")
    host = parsed.hostname.lower().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    uid = ""
    if host == "space.bilibili.com" and parts and parts[0].isdigit():
        uid = parts[0]
    elif host in {"m.bilibili.com", "bilibili.com"} and len(parts) >= 2 and parts[0] == "space" and parts[1].isdigit():
        uid = parts[1]
    if uid:
        return {
            "platform": "bilibili", "platform_label": "B站 UP 主",
            "uid": uid, "url": f"https://space.bilibili.com/{uid}",
        }
    if host in {"youtube.com", "m.youtube.com"}:
        if len(parts) >= 2 and parts[0] == "channel":
            channel = parts[1]
            return {
                "platform": "youtube", "platform_label": "YouTube 频道",
                "url": clean,
                "feed_url": "https://www.youtube.com/feeds/videos.xml?"
                + urllib.parse.urlencode({"channel_id": channel}),
            }
        if parts[:2] == ["feeds", "videos.xml"]:
            return {"platform": "youtube", "platform_label": "YouTube 频道", "url": clean, "feed_url": clean}
        if parts and parts[0].startswith("@"):
            return {"platform": "youtube", "platform_label": "YouTube 创作者", "url": clean}
    if host == "github.com" and parts:
        if len(parts) == 1:
            return {
                "platform": "github", "platform_label": "GitHub 创作者",
                "url": clean, "feed_url": f"https://github.com/{parts[0]}.atom",
            }
        if len(parts) >= 2 and not parts[1].endswith(".atom"):
            owner, repository = parts[:2]
            return {
                "platform": "github", "platform_label": "GitHub 仓库",
                "url": clean, "feed_url": f"https://github.com/{owner}/{repository}/commits/HEAD.atom",
            }
    if host.endswith(".substack.com"):
        return {
            "platform": "substack", "platform_label": "Substack 博客",
            "url": clean, "feed_url": f"{parsed.scheme}://{parsed.netloc}/feed",
        }
    if host == "medium.com" and parts and parts[0].startswith("@"):
        return {
            "platform": "medium", "platform_label": "Medium 博主",
            "url": clean, "feed_url": f"https://medium.com/feed/{parts[0]}",
        }
    return {"platform": "rss", "platform_label": "RSS / Atom", "url": clean, "feed_url": clean}


def list_followed_sources(store: GardenStore) -> list[dict[str, Any]]:
    rows = []
    for item in store.list_feeds():
        try:
            descriptor = describe_feed(str(item.get("url") or ""))
        except ValueError:
            descriptor = {"platform": "unknown", "platform_label": "链接待检查"}
        rows.append({**item, **{key: descriptor[key] for key in ("platform", "platform_label")}})
    return rows

def list_frontier_material(store: GardenStore, limit: int = 200) -> list[dict[str, Any]]:
    """Include followed-source raw inbox items without changing vault taxonomy."""
    direct = store.list_notes(kind="frontier", limit=limit)
    inbox = [
        note for note in store.list_notes(kind="raw", limit=limit)
        if "前沿" in (note.get("tags") or []) and note.get("source_url")
    ]
    candidates = sorted(
        [*direct, *inbox],
        key=lambda item: str(item.get("updated_at") or ""),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for note in candidates:
        key = str(note.get("source_url") or note.get("path") or "")
        if key not in seen:
            result.append(note)
            seen.add(key)
    return result[:limit]


def _fetch_bytes(url: str, *, referer: str = "") -> bytes:
    headers = {
        "User-Agent": _BROWSER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    cookie = os.getenv("GARDEN_BILIBILI_COOKIE", "").strip()
    if cookie and urllib.parse.urlsplit(url).hostname.lower().endswith("bilibili.com"):
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read(4_000_000)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 412, 429} and "bilibili.com" in url:
            raise ValueError(
                f"B站暂时触发游客访问限制（HTTP {exc.code}）；请稍后重试"
                "，或使用自己的浏览器登录态配置 GARDEN_BILIBILI_COOKIE"
            ) from exc
        raise


def _javascript_value(token: str, values: dict[str, Any]) -> Any:
    clean = token.strip()
    if clean in values:
        return values[clean]
    if clean.startswith("void"):
        return None
    try:
        return json.loads(clean)
    except (TypeError, ValueError):
        return clean


def _extract_bilibili_videos(page: str, uid: str, limit: int = 8) -> list[dict[str, Any]]:
    script_match = re.search(
        r"<script[^>]*>\s*window\.__pinia\s*=\s*(.*?)</script>", page, re.IGNORECASE | re.DOTALL,
    )
    if not script_match:
        raise ValueError("B站搜索页没有返回可读取的公开视频状态")
    state = script_match.group(1).strip()
    encoded = re.match(r"\(function\(([^)]*)\)\{(.*)\}\((.*)\)\);?\s*$", state, re.DOTALL)
    if not encoded:
        raise ValueError("B站公开视频状态格式已变化")
    parameters = [item.strip() for item in encoded.group(1).split(",") if item.strip()]
    raw_arguments = encoded.group(3)
    tokens = [match.group(0) for match in _JS_SCALAR.finditer(raw_arguments)]
    remainder = _JS_SCALAR.sub("", raw_arguments).replace(",", "").strip()
    if remainder or len(tokens) != len(parameters):
        raise ValueError("B站公开视频状态包含无法安全解析的数据")
    resolved = {
        symbol: _javascript_value(token, {})
        for symbol, token in zip(parameters, tokens)
    }
    uid_symbols = [
        symbol for symbol, value in resolved.items()
        if str(value) == str(uid)
    ]
    if not uid_symbols:
        raise ValueError(f"B站搜索结果中没有找到 UID {uid} 的公开视频")
    body = encoded.group(2)
    matcher = re.compile(
        r"\{type:[^{}]{0,260},author:(?P<author>[^,}]+),mid:"
        + r"(?:" + "|".join(re.escape(symbol) for symbol in uid_symbols) + r")"
        + r",(?P<record>[^{}]+)",
    )
    entries_by_url: dict[str, dict[str, Any]] = {}
    for match in matcher.finditer(body):
        fields = {
            field: _javascript_value(value, resolved)
            for field, value in _VIDEO_FIELDS.findall(match.group("record"))
        }
        bvid = str(fields.get("bvid") or "").strip()
        title = _strip_html(str(fields.get("title") or ""))
        if not bvid.startswith("BV") or not title:
            continue
        timestamp = fields.get("pubdate")
        published = ""
        if isinstance(timestamp, (int, float)) and timestamp > 0:
            published = datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")
        url = f"https://www.bilibili.com/video/{bvid}"
        entries_by_url[url] = {
            "title": title,
            "url": url,
            "summary": _strip_html(str(fields.get("description") or "")),
            "published": published,
        }
    entries = sorted(
        entries_by_url.values(), key=lambda item: item["published"], reverse=True,
    )
    if not entries:
        raise ValueError("已找到 B 站 UP 主，但没有发现可公开读取的视频")
    return entries[:limit]


def _fetch_bilibili_feed(source: dict[str, Any], *, name: str, limit: int) -> list[dict[str, Any]]:
    uid = str(source["uid"])
    creator = str(name or "").strip()
    if not creator:
        card_url = "https://api.bilibili.com/x/web-interface/card?" + urllib.parse.urlencode({"mid": uid})
        profile = json.loads(_fetch_bytes(card_url, referer=str(source["url"])))
        creator = str((profile.get("data") or {}).get("card", {}).get("name") or "").strip()
    if not creator:
        raise ValueError("无法识别 B 站 UP 主名称，请在博主名称栏填写其昵称")
    search_url = "https://search.bilibili.com/all?" + urllib.parse.urlencode({
        "keyword": creator, "order": "pubdate",
    })
    page = _fetch_bytes(search_url, referer="https://www.bilibili.com/").decode("utf-8", errors="replace")
    return _extract_bilibili_videos(page, uid, limit)


class _FeedDiscoveryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.feed_url = ""

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "link" or self.feed_url:
            return
        attrs = {key.lower(): str(value or "") for key, value in attributes}
        if "alternate" in attrs.get("rel", "").lower() and any(
            item in attrs.get("type", "").lower() for item in ("rss", "atom", "xml")
        ):
            self.feed_url = attrs.get("href", "")


def fetch_feed(url: str, limit: int = 8) -> list[dict[str, Any]]:
    return fetch_source(url, limit=limit)


def fetch_source(url: str, *, limit: int = 8, name: str = "") -> list[dict[str, Any]]:
    source = describe_feed(url)
    if source["platform"] == "bilibili":
        return _fetch_bilibili_feed(source, name=name, limit=limit)
    resolved_url = str(source.get("feed_url") or source["url"])
    raw = _fetch_bytes(resolved_url)
    if source["platform"] == "youtube" and not source.get("feed_url"):
        page = raw.decode("utf-8", errors="replace")
        match = re.search(r'"(?:channelId|externalId)"\s*:\s*"(UC[\w-]+)"', page)
        if not match:
            raise ValueError("YouTube 主页未公开频道 ID；请使用 /channel/UC... 地址")
        resolved_url = "https://www.youtube.com/feeds/videos.xml?" + urllib.parse.urlencode({
            "channel_id": match.group(1),
        })
        raw = _fetch_bytes(resolved_url)
    if raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        parser = _FeedDiscoveryParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        if not parser.feed_url:
            raise ValueError("该博主主页没有公开 RSS/Atom；目前可直接识别 B 站、YouTube、GitHub、Substack 和 Medium")
        resolved_url = urllib.parse.urljoin(resolved_url, parser.feed_url)
        describe_feed(resolved_url)
        raw = _fetch_bytes(resolved_url)
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
        channel = root.find("channel")
        if channel is None:
            channel = root
        for item in channel.findall("item")[:limit]:
            entries.append({
                "title": _text(item, ["title"]),
                "url": _text(item, ["link", "guid"]),
                "summary": _strip_html(_text(item, ["description", "content:encoded"])),
                "published": _text(item, ["pubDate", "date"]),
            })
    return [entry for entry in entries if entry["title"]]


def refresh_feeds(store: GardenStore) -> dict[str, Any]:
    fetched = added = 0
    errors: list[str] = []
    sources: list[dict[str, Any]] = []
    for feed in store.list_feeds():
        if not feed["enabled"]:
            continue
        source_added = 0
        try:
            descriptor = describe_feed(feed["url"])
            entries = fetch_source(feed["url"], name=feed["name"])
            parsed_videos = 0
            if descriptor["platform"] == "bilibili":
                # Keep the hourly patrol light: automatically inspect only the
                # two newest videos. Older items remain available for explicit
                # subtitle/ASR reading from the frontier page.
                from core.bilibili_mcp import inspect_public_video

                for entry in entries[:2]:
                    try:
                        inspection = inspect_public_video(str(entry.get("url") or ""))
                    except Exception as exc:
                        inspection = {
                            "status": "unavailable",
                            "message": f"公开字幕检查失败：{exc.__class__.__name__}",
                        }
                    entry["content_status"] = str(inspection.get("status") or "unavailable")
                    entry["content_status_message"] = str(inspection.get("message") or "")
                    if inspection.get("status") == "ready" and inspection.get("transcript"):
                        entry["transcript"] = str(inspection["transcript"])
                        entry["data_source"] = str(inspection.get("data_source") or "public_subtitle")
                        parsed_videos += 1
            fetched += len(entries)
            for entry in entries:
                key = entry["url"] or entry["title"]
                status = str(entry.get("content_status") or "")
                status_message = str(entry.get("content_status_message") or "")
                transcript = str(entry.get("transcript") or "").strip()
                content_parts = [entry["summary"] or f"来自 {feed['name']} 的新内容，点击来源阅读全文。"]
                tags = ["前沿", feed["name"]]
                if descriptor["platform"] == "bilibili":
                    tags.append("B站")
                    if transcript:
                        tags.extend(["视频解析", "字幕已解析"])
                        content_parts.extend([
                            f"> 内容解析状态：{status_message or '已读取公开字幕'}；关键表述请回到时间戳核对。",
                            "## 带时间戳的公开字幕\n" + transcript,
                        ])
                    else:
                        tags.append("仅元数据")
                        content_parts.append(
                            f"> 内容解析状态：{status_message or '当前仅取得标题和简介；系统没有假装读取视频内容。'}"
                        )
                content = "\n\n".join(part for part in content_parts if part)
                digest = hashlib.sha256((key + content).encode("utf-8")).hexdigest()
                note_payload = {
                    "path": f"feed::{feed['id']}::{hashlib.sha1(key.encode('utf-8')).hexdigest()}",
                    "title": entry["title"],
                    "kind": "frontier",
                    "content": content,
                    "tags": tags,
                    "source": feed["name"],
                    "source_url": entry["url"],
                    "content_hash": digest,
                }
                vault = store.setting("vault_path", "")
                if vault:
                    try:
                        raw_path = write_raw_material(
                            vault, entry["title"], content,
                            entry["url"], tags,
                        )
                        parsed = parse_markdown(raw_path, Path(vault).expanduser().resolve())
                        note_id, changed = store.upsert_note(parsed)
                        store.replace_wikilinks(note_id, parsed["wikilinks"])
                    except OSError:
                        note_id, changed = store.upsert_note(note_payload)
                else:
                    note_id, changed = store.upsert_note(note_payload)
                added += int(changed)
                source_added += int(changed)
                if changed:
                    store.add_task(
                        f"前沿待嫁接：{entry['title']}", "frontier", entry["title"],
                        {
                            "note_id": note_id, "url": entry["url"], "summary": content[:6000],
                            "content_status": status or ("ready" if transcript else "metadata_only"),
                        },
                        datetime.now().astimezone().isoformat(timespec="seconds"), 10,
                    )
            store.touch_feed(feed["id"])
            sources.append({
                "id": feed["id"], "name": feed["name"], "platform": descriptor["platform"],
                "platform_label": descriptor["platform_label"], "status": "ok",
                "fetched": len(entries), "added": source_added, "parsed_videos": parsed_videos,
            })
        except Exception as exc:
            message = f"{feed['name']}：{exc}"
            errors.append(message)
            sources.append({
                "id": feed["id"], "name": feed["name"], "status": "error", "error": str(exc),
            })
    store.add_activity("refresh_feeds", f"发现 {added} 篇新内容", min(added * 2, 20))
    return {"fetched": fetched, "added": added, "errors": errors, "sources": sources}
