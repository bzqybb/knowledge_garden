from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socket
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.agent import answer_from_wiki, briefing, daily_digest, hint_for_task, patrol_vault, save_agent_insight, update_agents_manifest
from core.bilibili_mcp import read_video as read_bilibili_video, runtime_status as bilibili_mcp_status
from core.config import DATA_DIR, WEB_DIR, llm_config
from core.compiler import ingest_raw, validate_links
from core.engine import add_interest, analyze_frontier, article_preview_metadata, evaluate_review, weekly_report
from core.inspiration import explore_inspiration, save_inspiration_seed
from core.feeds import describe_feed, list_followed_sources, list_frontier_material, refresh_feeds
from core.llm import LLMError, chat, prewarm_understanding_model
from core.learning_memory import LearningMemoryService
from core.mindmap import build_mindmap
from core.obsidian import sync_vault, write_raw_material
from core.paper_reader import deep_read_paper
from core.pdf_ocr import start_background_textbook_ocr, textbook_ocr_status, windows_ocr_available
from core.retrieval import build_semantic_links, ingest_pdf_directory, rebuild_domain_map, search_notes
from core.storage import GardenStore
from core.taxonomy import classify_unmounted_concepts, rebuild_concept_hierarchy
from core.tracememo import (
    TraceMemoClient,
    TraceMemoError,
    configure_tracememo,
    connection_summary,
    forget_tracememo_token,
    tracememo_config,
)
from core.web_research import fetch_wechat_article_text


STORE = GardenStore()
MEMORY = LearningMemoryService(STORE)
LLM_HEALTH_LOCK = threading.Lock()
LLM_HEALTH = {
    "llm_configured": llm_config().enabled,
    "llm_enabled": False,
    "llm_status": "checking" if llm_config().enabled else "offline",
    "llm_message": "正在验证理解 API……" if llm_config().enabled else "尚未配置理解 API。",
}


def refresh_llm_health() -> dict:
    config = llm_config()
    if not config.enabled:
        result = {
            "llm_configured": False,
            "llm_enabled": False,
            "llm_status": "offline",
            "llm_message": "尚未配置理解 API。",
        }
    else:
        try:
            chat("You are an API connection check. Reply with exactly OK.", "connection check", temperature=0)
            result = {
                "llm_configured": True,
                "llm_enabled": True,
                "llm_status": "connected",
                "llm_message": f"LangChain 已连接 {config.model}。",
            }
        except LLMError as exc:
            detail = str(exc).lower()
            if "401" in detail or "authentication" in detail or "api key" in detail:
                message = "API Key 验证失败，请用正确的服务商密钥重新配置。"
                status = "invalid_key"
            elif "429" in detail or "rate limit" in detail or "insufficient" in detail:
                message = "API 额度不足或请求受限，请检查服务商账户。"
                status = "limited"
            else:
                message = "理解 API 暂时无法连接，请检查网络、接口地址与模型名称。"
                status = "error"
            result = {
                "llm_configured": True,
                "llm_enabled": False,
                "llm_status": status,
                "llm_message": message,
            }
    with LLM_HEALTH_LOCK:
        LLM_HEALTH.update(result)
        return dict(LLM_HEALTH)


def llm_health() -> dict:
    with LLM_HEALTH_LOCK:
        return dict(LLM_HEALTH)


def wechat_connection_status() -> dict:
    base_url = STORE.setting("tracememo_base_url", "http://127.0.0.1:6131")
    summary = {
        "base_url": str(base_url), "token_configured": False, "token_saved": False,
        "service_online": False, "authorized": False,
    }
    try:
        summary.update(connection_summary(base_url))
        client = TraceMemoClient(tracememo_config(base_url), timeout=1.5)
        health = client.health()
        summary.update({"service_online": True, "health": health})
        if summary["token_configured"]:
            try:
                client.current_time()
                summary.update({"authorized": True, "message": "TraceMemo 已连接，可以按需读取微信历史。"})
            except TraceMemoError as exc:
                summary.update({"authorized": False, "message": str(exc)})
        else:
            summary.update({"authorized": False, "message": "TraceMemo 服务在线；请配置 API Center Token。"})
    except Exception as exc:
        summary.update({"service_online": False, "authorized": False, "message": str(exc)})
    return summary


def sync_configured_vault() -> dict | None:
    vault = STORE.setting("vault_path", "")
    return sync_vault(vault, STORE) if vault else None


def feed_patrol(stop_event: threading.Event, interval_minutes: int) -> None:
    """Proactively refresh followed sources while the local garden is awake."""
    while not stop_event.wait(max(5, interval_minutes) * 60):
        if not STORE.list_feeds():
            continue
        try:
            refresh_feeds(STORE)
        except Exception as exc:
            print(f"[Knowledge Garden] 关注源巡视失败：{exc}")


def vault_patrol(stop_event: threading.Event, interval_seconds: int = 8) -> None:
    """Watch Markdown metadata and keep the garden index aligned with Obsidian."""
    last_signature = None
    last_vault = ""
    while not stop_event.wait(max(3, interval_seconds)):
        vault_value = STORE.setting("vault_path", "")
        if not vault_value:
            last_signature = None
            last_vault = ""
            continue
        try:
            vault = Path(vault_value).expanduser().resolve()
            signature = tuple(sorted(
                (str(path.relative_to(vault)).replace("\\", "/"), path.stat().st_mtime_ns, path.stat().st_size)
                for path in vault.rglob("*.md") if ".obsidian" not in path.parts
            ))
            if str(vault) != last_vault or signature != last_signature:
                sync_vault(vault, STORE)
                patrol = patrol_vault(vault, STORE)
                if patrol["ingested"] or patrol["manifest"]["changed"]:
                    sync_vault(vault, STORE)
                last_signature = signature
                last_vault = str(vault)
        except Exception as exc:
            print(f"[Knowledge Garden] Obsidian 自动同步失败：{exc}")


def memory_patrol(stop_event: threading.Event, interval_minutes: int = 30) -> None:
    """Consolidate evidence and refresh decay projections without calling an LLM."""
    while not stop_event.wait(max(5, interval_minutes) * 60):
        try:
            MEMORY.reflect(force=False)
            MEMORY.refresh_knowledge_weights()
        except Exception as exc:
            print(f"[Knowledge Garden] 经验反思巡检失败：{exc}")


def understanding_warmup() -> None:
    """Prime GLM/HTTPS in the background so the first user turn stays fast."""
    try:
        ok, message = prewarm_understanding_model()
        colorless_status = "就绪" if ok else "未就绪（提问时会使用安全兜底）"
        print(f"[Knowledge Garden] 问题理解器{colorless_status}：{message}")
    except Exception as exc:
        # Warm-up is an optimization, never a reason to terminate the garden.
        print(f"[Knowledge Garden] 问题理解器预热已跳过：{exc.__class__.__name__}")


def model_warmups(stop_event: threading.Event) -> None:
    """Warm remote model clients sequentially after the UI can already load."""
    if stop_event.wait(3):
        return
    refresh_llm_health()
    if not stop_event.is_set():
        understanding_warmup()


def retrieval_warmup(stop_event: threading.Event) -> None:
    """Load optional local retrieval models after the HTTP server is available."""
    if os.getenv("GARDEN_PREWARM_RETRIEVAL", "1").strip().lower() in {"0", "false", "no"}:
        return
    # Importing PyTorch/SentenceTransformers is CPU-heavy on Windows.  Give the
    # browser enough time to fetch the shell and initial API state first.
    if stop_event.wait(8):
        return
    try:
        if os.getenv("GARDEN_DISABLE_SEMANTIC", "").strip().lower() not in {"1", "true", "yes"}:
            from core.semantic_index import _load as load_semantic_index

            if load_semantic_index() is None:
                print("[Knowledge Garden] 本地语义索引尚未建立，已跳过检索预热。")
                return
        if stop_event.is_set():
            return
        if os.getenv("GARDEN_DISABLE_RERANKER", "").strip().lower() not in {"1", "true", "yes"}:
            from core.reranker import _load_model as load_reranker

            load_reranker()
        print("[Knowledge Garden] 语义检索与精排模型已在后台预热。")
    except Exception as exc:
        # Retrieval remains optional on machines without downloaded models.
        print(f"[Knowledge Garden] 本地检索预热已跳过：{exc.__class__.__name__}")


class GardenHandler(BaseHTTPRequestHandler):
    server_version = "KnowledgeGarden/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[知识花园] {self.address_string()} - {fmt % args}")

    def _json(self, data, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The browser may leave a long-running import/rebuild request before
            # the worker finishes.  The work is still valid; there is simply no
            # response socket left to write to.
            return

    def _body(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("请求内容不是有效 JSON") from exc

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (WEB_DIR / relative).resolve()
        if WEB_DIR.resolve() not in candidate.parents and candidate != WEB_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"
        payload = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                model_health = llm_health()
                self._json({
                    "stats": STORE.stats(),
                    "settings": {
                        "vault_path": STORE.setting("vault_path", ""),
                        "learning_level": STORE.setting("learning_level", "本科入门"),
                        "interests": STORE.setting("interests", []),
                        "frontier_focus": STORE.setting("frontier_focus", ""),
                        "textbook_directory": STORE.setting("textbook_directory", str(DATA_DIR / "textbook_kb")),
                        "textbook_ocr": textbook_ocr_status(STORE),
                        "classification_queue": list((STORE.setting("classification_queue_v1", {}) or {}).values()),
                        "wechat": wechat_connection_status(),
                        **model_health,
                    },
                    "tasks": STORE.list_tasks(limit=8),
                    "cards": STORE.list_cards(limit=5),
                    "feeds": list_followed_sources(STORE),
                    "report": weekly_report(STORE),
                    "agent": briefing(STORE),
                })
            elif parsed.path == "/api/graph":
                self._json(STORE.graph())
            elif parsed.path == "/api/mindmap":
                self._json(build_mindmap(STORE))
            elif parsed.path == "/api/notes":
                params = parse_qs(parsed.query)
                kind = params.get("kind", [None])[0]
                notes = list_frontier_material(STORE, 200) if kind == "frontier" else STORE.list_notes(kind, 200)
                self._json({"notes": notes})
            elif parsed.path == "/api/textbooks/ocr/status":
                self._json({"ok": True, "status": textbook_ocr_status(STORE)})
            elif parsed.path == "/api/cards":
                self._json({"cards": STORE.list_cards(50)})
            elif parsed.path == "/api/tasks":
                self._json({"tasks": STORE.list_tasks(include_done=True)})
            elif parsed.path == "/api/report":
                self._json(weekly_report(STORE))
            elif parsed.path == "/api/daily":
                params = parse_qs(parsed.query)
                self._json(daily_digest(STORE, force=params.get("refresh", ["0"])[0] == "1"))
            elif parsed.path == "/api/bilibili/status":
                self._json({"ok": True, "status": bilibili_mcp_status()})
            elif parsed.path == "/api/memory":
                self._json(MEMORY.overview())
            elif parsed.path == "/api/wechat/status":
                self._json({"ok": True, "status": wechat_connection_status()})
            elif parsed.path == "/api/wechat/recent":
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", [20])[0])
                base_url = STORE.setting("tracememo_base_url", "http://127.0.0.1:6131")
                self._json({"ok": True, "result": TraceMemoClient(tracememo_config(base_url)).recent_chats(limit)})
            elif parsed.path == "/api/wechat/official-accounts":
                params = parse_qs(parsed.query)
                filter_text = str(params.get("filter", [""])[0])
                base_url = STORE.setting("tracememo_base_url", "http://127.0.0.1:6131")
                client = TraceMemoClient(tracememo_config(base_url))
                self._json({"ok": True, "result": client.official_accounts(filter_text)})
            elif parsed.path == "/api/wechat/candidates":
                self._json({"ok": True, "candidates": STORE.list_wechat_candidates()})
            elif parsed.path == "/api/note":
                params = parse_qs(parsed.query)
                note_id = int(params.get("id", [0])[0])
                note = STORE.get_note(note_id)
                if not note:
                    raise ValueError("知识节点不存在")
                self._json({"note": note})
            elif parsed.path.startswith("/api/"):
                self._json({"error": "接口不存在"}, 404)
            else:
                self._serve_static(parsed.path)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/settings":
                for key in ("vault_path", "learning_level", "interests", "frontier_focus", "textbook_directory"):
                    if key in body:
                        STORE.set_setting(key, body[key])
                self._json({"ok": True})
            elif path == "/api/wechat/config":
                base_url = str(body.get("base_url", "http://127.0.0.1:6131"))
                status = configure_tracememo(
                    base_url=base_url,
                    token=str(body.get("token", "")),
                    save_token=bool(body.get("save_token", True)),
                )
                STORE.set_setting("tracememo_base_url", status["base_url"])
                self._json({"ok": True, "status": wechat_connection_status()})
            elif path == "/api/wechat/forget":
                forget_tracememo_token()
                self._json({"ok": True, "status": wechat_connection_status()})
            elif path == "/api/wechat/preview":
                talker = str(body.get("talker", "")).strip()
                time_range = str(body.get("time", "")).strip()
                base_url = STORE.setting("tracememo_base_url", "http://127.0.0.1:6131")
                client = TraceMemoClient(tracememo_config(base_url))
                clock = client.current_time()
                contact = client.resolve(talker)
                result = client.chatlog(
                    talker,
                    time_range=time_range,
                    start_time=str(body.get("start_time", "")),
                    end_time=str(body.get("end_time", "")),
                )
                self._json({"ok": True, "preview": {
                    "clock": clock, "contact": contact, "query": result.get("query", {}),
                    "count": result.get("count", len(result.get("messages", []))),
                    "messages": result.get("messages", []), "truncated": result.get("truncated", False),
                    "talker": talker, "time": time_range,
                }})
            elif path == "/api/wechat/articles":
                talker = str(body.get("talker", "")).strip()
                if not talker:
                    raise ValueError("请先选择一个公众号")
                days = int(body.get("days", 30))
                base_url = STORE.setting("tracememo_base_url", "http://127.0.0.1:6131")
                client = TraceMemoClient(tracememo_config(base_url))
                contact = client.resolve(talker)
                result = client.official_articles(talker, days=days, contact=contact)
                result["contact"] = contact
                self._json({"ok": True, "result": result})
            elif path == "/api/wechat/article/read":
                url = str(body.get("url", "")).strip()
                text = fetch_wechat_article_text(url)
                self._json({"ok": True, "url": url, "text": text, "access_scope": "open_fulltext"})
            elif path == "/api/wechat/article/preview":
                url = str(body.get("url", "")).strip()
                title = str(body.get("title", "")).strip()
                description = str(body.get("description", "")).strip()
                text = fetch_wechat_article_text(url)
                self._json({
                    "ok": True,
                    "url": url,
                    "preview": article_preview_metadata(title, text, description),
                })
            elif path == "/api/wechat/candidates":
                messages = body.get("messages") if isinstance(body.get("messages"), list) else []
                candidate = STORE.create_wechat_candidate(
                    title=str(body.get("title", "微信讨论候选")),
                    talker=str(body.get("talker", "")),
                    time_range=str(body.get("time", "")),
                    contact=body.get("contact") if isinstance(body.get("contact"), dict) else {},
                    query=body.get("query") if isinstance(body.get("query"), dict) else {},
                    messages=messages,
                )
                self._json({"ok": True, "candidate": candidate})
            elif path == "/api/wechat/candidates/review":
                candidate_id = str(body.get("candidate_id", ""))
                accepted = bool(body.get("accepted"))
                raw_path = ""
                ingest = None
                if accepted:
                    candidate = STORE.get_wechat_candidate(candidate_id)
                    if not candidate:
                        raise ValueError("微信候选不存在")
                    vault = STORE.setting("vault_path", "")
                    if not vault:
                        raise ValueError("确认沉淀前请先连接 Obsidian Vault")
                    query = candidate.get("query") if isinstance(candidate.get("query"), dict) else {}
                    source_kind = str(query.get("source") or "chat")
                    article = query.get("article") if isinstance(query.get("article"), dict) else {}
                    article_url = str(article.get("url") or "").strip()
                    if source_kind == "official_account":
                        if not article_url:
                            raise ValueError("公众号候选缺少原文链接，尚不能可靠归类")
                        full_text = fetch_wechat_article_text(article_url)
                        if len(re.sub(r"\s+", "", full_text)) < 180:
                            raise ValueError("公众号正文过短或被验证页拦截，暂不写入知识图谱")
                        lines = [
                            "> 来源：由用户授权，通过 TraceMemo 定位公众号文章，并回到原网址读取正文。",
                            f"> 公众号：{article.get('account_name') or article.get('publisher') or candidate.get('talker', '')}",
                            f"> 原文：{article_url}",
                            f"> 发布时间：{article.get('sent_at') or candidate.get('time_range') or '未知'}",
                            "> 边界：下文是从原网址取得的正文；自动分类必须引用正文证据，不能根据公众号名猜测。",
                            "",
                            "## 原文正文",
                            "",
                            full_text,
                        ]
                        path_obj = write_raw_material(
                            vault, candidate["title"], "\n".join(lines), article_url,
                            ["微信", "微信公众号", "用户确认", "已读取正文"], garden_type="frontier",
                        )
                    else:
                        lines = [
                            "> 来源：由用户授权，通过 TraceMemo Local HTTP API 按需读取。",
                            f"> 会话：{candidate.get('talker', '')} · 时间范围：{candidate.get('time_range', '') or '未限定'}",
                            "> 边界：以下是经用户选择的消息片段；聊天中的说法不自动视为已核验事实，因此只存入 raw 证据区，不自动生成概念或知识图谱分类。",
                            "",
                            "## 经确认的原始片段",
                            "",
                        ]
                        for item in candidate.get("messages", []):
                            sender = item.get("sender") or "未知发送者"
                            sent_at = item.get("sent_at") or "时间未知"
                            lines.extend([f"### {sender} · {sent_at}", "", str(item.get("content", "")).strip(), ""])
                        path_obj = write_raw_material(
                            vault, candidate["title"], "\n".join(lines), "",
                            ["微信", "TraceMemo", "用户确认", "待事实核验"], garden_type="interest",
                        )
                    raw_path = str(path_obj.resolve().relative_to(Path(vault).resolve())).replace("\\", "/")
                    # Ordinary chat is personal evidence, not a factual source.
                    # Official-account articles may enter the compiler only after
                    # their actual body has been fetched from the original URL.
                    ingest = ingest_raw(vault, raw_path, STORE) if source_kind == "official_account" else None
                    update_agents_manifest(vault)
                    sync_vault(vault, STORE)
                reviewed = STORE.review_wechat_candidate(candidate_id, accepted, raw_path)
                self._json({"ok": True, "candidate": reviewed, "ingest": ingest})
            elif path == "/api/llm/recheck":
                self._json({"ok": True, "settings": refresh_llm_health()})
            elif path == "/api/sync":
                vault = body.get("vault_path") or STORE.setting("vault_path", "")
                result = sync_vault(vault, STORE)
                patrol = patrol_vault(vault, STORE)
                if patrol["ingested"] or patrol["manifest"]["changed"]:
                    result = sync_vault(vault, STORE)
                self._json({"ok": True, "result": result, "agent": patrol})
            elif path == "/api/textbooks/import":
                directory = body.get("directory") or STORE.setting("textbook_directory", "") or str(DATA_DIR / "textbook_kb")
                max_pages = body.get("max_pages")
                result = ingest_pdf_directory(directory, STORE, max_pages=max_pages)
                if body.get("auto_ocr", True) and windows_ocr_available():
                    selected = body.get("ocr_books") or []
                    includes = tuple(str(item).strip() for item in selected if str(item).strip())
                    result["ocr"] = start_background_textbook_ocr(STORE, directory, includes=includes)
                self._json({"ok": True, "result": result})
            elif path == "/api/textbooks/ocr/start":
                directory = body.get("directory") or STORE.setting("textbook_directory", "") or str(DATA_DIR / "textbook_kb")
                includes = tuple(str(item).strip() for item in body.get("books", []) if str(item).strip())
                self._json({"ok": True, "status": start_background_textbook_ocr(STORE, directory, includes=includes)})
            elif path == "/api/links/rebuild":
                semantic = build_semantic_links(STORE)
                classification = classify_unmounted_concepts(STORE)
                hierarchy = rebuild_concept_hierarchy(STORE, force=True)
                self._json({
                    "ok": True,
                    "created": semantic + classification["classified"] + hierarchy["relations"],
                    "semantic": semantic, "classification": classification, "hierarchy": hierarchy,
                })
            elif path == "/api/domains/rebuild":
                # A manual rebuild is the explicit opt-in point for upgrading
                # cached evidence-only classifications with LangChain.  Normal
                # textbook imports keep using the fast cache-backed path.
                self._json({"ok": True, "created": rebuild_domain_map(STORE, force_model=True)})
            elif path == "/api/links/review":
                ok = STORE.review_link(int(body.get("id", 0)), bool(body.get("accepted")))
                self._json({"ok": ok, "stats": STORE.stats()})
            elif path == "/api/ingest":
                vault = body.get("vault_path") or STORE.setting("vault_path", "")
                raw_file = str(body.get("raw_file", "")).replace("\\", "/")
                result = ingest_raw(vault, raw_file, STORE)
                result["agents"] = update_agents_manifest(vault)
                self._json({"ok": True, "result": result})
            elif path == "/api/links/validate":
                vault = body.get("vault_path") or STORE.setting("vault_path", "")
                self._json({"ok": True, "result": validate_links(vault)})
            elif path == "/api/analyze":
                result = analyze_frontier(STORE, str(body.get("title", "前沿材料")), str(body.get("text", "")), str(body.get("url", "")))
                vault = STORE.setting("vault_path", "")
                if vault and result.get("raw_path"):
                    raw_relative = str(Path(result["raw_path"]).resolve().relative_to(Path(vault).resolve())).replace("\\", "/")
                    result["ingest"] = ingest_raw(vault, raw_relative, STORE)
                    result["agents"] = update_agents_manifest(vault)
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            elif path == "/api/agent/ask":
                history = body.get("history") if isinstance(body.get("history"), list) else []
                self._json({"ok": True, "result": answer_from_wiki(
                    STORE,
                    str(body.get("question", "")),
                    history,
                    str(body.get("session_id", "")) or None,
                )})
            elif path == "/api/agent/personalization-feedback":
                helpful_value = body.get("helpful")
                if not isinstance(helpful_value, bool):
                    raise ValueError("helpful 必须是 true 或 false")
                self._json({"ok": True, "result": MEMORY.record_personalization_feedback(
                    request_id=str(body.get("request_id", "")),
                    helpful=helpful_value,
                    feedback_note=str(body.get("feedback_note", "")),
                )})
            elif path == "/api/agent/save":
                result = save_agent_insight(
                    STORE,
                    str(body.get("question", "")),
                    str(body.get("answer", "")),
                    body.get("citations") if isinstance(body.get("citations"), list) else [],
                    body.get("web_sources") if isinstance(body.get("web_sources"), list) else [],
                    str(body.get("followup", "")),
                    body.get("messages") if isinstance(body.get("messages"), list) else [],
                )
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            elif path == "/api/agent/hint":
                self._json({"ok": True, "result": hint_for_task(STORE, int(body.get("task_id", 0)))})
            elif path == "/api/agent/patrol":
                vault = body.get("vault_path") or STORE.setting("vault_path", "")
                result = patrol_vault(vault, STORE)
                result["sync"] = sync_vault(vault, STORE)
                self._json({"ok": True, "result": result})
            elif path == "/api/memory/reflect":
                self._json({"ok": True, "result": MEMORY.reflect(force=True)})
            elif path == "/api/daily/save":
                vault = STORE.setting("vault_path", "")
                if not vault:
                    raise ValueError("请先连接 Obsidian Vault")
                title = str(body.get("title", "今日推荐")).strip()
                abstract = str(body.get("abstract", "")).strip()
                url = str(body.get("url", "")).strip()
                interest = str(body.get("interest", "兴趣推荐")).strip()
                deep_read = body.get("deep_read") if isinstance(body.get("deep_read"), dict) else {}
                analysis = deep_read.get("analysis") if isinstance(deep_read.get("analysis"), dict) else {}
                deep_sections: list[str] = []
                if analysis:
                    findings = analysis.get("findings") if isinstance(analysis.get("findings"), list) else []
                    finding_lines = [
                        f"- {item.get('claim')}" + (f"\n  - 原文证据：{item.get('evidence')}" if item.get("evidence") else "")
                        for item in findings if isinstance(item, dict) and item.get("claim")
                    ]
                    connections = (
                        analysis.get("local_connections")
                        if isinstance(analysis.get("local_connections"), list) else []
                    )
                    connection_lines = []
                    for item in connections:
                        if not isinstance(item, dict) or not item.get("title"):
                            continue
                        line = f"- 《{item.get('title')}》：{item.get('bridge') or ''}"
                        if item.get("path"):
                            line += f"\n  - 本地来源：{item.get('path')}"
                        if item.get("mastery"):
                            line += (
                                f"\n  - 个性化证据：{item['mastery'].get('concept') or ''} · "
                                f"{item['mastery'].get('stage') or ''}（复习/作答记录）"
                            )
                        else:
                            line += "\n  - 边界：仅本地资料命中，不代表用户已经掌握。"
                        connection_lines.append(line)
                    deep_sections = [
                        f"## 深读范围\n{deep_read.get('scope_label') or deep_read.get('scope') or '未知'}：{deep_read.get('source_note') or ''}",
                        f"## 研究问题\n{analysis.get('problem') or ''}",
                        f"## 创新点\n{analysis.get('novelty') or ''}",
                        f"## 方法\n{analysis.get('method') or ''}",
                        "## 核心发现与证据\n" + ("\n".join(finding_lines) or "当前没有通过逐字证据校验的核心发现。"),
                        "## 与本地知识的连接\n" + (
                            "\n".join(connection_lines)
                            or str(analysis.get("local_connection_note") or "没有足够强的本地连接，未强行关联教材。")
                        ),
                        "## 局限\n" + "\n".join(f"- {item}" for item in (analysis.get("limitations") or [])),
                    ]
                material = abstract or "园丁发现了这条与你学习画像相关的资料，等待进一步阅读。"
                if deep_sections:
                    material += "\n\n" + "\n\n".join(deep_sections)
                raw_path = write_raw_material(
                    vault, title,
                    material + f"\n\n原始链接：{url}",
                    url, [interest, "每日推荐", "待阅读"],
                )
                patrol = patrol_vault(vault, STORE)
                sync = sync_vault(vault, STORE)
                self._json({"ok": True, "result": {"path": str(raw_path), "patrol": patrol, "sync": sync}})
            elif path == "/api/daily/read":
                url = str(body.get("url", "")).strip()
                if not url:
                    raise ValueError("缺少文章链接")
                read_urls = [str(item) for item in (STORE.setting("frontier_read_urls", []) or [])]
                if url not in read_urls:
                    read_urls.append(url)
                    STORE.set_setting("frontier_read_urls", read_urls[-500:])
                    STORE.add_activity("frontier_read", str(body.get("title", "前沿文章"))[:100], 2)
                self._json({"ok": True, "read": True})
            elif path == "/api/daily/deep-read":
                article = body.get("article") if isinstance(body.get("article"), dict) else body
                self._json({
                    "ok": True,
                    "result": deep_read_paper(STORE, article, force=bool(body.get("force"))),
                })
            elif path == "/api/interest":
                tags = body.get("tags", [])
                if isinstance(tags, str):
                    tags = [part.strip() for part in tags.replace("，", ",").split(",") if part.strip()]
                result = add_interest(STORE, str(body.get("title", "灵感碎片")), str(body.get("content", "")), tags)
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            elif path == "/api/inspiration/ask":
                history = body.get("history") if isinstance(body.get("history"), list) else []
                self._json({"ok": True, "result": explore_inspiration(
                    STORE, str(body.get("message", "")), history,
                    str(body.get("session_id", "")) or None,
                )})
            elif path == "/api/inspiration/save":
                messages = body.get("messages") if isinstance(body.get("messages"), list) else []
                latest = body.get("latest") if isinstance(body.get("latest"), dict) else {}
                tags = body.get("tags") if isinstance(body.get("tags"), list) else []
                result = save_inspiration_seed(STORE, str(body.get("title", "")), messages, latest, tags)
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            elif path == "/api/tasks/complete":
                self._json({"ok": STORE.complete_task(int(body.get("id", 0))), "stats": STORE.stats()})
            elif path == "/api/tasks/answer":
                task = STORE.get_task(int(body.get("id", 0)))
                if not task:
                    raise ValueError("复习任务不存在")
                history = body.get("history") if isinstance(body.get("history"), list) else []
                refs = search_notes(STORE, task.get("concept", ""), kinds={"concept", "bridge", "knowledge", "course"}, limit=3)
                context = "\n".join(f"[{item['title']}] {item['snippet']}" for item in refs)
                assessment = evaluate_review(
                    task, body.get("answer"), int(body.get("self_rating", 2)), history, context,
                    {"learning_level": STORE.setting("learning_level", "本科入门"), "interests": STORE.setting("interests", [])},
                )
                mastery_plan = MEMORY.plan_mastery_update(
                    task, assessment["quality"], int(body.get("self_rating", 2))
                )
                if assessment.get("needs_followup"):
                    result = {**assessment, "completed": False, "earned_xp": 0}
                else:
                    result = STORE.record_review(
                        task["id"], assessment["quality"], assessment["feedback"],
                        str(body.get("answer", "")), mastery_plan["next_interval_days"],
                    )
                    result.update({
                        "correct": assessment.get("correct"), "understood": assessment.get("understood", ""),
                        "followup": assessment.get("followup", ""), "completed": True,
                    })
                result["mastery"] = MEMORY.apply_mastery_update(
                    mastery_plan,
                    answer=str(body.get("answer", "")),
                    task_id=task["id"],
                )
                self._json({"ok": True, "result": result, "stats": STORE.stats()})
            elif path == "/api/feeds":
                name = str(body.get("name", "")).strip()
                if not name:
                    raise ValueError("请填写要追踪的博主或订阅源名称")
                source = describe_feed(str(body.get("url", "")).strip())
                feed_id = STORE.add_feed(name, str(source["url"]))
                self._json({
                    "ok": True, "id": feed_id, "source": source, "feeds": list_followed_sources(STORE),
                })
            elif path == "/api/feeds/refresh":
                result = refresh_feeds(STORE)
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            elif path == "/api/bilibili/video/read":
                result = read_bilibili_video(
                    STORE,
                    str(body.get("url") or body.get("bvid") or ""),
                    allow_asr=bool(body.get("allow_asr")),
                    page=int(body.get("page") or 1),
                )
                result["sync"] = sync_configured_vault()
                self._json({"ok": True, "result": result})
            else:
                self._json({"error": "接口不存在"}, 404)
        except Exception as exc:
            self._error(exc)

    def _error(self, exc: Exception) -> None:
        traceback.print_exc()
        status = 400 if isinstance(exc, (ValueError, FileNotFoundError)) else 500
        self._json({"ok": False, "error": str(exc) or exc.__class__.__name__}, status)


class GardenHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description="知识花园智能体")
    parser.add_argument("--host", default=os.getenv("GARDEN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GARDEN_PORT", "8765")))
    args = parser.parse_args()
    server = GardenHTTPServer((args.host, args.port), GardenHandler)
    # A provider connection check can take tens of seconds when the network or
    # upstream model is slow.  Binding the local server must not wait for that
    # remote round trip: the UI can render the existing "checking" state and
    # pick up the final health status through its normal polling endpoint.
    health = llm_health()
    print(f"[Knowledge Garden] {health['llm_message']}")
    stop_event = threading.Event()
    model_warm = threading.Thread(
        target=model_warmups,
        args=(stop_event,),
        daemon=True,
        name="model-warmups",
    )
    model_warm.start()
    interval = int(os.getenv("GARDEN_FEED_INTERVAL_MINUTES", "60"))
    patrol = threading.Thread(target=feed_patrol, args=(stop_event, interval), daemon=True, name="feed-patrol")
    patrol.start()
    vault_interval = int(os.getenv("GARDEN_VAULT_INTERVAL_SECONDS", "8"))
    vault_watch = threading.Thread(target=vault_patrol, args=(stop_event, vault_interval), daemon=True, name="vault-patrol")
    vault_watch.start()
    memory_interval = int(os.getenv("GARDEN_MEMORY_INTERVAL_MINUTES", "30"))
    memory_watch = threading.Thread(
        target=memory_patrol,
        args=(stop_event, memory_interval),
        daemon=True,
        name="memory-patrol",
    )
    memory_watch.start()
    retrieval_warm = threading.Thread(
        target=retrieval_warmup,
        args=(stop_event,),
        daemon=True,
        name="retrieval-warmup",
    )
    retrieval_warm.start()
    print(f"\n[Knowledge Garden] 知识花园已启动：http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止。API Key 仅在进程内存中解密使用。\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n知识花园已休眠。")
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
