from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from core.config import ROOT
from core.llm import LLMError, chat_json
from core.storage import GardenStore


PROJECT_ROOT = ROOT
MCP_HOME = PROJECT_ROOT / "vendor" / "bilibili-home"
_BVID = re.compile(r"(?:video/)?(BV[0-9A-Za-z]{10,})", re.IGNORECASE)
_BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class BilibiliMCPError(RuntimeError):
    pass


def _request_json(url: str, *, referer: str = "https://www.bilibili.com/") -> dict[str, Any]:
    request = Request(url, headers={
        "User-Agent": _BROWSER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        "Referer": referer,
    })
    with urlopen(request, timeout=20) as response:
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise ValueError("B站字幕响应过大")
    result = json.loads(payload)
    return result if isinstance(result, dict) else {}


def _subtitle_timestamp(seconds: Any) -> str:
    value = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def inspect_public_video(bvid_or_url: str, *, page: int = 1) -> dict[str, Any]:
    """Read public subtitles without claiming access to frames or audio."""
    match = _BVID.search(str(bvid_or_url or ""))
    if not match:
        raise ValueError("请输入有效的 B站 BV 号或视频链接")
    bvid = match.group(1)
    source_url = f"https://www.bilibili.com/video/{bvid}"
    view = _request_json(
        "https://api.bilibili.com/x/web-interface/view?" + urlencode({"bvid": bvid}),
        referer=source_url,
    )
    data = view.get("data") if isinstance(view.get("data"), dict) else {}
    pages = data.get("pages") if isinstance(data.get("pages"), list) else []
    page_index = max(0, min(len(pages) - 1, int(page) - 1)) if pages else 0
    cid = (pages[page_index] if pages else {}).get("cid")
    metadata = {
        "title": str(data.get("title") or bvid),
        "author": str((data.get("owner") or {}).get("name") or ""),
        "description": str(data.get("desc") or ""),
        "page": page_index + 1,
        "cid": cid,
    }
    if not cid:
        return {
            "status": "metadata_only", "message": "已读取视频元数据，但没有取得分P标识。",
            "bvid": bvid, "source_url": source_url, "metadata": metadata,
        }
    player = _request_json(
        "https://api.bilibili.com/x/player/v2?" + urlencode({"bvid": bvid, "cid": cid}),
        referer=source_url,
    )
    player_data = player.get("data") if isinstance(player.get("data"), dict) else {}
    subtitle = player_data.get("subtitle") if isinstance(player_data.get("subtitle"), dict) else {}
    tracks = subtitle.get("subtitles") if isinstance(subtitle.get("subtitles"), list) else []
    if not tracks:
        return {
            "status": "no_subtitle", "message": "该视频没有向游客公开字幕，可在本机授权后重试或显式启用 ASR。",
            "bvid": bvid, "source_url": source_url, "metadata": metadata,
        }
    language_priority = ("ai-zh", "zh-Hans", "zh-CN", "zh-Hant", "zh")
    track = min(
        tracks,
        key=lambda item: language_priority.index(str(item.get("lan") or ""))
        if str(item.get("lan") or "") in language_priority else len(language_priority),
    )
    subtitle_url = str(track.get("subtitle_url") or "").strip()
    if subtitle_url.startswith("//"):
        subtitle_url = "https:" + subtitle_url
    if not subtitle_url.startswith("https://"):
        return {
            "status": "no_subtitle", "message": "字幕轨道存在，但没有返回可安全读取的 HTTPS 地址。",
            "bvid": bvid, "source_url": source_url, "metadata": metadata,
        }
    subtitle_json = _request_json(subtitle_url, referer=source_url)
    segments = subtitle_json.get("body") if isinstance(subtitle_json.get("body"), list) else []
    lines = []
    for segment in segments:
        if not isinstance(segment, dict) or not str(segment.get("content") or "").strip():
            continue
        lines.append(
            f"[{_subtitle_timestamp(segment.get('from'))} --> {_subtitle_timestamp(segment.get('to'))}] "
            + str(segment.get("content") or "").strip()
        )
    transcript = "\n".join(lines).strip()
    if not transcript:
        return {
            "status": "no_subtitle", "message": "字幕轨道为空，可显式启用本地 ASR。",
            "bvid": bvid, "source_url": source_url, "metadata": metadata,
        }
    language = str(track.get("lan_doc") or track.get("lan") or "未知语言")
    ai_generated = str(track.get("lan") or "").lower().startswith("ai-") or "自动" in language
    return {
        "status": "ready", "message": f"已读取{language}公开字幕。", "bvid": bvid,
        "source_url": source_url, "metadata": metadata, "transcript": transcript,
        "language": language, "data_source": "public_ai_subtitle" if ai_generated else "public_subtitle",
        "segment_count": len(lines),
    }


def _find_node() -> Path | None:
    configured = os.getenv("GARDEN_NODE_EXE", "").strip()
    candidates: list[Path] = [Path(configured)] if configured else []
    candidates.append(PROJECT_ROOT / "vendor" / "node" / ("node.exe" if os.name == "nt" else "node"))
    discovered = shutil.which("node")
    if discovered:
        candidates.append(Path(discovered))
    user = Path(os.environ.get("USERPROFILE") or Path.home())
    candidates.extend(user.glob(".cache/codex-runtimes/*/dependencies/node/bin/node.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_package_file(relative: str) -> Path | None:
    configured = os.getenv("GARDEN_BILIBILI_MCP_ROOT", "").strip()
    roots: list[Path] = [Path(configured)] if configured else []
    roots.extend(
        path.parent
        for path in (PROJECT_ROOT / "vendor" / "pnpm-store").glob(
            "v*/links/@xzxzzx/bilibili-mcp/*/*/node_modules/@xzxzzx/bilibili-mcp/package.json"
        )
    )
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _mcp_environment() -> dict[str, str]:
    env = dict(os.environ)
    MCP_HOME.mkdir(parents=True, exist_ok=True)
    # Node's os.homedir() uses USERPROFILE on Windows. Limit the override to the
    # child process so credentials and optional ASR assets remain on drive D.
    if os.name == "nt":
        env["USERPROFILE"] = str(MCP_HOME)
    env.setdefault("BILIBILI_REQUEST_TIMEOUT_MS", "20000")
    return env


def runtime_status() -> dict[str, Any]:
    node = _find_node()
    entry = _find_package_file("dist/index.js")
    if not node:
        return {
            "installed": False,
            "configured": False,
            "logged_in": False,
            "public_subtitle_fallback": True,
            "message": "缺少 Node.js 20+，Bilibili MCP 暂不可用。",
        }
    if not entry:
        return {
            "installed": False,
            "configured": False,
            "logged_in": False,
            "public_subtitle_fallback": True,
            "message": "尚未在项目 D 盘缓存 @xzxzzx/bilibili-mcp。",
        }
    try:
        payload = call_tool("check_bilibili_credentials", {}, timeout=18)
    except BilibiliMCPError as exc:
        return {
            "installed": True,
            "configured": False,
            "logged_in": False,
            "public_subtitle_fallback": True,
            "message": str(exc),
        }
    return {
        "installed": True,
        "configured": bool(payload.get("configured")),
        "logged_in": bool(payload.get("logged_in")),
        "public_subtitle_fallback": True,
        "source": payload.get("source", "none"),
        "next_steps": payload.get("next_steps_zh") or payload.get("next_steps") or [],
        "message": (
            "Bilibili MCP 已登录，可以读取字幕。"
            if payload.get("logged_in")
            else "Bilibili MCP 已接入，等待你在本地完成一次 B站登录授权。"
        ),
        "data_dir": str(MCP_HOME / ".bilibili-mcp"),
    }


def _start_server() -> tuple[subprocess.Popen[str], queue.Queue[dict[str, Any]], list[str]]:
    node = _find_node()
    entry = _find_package_file("dist/index.js")
    if not node or not entry:
        raise BilibiliMCPError("Bilibili MCP 运行时尚未安装完整")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(node), str(entry)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_mcp_environment(),
        creationflags=flags,
    )
    messages: queue.Queue[dict[str, Any]] = queue.Queue()
    errors: list[str] = []

    def read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            clean = line.strip()
            if clean:
                errors.append(clean)
                del errors[:-12]

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    return process, messages, errors


def _send(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise BilibiliMCPError("Bilibili MCP 标准输入不可用")
    process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _wait_for(
    process: subprocess.Popen[str], messages: queue.Queue[dict[str, Any]], request_id: int,
    errors: list[str], timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None and messages.empty():
            detail = errors[-1] if errors else f"退出码 {process.returncode}"
            raise BilibiliMCPError(f"Bilibili MCP 意外退出：{detail}")
        try:
            message = messages.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if message.get("id") != request_id:
            continue
        if message.get("error"):
            error = message["error"]
            raise BilibiliMCPError(str(error.get("message") if isinstance(error, dict) else error))
        result = message.get("result")
        return result if isinstance(result, dict) else {"value": result}
    raise BilibiliMCPError(f"Bilibili MCP 调用超时（{timeout:.0f} 秒）")


def _structured_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("isError"):
        text = ""
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text += str(item.get("text") or "")
        raise BilibiliMCPError(text.strip() or "Bilibili MCP 工具返回错误")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content") or []:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = str(item.get("text") or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return result


def call_tool(name: str, arguments: dict[str, Any], *, timeout: float = 35) -> dict[str, Any]:
    process, messages, errors = _start_server()
    try:
        _send(process, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "knowledge-garden", "version": "1.0"},
            },
        })
        _wait_for(process, messages, 1, errors, min(timeout, 12))
        _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        _send(process, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        return _structured_result(_wait_for(process, messages, 2, errors, timeout))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def _video_analysis(title: str, transcript: str, data_source: str) -> dict[str, Any]:
    fallback = {
        "overview": "已取得视频转录，等待模型进一步提炼。",
        "key_points": [],
        "concepts": [],
        "caveats": ["请结合带时间戳的原始转录核对关键主张。"],
        "chapter_outline": [],
        "questions": ["视频最核心的可验证主张是什么？"],
    }
    if len(re.sub(r"\s+", "", transcript)) < 120:
        return fallback
    try:
        result = chat_json(
            "你是严谨的视频学术导读编辑。只根据提供的字幕/ASR转录分析，不把UP主观点自动当作事实。只返回JSON。关键点必须保留对应时间戳或短原话；无法从转录确认时明确写入caveats。",
            f"标题：{title}\n转录来源：{data_source}\n转录：\n{transcript[:28000]}\n\n"
            "返回 overview（2~4句）、key_points（3~7项，每项含 point、evidence、timestamp）、concepts、"
            "chapter_outline（每项含 title、timestamp、summary）、caveats、questions（2~4项）。",
            timeout=75,
            max_retries=1,
        )
    except LLMError:
        result = None
    if not isinstance(result, dict):
        return fallback
    return {
        key: result.get(key, fallback[key])
        for key in fallback
    }


def read_video(
    store: GardenStore, bvid_or_url: str, *, allow_asr: bool = False, page: int = 1,
) -> dict[str, Any]:
    match = _BVID.search(str(bvid_or_url or ""))
    if not match:
        raise ValueError("请输入有效的 B站 BV 号或视频链接")
    bvid = match.group(1)
    public_result: dict[str, Any] = {}
    try:
        public_result = inspect_public_video(bvid, page=page)
    except Exception:
        public_result = {}
    if public_result.get("status") == "ready":
        metadata = public_result.get("metadata") or {}
        transcript = public_result
    else:
        try:
            metadata = call_tool("get_video_metadata", {"bvid_or_url": bvid}, timeout=30)
            transcript = call_tool(
                "get_video_transcript",
                {
                    "bvid_or_url": bvid,
                    "page": max(1, int(page)),
                    "preferred_lang": "zh-Hans",
                    "include_timestamps": True,
                    "fallback_to_asr": bool(allow_asr),
                },
                timeout=1900 if allow_asr else 45,
            )
        except BilibiliMCPError:
            credential = call_tool("check_bilibili_credentials", {}, timeout=18)
            if not credential.get("logged_in"):
                detail = str(public_result.get("message") or "公开字幕读取失败")
                raise ValueError(
                    f"{detail} Bilibili MCP 已安装但尚未完成本地登录授权；"
                    "请运行项目根目录的“配置B站视频解析.cmd”。"
                )
            raise
    transcript_text = str(
        transcript.get("transcript")
        or transcript.get("text")
        or transcript.get("subtitle_text")
        or ""
    ).strip()
    data_source = str(transcript.get("data_source") or "subtitle")
    title = str(metadata.get("title") or transcript.get("title") or bvid)
    analysis = _video_analysis(title, transcript_text, data_source)
    source_url = str(transcript.get("source_url") or f"https://www.bilibili.com/video/{bvid}")
    content = "\n\n".join([
        f"> 视频来源：{source_url}",
        f"> 内容依据：{data_source}；AI字幕和ASR都可能识别错误，关键表述应回到时间戳核对。",
        "## 导读摘要\n" + str(analysis.get("overview") or ""),
        "## 带时间戳的转录\n" + (transcript_text or "未取得可用字幕。"),
    ])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    note_id, _ = store.upsert_note({
        "path": f"bilibili::{bvid}::p{max(1, int(page))}",
        "title": title,
        "kind": "frontier",
        "content": content,
        "tags": ["前沿", "B站", "视频解析", data_source],
        "source": "Bilibili MCP",
        "source_url": source_url,
        "content_hash": digest,
    })
    store.add_activity("bilibili_read", title[:100], 4)
    return {
        "bvid": bvid,
        "title": title,
        "source_url": source_url,
        "data_source": data_source,
        "transcript": transcript_text,
        "analysis": analysis,
        "metadata": metadata,
        "note_id": note_id,
        "asr_used": data_source == "asr",
    }


def run_setup() -> int:
    node = _find_node()
    cli = _find_package_file("dist/cli.js")
    if not node or not cli:
        print("Bilibili MCP 尚未下载。请先在知识花园项目中完成 MCP 安装。")
        return 1
    MCP_HOME.mkdir(parents=True, exist_ok=True)
    return subprocess.call([str(node), str(cli), "setup"], env=_mcp_environment())


def run_qr_setup(*, timeout: int = 180) -> int:
    """Authorize with Bilibili's official QR-login flow and save MCP-compatible credentials."""
    try:
        import qrcode
        from http.cookiejar import CookieJar
        from urllib.request import HTTPCookieProcessor, build_opener
    except ImportError:
        print("缺少 qrcode 依赖，请先在项目虚拟环境中安装 qrcode。")
        return 1

    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))

    def qr_request(url: str) -> dict[str, Any]:
        request = Request(url, headers={
            "User-Agent": _BROWSER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.bilibili.com/",
        })
        with opener.open(request, timeout=20) as response:
            payload = json.load(response)
        return payload if isinstance(payload, dict) else {}

    generated = qr_request(
        "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
    )
    data = generated.get("data") if isinstance(generated.get("data"), dict) else {}
    login_url = str(data.get("url") or "").strip()
    qrcode_key = str(data.get("qrcode_key") or "").strip()
    if generated.get("code") != 0 or not login_url or not qrcode_key:
        print("B站没有返回有效的登录二维码，请稍后重试。")
        return 1

    MCP_HOME.mkdir(parents=True, exist_ok=True)
    image_path = MCP_HOME / "bilibili-login-qrcode.png"
    qrcode.make(login_url).save(image_path)
    print(f"QR_IMAGE={image_path}", flush=True)
    print("请使用哔哩哔哩手机 App 扫码，并在手机上确认登录。", flush=True)

    deadline = time.monotonic() + max(30, int(timeout))
    last_code: int | None = None
    while time.monotonic() < deadline:
        poll = qr_request(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/poll?"
            + urlencode({"qrcode_key": qrcode_key})
        )
        poll_data = poll.get("data") if isinstance(poll.get("data"), dict) else {}
        raw_code = poll_data.get("code")
        if raw_code is None:
            time.sleep(2)
            continue
        code = int(raw_code)
        if code != last_code:
            if code == 86101:
                print("等待扫码……", flush=True)
            elif code == 86090:
                print("已扫码，请在手机上点击确认。", flush=True)
            elif code == 86038:
                print("二维码已过期，请重新运行扫码授权。", flush=True)
                return 1
            last_code = code
        if code == 0:
            callback_url = str(poll_data.get("url") or "")
            decoded_url = callback_url
            for _ in range(4):
                expanded = unquote(decoded_url)
                if expanded == decoded_url:
                    break
                decoded_url = expanded
            pairs = {
                match.group(1).lower(): unquote(match.group(2)).strip()
                for match in re.finditer(
                    r"(?:[?&])(SESSDATA|bili_jct|DedeUserID)=([^&#]+)",
                    decoded_url,
                    flags=re.IGNORECASE,
                )
            }
            sessdata = pairs.get("sessdata", "")
            bili_jct = pairs.get("bili_jct", "")
            dedeuserid = pairs.get("dedeuserid", "")
            if callback_url.startswith("https://"):
                try:
                    callback_request = Request(
                        callback_url,
                        headers={"User-Agent": _BROWSER_AGENT, "Referer": "https://www.bilibili.com/"},
                    )
                    with opener.open(callback_request, timeout=20) as response:
                        response.read(1024)
                except Exception:
                    pass
            cookie_values = {cookie.name.lower(): cookie.value for cookie in cookie_jar}
            sessdata = sessdata or cookie_values.get("sessdata", "")
            bili_jct = bili_jct or cookie_values.get("bili_jct", "")
            dedeuserid = dedeuserid or cookie_values.get("dedeuserid", "")
            if not sessdata or not bili_jct or not dedeuserid:
                print("扫码成功，但回调中缺少完整凭证；未写入任何配置。")
                return 1
            config_dir = MCP_HOME / ".bilibili-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            config_path.write_text(json.dumps({
                "sessdata": sessdata,
                "bili_jct": bili_jct,
                "dedeuserid": dedeuserid,
                "expiresAt": int(time.time() * 1000) + 30 * 24 * 60 * 60 * 1000,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                os.chmod(config_path, 0o600)
            except OSError:
                pass
            print("LOGIN_SAVED=1", flush=True)
            print("扫码授权成功，凭证已安全保存到项目 D 盘。", flush=True)
            return 0
        time.sleep(2)
    print("等待扫码超时，请重新运行扫码授权。")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge Garden Bilibili MCP adapter")
    parser.add_argument("command", choices=("setup", "qr-setup", "doctor"))
    args = parser.parse_args()
    if args.command == "setup":
        return run_setup()
    if args.command == "qr-setup":
        return run_qr_setup()
    print(json.dumps(runtime_status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
