from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bilibili_mcp import read_video, runtime_status
from core.config import ROOT
from core.feeds import fetch_source
from core.storage import GardenStore


RESULTS_DIR = ROOT / "evals" / "results"


def latest_source(uid: str, *, creator: str = "", refresh: bool = False, limit: int = 10) -> Path:
    candidates = sorted(
        RESULTS_DIR.glob(f"bilibili-{uid}-latest10-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates and not refresh:
        return candidates[0]
    if not creator:
        raise FileNotFoundError(f"没有找到 UID {uid} 的最新视频清单；刷新时请提供 --creator")
    entries = fetch_source(
        f"https://space.bilibili.com/{uid}", name=creator, limit=max(10, limit),
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"bilibili-{uid}-latest10-{stamp}.json"
    payload = {
        "suite": "bilibili_creator_latest",
        "creator": creator,
        "uid": uid,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "items": [{"rank": index, **item} for index, item in enumerate(entries, 1)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['creator']} 最新视频登录后解析报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 成功解析：{report['parsed']}/{report['requested']}",
        f"- 总耗时：{report['elapsed_seconds']} 秒",
        f"- 边界：{report['boundary']}",
        "",
    ]
    for item in report["items"]:
        lines.extend([
            f"## {item['rank']}. {item['title']}",
            "",
            f"- 地址：{item['url']}",
            f"- 状态：{item['status']}",
            f"- 内容依据：{item.get('data_source') or '无'}",
            f"- 字幕字符数：{item.get('transcript_chars', 0)}",
            "",
        ])
        analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else {}
        overview = str(analysis.get("overview") or item.get("error") or "").strip()
        if overview:
            lines.extend(["### 导读", "", overview, ""])
        key_points = analysis.get("key_points") if isinstance(analysis.get("key_points"), list) else []
        if key_points:
            lines.extend(["### 关键点", ""])
            for point in key_points:
                if isinstance(point, dict):
                    point_text = str(point.get("point") or point.get("summary") or "").strip()
                    timestamp = str(point.get("timestamp") or "").strip()
                    evidence = str(point.get("evidence") or "").strip()
                    suffix = f"（{timestamp}）" if timestamp else ""
                    lines.append(f"- {point_text}{suffix}")
                    if evidence:
                        lines.append(f"  - 字幕依据：{evidence}")
                elif str(point).strip():
                    lines.append(f"- {str(point).strip()}")
            lines.append("")
        caveats = analysis.get("caveats") if isinstance(analysis.get("caveats"), list) else []
        if caveats:
            lines.extend(["### 阅读边界", ""])
            lines.extend(f"- {str(value).strip()}" for value in caveats if str(value).strip())
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    report["parsed"] = sum(1 for item in report["items"] if item.get("status") == "ready")
    report["elapsed_seconds"] = round(time.monotonic() - report["_started_monotonic"], 2)
    serializable = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(serializable), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", default="520819684")
    parser.add_argument("--creator", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-asr", action="store_true")
    args = parser.parse_args()

    source_path = latest_source(
        args.uid, creator=args.creator, refresh=args.refresh, limit=args.limit,
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_items = list(source.get("items") or [])
    start_index = max(0, int(args.start) - 1)
    selected = source_items[start_index : start_index + max(1, args.limit)]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"bilibili-{args.uid}-authenticated-{stamp}.json"
    md_path = json_path.with_suffix(".md")
    report: dict[str, Any] = {
        "suite": "bilibili_creator_authenticated",
        "creator": source.get("creator") or args.uid,
        "uid": args.uid,
        "requested": len(selected),
        "parsed": 0,
        "boundary": "导读只依据实际取得的字幕；AI字幕可能识别错误，重要主张必须回到时间戳和外部来源复核。",
        "runtime": runtime_status(),
        "source_list": str(source_path),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": 0,
        "items": [],
        "_started_monotonic": time.monotonic(),
    }
    store = GardenStore()
    for index, source_item in enumerate(selected, 1):
        started = time.monotonic()
        item: dict[str, Any] = {
            "rank": source_item.get("rank") or index,
            "title": source_item.get("title") or "",
            "url": source_item.get("url") or "",
            "status": "error",
        }
        try:
            result = read_video(store, item["url"], allow_asr=args.allow_asr)
            transcript = str(result.get("transcript") or "")
            item.update({
                "title": result.get("title") or item["title"],
                "status": "ready" if transcript.strip() else "empty_transcript",
                "data_source": result.get("data_source"),
                "transcript_chars": len(transcript),
                "transcript_excerpt": transcript[:800],
                "analysis": result.get("analysis"),
                "note_id": result.get("note_id"),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
        except Exception as exc:
            item.update({
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
        report["items"].append(item)
        save_report(report, json_path, md_path)
        print(
            f"PROGRESS {index}/{len(selected)} {item['status']} "
            f"{item.get('data_source') or '-'} {item['elapsed_seconds']}s",
            flush=True,
        )
    print(f"REPORT_JSON={json_path}", flush=True)
    print(f"REPORT_MD={md_path}", flush=True)
    return 0 if report["parsed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
