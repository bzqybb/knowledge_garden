from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ROOT
from evals.bilibili_creator_eval import render_markdown as render_bilibili
from evals.wechat_official_eval import render_markdown as render_wechat


RESULTS = ROOT / "evals" / "results"


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def active_latency(items: list[dict]) -> dict:
    values = sorted(float(item.get("elapsed_seconds") or 0) for item in items)
    if not values:
        return {"count": 0, "median_seconds": 0, "max_seconds": 0}
    return {
        "count": len(values),
        "median_seconds": round(median(values), 2),
        "max_seconds": round(max(values), 2),
    }


def main() -> int:
    bili_base = load("bilibili-520819684-authenticated-20260827-013451.json")
    bili_tail = load("bilibili-520819684-authenticated-20260827-071251.json")
    bili_items = list(bili_base["items"][:7]) + list(bili_tail["items"])
    bili_final = {
        **bili_base,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested": 10,
        "parsed": sum(1 for item in bili_items if item.get("status") == "ready"),
        "items": bili_items,
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds") or 0) for item in bili_items), 2),
        "active_latency": active_latency(bili_items),
        "consolidated_from": [
            "bilibili-520819684-authenticated-20260827-013451.json",
            "bilibili-520819684-authenticated-20260827-071251.json",
        ],
    }
    bili_json = RESULTS / "bilibili-520819684-latest10-final.json"
    bili_md = bili_json.with_suffix(".md")
    bili_json.write_text(json.dumps(bili_final, ensure_ascii=False, indent=2), encoding="utf-8")
    bili_md.write_text(render_bilibili(bili_final), encoding="utf-8")

    wechat_base = load("wechat-新智元-latest20-20260827-072552.json")
    wechat_tail = load("wechat-新智元-latest10-20260827-075718.json")
    wechat_items = list(wechat_base["items"][:10])
    for offset, item in enumerate(wechat_tail["items"], 11):
        merged = dict(item)
        merged["rank"] = offset
        wechat_items.append(merged)
    wechat_final = {
        **wechat_base,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested": 20,
        "fulltext_ready": sum(1 for item in wechat_items if item.get("access_scope") == "open_fulltext"),
        "guided": sum(1 for item in wechat_items if item.get("status") == "guided"),
        "items": wechat_items,
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds") or 0) for item in wechat_items), 2),
        "active_latency": active_latency(wechat_items),
        "consolidated_from": [
            "wechat-新智元-latest20-20260827-072552.json",
            "wechat-新智元-latest10-20260827-075718.json",
        ],
    }
    wechat_json = RESULTS / "wechat-新智元-latest20-final.json"
    wechat_md = wechat_json.with_suffix(".md")
    wechat_json.write_text(json.dumps(wechat_final, ensure_ascii=False, indent=2), encoding="utf-8")
    wechat_md.write_text(render_wechat(wechat_final), encoding="utf-8")

    print(f"BILIBILI_JSON={bili_json}")
    print(f"BILIBILI_MD={bili_md}")
    print(f"WECHAT_JSON={wechat_json}")
    print(f"WECHAT_MD={wechat_md}")
    print(json.dumps({
        "bilibili": {"parsed": bili_final["parsed"], "latency": bili_final["active_latency"]},
        "wechat": {
            "fulltext_ready": wechat_final["fulltext_ready"],
            "guided": wechat_final["guided"],
            "latency": wechat_final["active_latency"],
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
