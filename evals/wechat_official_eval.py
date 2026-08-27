from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ROOT
from core.llm import LLMError, chat_json
from core.storage import GardenStore
from core.tracememo import TraceMemoClient, tracememo_config
from core.web_research import fetch_wechat_article_text


RESULTS_DIR = ROOT / "evals" / "results"


def guide_article(title: str, description: str, body: str) -> dict[str, Any]:
    fallback = {
        "overview": description or "已取得正文，但模型导读暂时不可用。",
        "why_read": "",
        "key_claims": [],
        "verification_questions": ["文章的核心主张是否有一手来源或可复现实验支持？"],
        "caveats": ["公众号文章是二手报道，标题和作者判断不自动等于已核验事实。"],
        "tags": [],
        "generated": False,
    }
    try:
        result = chat_json(
            "你是严谨的科技文章导读编辑。只能根据提供的文章正文总结，必须把“文章声称”与“已独立核验”区分开。"
            "不要复述长段原文，不得根据标题补写正文没有的信息。只返回JSON。",
            f"标题：{title}\n卡片摘要：{description}\n正文：\n{body[:32000]}\n\n"
            "返回 overview（3~5句连贯导读）、why_read（适合什么读者、为什么值得看）、"
            "key_claims（3~6项，每项含 claim、support、confidence，其中support只写短证据概括，不长引原文）、"
            "verification_questions（2~4项）、caveats（至少指出信息源和未核验边界）、tags（3~6项）。",
            timeout=75,
            max_retries=1,
        )
    except LLMError:
        result = None
    if not isinstance(result, dict):
        return fallback
    guided = {key: result.get(key, value) for key, value in fallback.items()}
    guided["generated"] = True
    return guided


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['account']}最新文章导读测试",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 正文成功：{report['fulltext_ready']}/{report['requested']}",
        f"- 完整导读：{report['guided']}/{report['requested']}",
        f"- 总耗时：{report['elapsed_seconds']} 秒",
        "- 边界：只有实际回源取得正文的文章才标记为完整导读；公众号主张不自动视为事实。",
        "",
    ]
    for item in report["items"]:
        lines.extend([
            f"## {item['rank']}. {item['title']}",
            "",
            f"- 发布时间：{item.get('sent_at') or '未知'}",
            f"- 原文：{item['url']}",
            f"- 状态：{item['status']}",
            f"- 访问范围：{item['access_scope']}",
            f"- 正文字数：{item.get('body_chars', 0)}",
            f"- 耗时：{item.get('elapsed_seconds', 0)} 秒",
            "",
        ])
        guide = item.get("guide") if isinstance(item.get("guide"), dict) else {}
        overview = str(guide.get("overview") or item.get("description") or item.get("error") or "").strip()
        if overview:
            lines.extend(["### 导读", "", overview, ""])
        why_read = str(guide.get("why_read") or "").strip()
        if why_read:
            lines.extend(["### 为什么值得读", "", why_read, ""])
        claims = guide.get("key_claims") if isinstance(guide.get("key_claims"), list) else []
        if claims:
            lines.extend(["### 文章的关键主张", ""])
            for claim in claims:
                if isinstance(claim, dict):
                    text = str(claim.get("claim") or "").strip()
                    support = str(claim.get("support") or "").strip()
                    confidence = str(claim.get("confidence") or "").strip()
                    lines.append(f"- {text}" + (f"〔文内支持度：{confidence}〕" if confidence else ""))
                    if support:
                        lines.append(f"  - 文内依据概括：{support}")
                elif str(claim).strip():
                    lines.append(f"- {str(claim).strip()}")
            lines.append("")
        questions = guide.get("verification_questions") if isinstance(guide.get("verification_questions"), list) else []
        if questions:
            lines.extend(["### 建议核验", ""])
            lines.extend(f"- {str(value).strip()}" for value in questions if str(value).strip())
            lines.append("")
        caveats = guide.get("caveats") if isinstance(guide.get("caveats"), list) else []
        if caveats:
            lines.extend(["### 边界", ""])
            lines.extend(f"- {str(value).strip()}" for value in caveats if str(value).strip())
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    report["fulltext_ready"] = sum(1 for item in report["items"] if item.get("access_scope") == "open_fulltext")
    report["guided"] = sum(1 for item in report["items"] if item.get("status") == "guided")
    report["elapsed_seconds"] = round(time.monotonic() - report["_started"], 2)
    serializable = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(serializable), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="新智元")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    store = GardenStore()
    base_url = store.setting("tracememo_base_url", "http://127.0.0.1:6131")
    client = TraceMemoClient(tracememo_config(base_url), timeout=30)
    accounts = client.official_accounts(args.account)
    matches = accounts.get("items") if isinstance(accounts.get("items"), list) else []
    exact = [
        item for item in matches
        if str(item.get("display_name") or item.get("m_nsNickName") or "").strip() == args.account.strip()
    ]
    if not exact:
        recent = client.recent_chats(200)
        recent_items = recent.get("items") if isinstance(recent.get("items"), list) else []
        exact = [
            {
                **item,
                "display_name": str(item.get("m_nsNickName") or "").strip(),
                "account_id": str(item.get("m_nsUsrName") or "").strip(),
            }
            for item in recent_items
            if bool(item.get("isOfficialAccount"))
            and str(item.get("m_nsNickName") or "").strip() == args.account.strip()
        ]
    if len(exact) == 1:
        matches = exact
    if len(matches) != 1:
        raise RuntimeError(f"公众号匹配数量不是1：{len(matches)}")
    account = matches[0]
    result = client.official_articles(
        str(account.get("account_id") or ""),
        days=args.days,
        contact=account,
    )
    available_articles = list(result.get("articles") or [])
    start_index = max(0, int(args.start) - 1)
    articles = available_articles[start_index : start_index + max(1, args.limit)]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"wechat-{args.account}-latest{len(articles)}-{stamp}.json"
    md_path = json_path.with_suffix(".md")
    report: dict[str, Any] = {
        "suite": "wechat_official_account_latest",
        "account": args.account,
        "requested": len(articles),
        "available_in_window": result.get("count", len(articles)),
        "fulltext_ready": 0,
        "guided": 0,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": 0,
        "items": [],
        "_started": time.monotonic(),
    }

    for index, message in enumerate(articles, 1):
        started = time.monotonic()
        article = message.get("article") if isinstance(message.get("article"), dict) else {}
        title = str(article.get("title") or f"文章{index}")
        description = str(article.get("description") or "")
        url = str(article.get("url") or "")
        item: dict[str, Any] = {
            "rank": index,
            "title": title,
            "description": description,
            "url": url,
            "sent_at": message.get("sent_at"),
            "publisher": article.get("publisher") or args.account,
            "status": "card_only",
            "access_scope": "article_card",
            "body_chars": 0,
        }
        try:
            body_error: Exception | None = None
            body = ""
            for fetch_attempt in range(3):
                try:
                    body = fetch_wechat_article_text(url, timeout=25)
                    break
                except Exception as exc:
                    body_error = exc
                    if fetch_attempt < 2:
                        time.sleep(2.0 * (fetch_attempt + 1))
            if not body:
                raise body_error or RuntimeError("公众号正文读取失败")
            item.update({
                "status": "fulltext_ready",
                "access_scope": "open_fulltext",
                "body_chars": len(body),
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            })
            guide = guide_article(title, description, body)
            item["guide"] = guide
            item["status"] = "guided" if guide.get("generated") else "guide_fallback"
        except Exception as exc:
            item["error"] = str(exc)
            item["guide"] = {
                "overview": description or "微信原文正文未能回源，本轮只保留文章卡片。",
                "why_read": "",
                "key_claims": [],
                "verification_questions": ["在浏览器通过微信验证后重新读取正文。"],
                "caveats": ["本轮没有取得正文，不能据此评价文章论证与事实准确性。"],
                "tags": [],
            }
        item["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["items"].append(item)
        save(report, json_path, md_path)
        print(
            f"PROGRESS {index}/{len(articles)} {item['status']} "
            f"{item['access_scope']} {item['elapsed_seconds']}s",
            flush=True,
        )
    print(f"REPORT_JSON={json_path}", flush=True)
    print(f"REPORT_MD={md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
