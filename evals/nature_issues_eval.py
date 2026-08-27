from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ROOT
from core.llm import LLMError, chat_json
from core.storage import GardenStore


RESULTS_DIR = ROOT / "evals" / "results"
NATURE_ROOT = "https://www.nature.com"
RESEARCH_TYPES = {"Article", "Review Article"}


def clean_text(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", unescape(" ".join(parts))).strip()


class NatureIssueParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_section = ""
        self.issue_title = ""
        self.items: list[dict[str, Any]] = []
        self.h1_depth = 0
        self.h2_depth = 0
        self.heading_parts: list[str] = []
        self.in_article = False
        self.card: dict[str, Any] = {}
        self.title_depth = 0
        self.description_depth = 0
        self.type_depth = 0
        self.title_parts: list[str] = []
        self.description_parts: list[str] = []
        self.type_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key: value or "" for key, value in attrs_list}
        if self.in_article:
            if self.title_depth:
                self.title_depth += 1
            elif tag == "h3":
                self.title_depth = 1
                self.title_parts = []
            if self.description_depth:
                self.description_depth += 1
            elif attrs.get("itemprop") == "description":
                self.description_depth = 1
                self.description_parts = []
            classes = set(attrs.get("class", "").split())
            if self.type_depth:
                self.type_depth += 1
            elif "c-meta__type" in classes:
                self.type_depth = 1
                self.type_parts = []
            if tag == "a" and self.title_depth and attrs.get("href"):
                self.card.setdefault("url", urljoin(NATURE_ROOT, attrs["href"]))
            if tag == "time" and attrs.get("datetime"):
                self.card["published"] = attrs["datetime"]
            return
        if tag == "h1":
            self.h1_depth = 1
            self.heading_parts = []
        elif tag == "h2":
            self.h2_depth = 1
            self.heading_parts = []
        elif tag == "article":
            self.in_article = True
            self.card = {"section": self.current_section}
            self.title_parts = []
            self.description_parts = []
            self.type_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_article:
            if self.title_depth:
                self.title_depth -= 1
                if self.title_depth == 0:
                    self.card["title"] = clean_text(self.title_parts)
            if self.description_depth:
                self.description_depth -= 1
                if self.description_depth == 0:
                    self.card["summary"] = clean_text(self.description_parts)
            if self.type_depth:
                self.type_depth -= 1
                if self.type_depth == 0:
                    self.card["content_type"] = clean_text(self.type_parts)
            if tag == "article":
                title = str(self.card.get("title") or "").strip()
                url = str(self.card.get("url") or "").strip()
                if title and url:
                    self.items.append(dict(self.card))
                self.in_article = False
                self.card = {}
                self.title_depth = 0
                self.description_depth = 0
                self.type_depth = 0
            return
        if tag == "h1" and self.h1_depth:
            self.h1_depth = 0
            self.issue_title = clean_text(self.heading_parts)
        elif tag == "h2" and self.h2_depth:
            self.h2_depth = 0
            section = clean_text(self.heading_parts)
            if section and section != "Table of Contents":
                self.current_section = section

    def handle_data(self, data: str) -> None:
        if self.in_article:
            if self.title_depth:
                self.title_parts.append(data)
            if self.description_depth:
                self.description_parts.append(data)
            if self.type_depth:
                self.type_parts.append(data)
            return
        if self.h1_depth or self.h2_depth:
            self.heading_parts.append(data)


def fetch_issue(issue: str) -> dict[str, Any]:
    url = f"https://www.nature.com/nature/volumes/656/issues/{issue}"
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KnowledgeGarden/1.0",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urlopen(request, timeout=25) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Nature issue page is too large")
    parser = NatureIssueParser()
    parser.feed(data.decode("utf-8", errors="replace"))
    items = parser.items
    section_counts = dict(Counter(str(item.get("section") or "Unknown") for item in items))
    type_counts = dict(Counter(str(item.get("content_type") or "Unknown") for item in items))
    research = [
        item for item in items
        if item.get("section") == "Research"
        and item.get("content_type") in RESEARCH_TYPES
    ]
    context = [
        item for item in items
        if item.get("section") == "Research"
        and item.get("content_type") == "News & Views"
    ]
    return {
        "issue": issue,
        "issue_title": parser.issue_title,
        "url": url,
        "toc_count": len(items),
        "section_counts": section_counts,
        "type_counts": type_counts,
        "research_count": len(research),
        "research_context_count": len(context),
        "items": items,
        "research_items": research,
        "research_context": context,
    }


def batch_guides(items: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    fallback = []
    for index, item in enumerate(items):
        fallback.append({
            "index": index,
            "research_question": "",
            "main_result": item.get("summary") or "",
            "why_it_matters": "",
            "method_signal": "官方目录摘要未提供足够方法细节。",
            "reading_priority": 3,
            "reading_route": "先核对摘要和图表，再阅读方法与限制。",
            "verification_limits": "本导读只依据 Nature 期刊目录摘要，不等同于阅读全文或独立复核。",
            "generated": False,
        })
    payload = [
        {
            "index": index,
            "title": item.get("title"),
            "content_type": item.get("content_type"),
            "summary": item.get("summary"),
        }
        for index, item in enumerate(items)
    ]
    try:
        result = chat_json(
            "你是基础学科前沿雷达编辑。只能依据Nature官方目录摘要写中文导读，不得补写摘要没有的方法、数据或因果。"
            "reading_priority要结合用户方向，但不能为了迎合而改变论文意义。只返回JSON。",
            f"用户当前方向：{profile or '未明确，按基础科学普适价值排序'}\n"
            f"论文卡片：{json.dumps(payload, ensure_ascii=False)}\n"
            "返回 items，逐项保留index，并给出research_question、main_result、why_it_matters、"
            "method_signal、reading_priority（1~5整数）、reading_route、verification_limits。"
            "verification_limits必须说明这是目录摘要级导读。",
            timeout=90,
            max_retries=1,
        )
    except LLMError:
        result = None
    values = result.get("items") if isinstance(result, dict) and isinstance(result.get("items"), list) else []
    by_index = {
        int(value.get("index")): value
        for value in values
        if isinstance(value, dict) and str(value.get("index", "")).isdigit()
    }
    guided = []
    for index, base in enumerate(fallback):
        value = by_index.get(index)
        if not isinstance(value, dict):
            guided.append(base)
            continue
        merged = {key: value.get(key, default) for key, default in base.items()}
        merged["index"] = index
        merged["generated"] = True
        try:
            merged["reading_priority"] = max(1, min(5, int(merged["reading_priority"])))
        except (TypeError, ValueError):
            merged["reading_priority"] = 3
        guided.append(merged)
    return guided


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Nature 最近两期前沿研究导读测试",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 用户方向：{report['profile'] or '未明确'}",
        f"- 研究论文/综述：{report['research_total']}",
        f"- 模型导读成功：{report['guided_total']}/{report['research_total']}",
        f"- 总耗时：{report['elapsed_seconds']} 秒",
        "- 证据范围：Nature官方期刊目录中的标题与摘要；没有取得全文时不评价完整方法和统计可靠性。",
        "",
    ]
    for issue in report["issues"]:
        lines.extend([
            f"## {issue['issue_title'] or issue['issue']}",
            "",
            f"- 官方目录：{issue['url']}",
            f"- 全目录条目：{issue['toc_count']}",
            f"- Article/Review Article：{issue['research_count']}",
            f"- News & Views：{issue['research_context_count']}（单列，不计入原始研究）",
            "",
            "### 栏目计数",
            "",
        ])
        for section, count in issue["section_counts"].items():
            lines.append(f"- {section}: {count}")
        lines.append("")
        for item in issue["research_items"]:
            guide = item.get("guide") if isinstance(item.get("guide"), dict) else {}
            priority = guide.get("reading_priority", 3)
            lines.extend([
                f"### [{priority}/5] {item['title']}",
                "",
                f"- 类型：{item.get('content_type') or '未知'}",
                f"- 日期：{item.get('published') or '未知'}",
                f"- 链接：{item['url']}",
                f"- 访问范围：issue_summary",
                "",
            ])
            fields = [
                ("研究问题", "research_question"),
                ("主要结果", "main_result"),
                ("为什么重要", "why_it_matters"),
                ("方法线索", "method_signal"),
                ("阅读路线", "reading_route"),
                ("证据边界", "verification_limits"),
            ]
            for label, key in fields:
                value = str(guide.get(key) or "").strip()
                if value:
                    lines.extend([f"**{label}**", "", value, ""])
        if issue["research_context"]:
            lines.extend(["### News & Views（研究解读，不是原始论文）", ""])
            for item in issue["research_context"]:
                lines.append(f"- [{item['title']}]({item['url']})：{item.get('summary') or ''}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    report["research_total"] = sum(issue["research_count"] for issue in report["issues"])
    report["guided_total"] = sum(
        1 for issue in report["issues"] for item in issue["research_items"]
        if isinstance(item.get("guide"), dict) and item["guide"].get("generated")
    )
    report["elapsed_seconds"] = round(time.monotonic() - report["_started"], 2)
    serializable = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(serializable), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", nargs="+", default=["8129", "8128"])
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    store = GardenStore()
    focus = str(store.setting("frontier_focus", "") or "").strip()
    interests = store.setting("interests", [])
    interest_text = "、".join(str(value) for value in interests if str(value).strip()) if isinstance(interests, list) else str(interests or "")
    profile = "；".join(value for value in (focus, interest_text) if value)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"nature-latest-two-{'-'.join(args.issues)}-{stamp}.json"
    md_path = json_path.with_suffix(".md")
    report: dict[str, Any] = {
        "suite": "nature_latest_two_issues",
        "issues_requested": args.issues,
        "profile": profile,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "research_total": 0,
        "guided_total": 0,
        "elapsed_seconds": 0,
        "issues": [],
        "_started": time.monotonic(),
    }

    for issue_number in args.issues:
        issue = fetch_issue(issue_number)
        report["issues"].append(issue)
        save(report, json_path, md_path)
        print(
            f"ISSUE {issue_number} toc={issue['toc_count']} "
            f"research={issue['research_count']} context={issue['research_context_count']}",
            flush=True,
        )
        research = issue["research_items"]
        size = max(1, int(args.batch_size))
        for start in range(0, len(research), size):
            chunk = research[start : start + size]
            guides = batch_guides(chunk, profile)
            for item, guide in zip(chunk, guides):
                item["guide"] = guide
            save(report, json_path, md_path)
            print(
                f"PROGRESS issue={issue_number} {min(start + size, len(research))}/{len(research)} "
                f"guided={sum(1 for guide in guides if guide.get('generated'))}",
                flush=True,
            )
    print(f"REPORT_JSON={json_path}", flush=True)
    print(f"REPORT_MD={md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
