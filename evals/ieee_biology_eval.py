from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ROOT
from core.paper_reader import deep_read_paper
from core.storage import GardenStore
from core.web_research import _abstract_from_index


RESULTS_DIR = ROOT / "evals" / "results"
SELECTIONS = [
    {"xplore_id": "11592338", "doi": "10.1109/JBHI.2026.3708743", "field": "脑科学 × fMRI解码"},
    {"xplore_id": "11488043", "doi": "10.1109/JBHI.2026.3685529", "field": "肿瘤病理 × 生物标志物"},
    {"xplore_id": "11586018", "doi": "10.1109/ACCESS.2026.3708197", "field": "脑机接口 × EEG"},
    {"xplore_id": "11084842", "doi": "10.1109/TMI.2025.3589797", "field": "医学影像 × 肿瘤分割"},
    {"xplore_id": "11086512", "doi": "10.1109/JBHI.2025.3589889", "field": "神经康复 × 肌电信号"},
]


def openalex_by_doi(selection: dict[str, str]) -> dict[str, Any]:
    doi = selection["doi"]
    request = Request(
        "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="/"),
        headers={"User-Agent": "KnowledgeGarden/1.0 (local learning assistant)"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or not payload.get("display_name"):
        raise RuntimeError(f"OpenAlex没有返回DOI {doi} 的有效元数据")
    authors = []
    for authorship in payload.get("authorships") or []:
        name = str((authorship.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
        if len(authors) >= 6:
            break
    primary = payload.get("primary_location") or {}
    best_oa = payload.get("best_oa_location") or {}
    source = primary.get("source") or {}
    return {
        **selection,
        "title": str(payload.get("display_name") or "").strip(),
        "url": f"https://ieeexplore.ieee.org/document/{selection['xplore_id']}",
        "year": payload.get("publication_year"),
        "publication_date": payload.get("publication_date"),
        "authors": authors,
        "venue": str(source.get("display_name") or "IEEE").strip(),
        "abstract": _abstract_from_index(payload.get("abstract_inverted_index"), limit=6000),
        "open_access": bool((payload.get("open_access") or {}).get("is_oa")),
        "oa_status": str((payload.get("open_access") or {}).get("oa_status") or "closed"),
        "pdf_url": str(best_oa.get("pdf_url") or "").strip(),
        "repository_url": str(best_oa.get("landing_page_url") or "").strip(),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# IEEE Xplore 生物与生物医学论文导读迁移测试", "",
        f"- 生成时间：{report['generated_at']}",
        f"- 完成：{report['completed']}/{report['requested']}",
        f"- 开放全文：{report['fulltext']}/{report['requested']}",
        f"- 仅摘要：{report['abstract_only']}/{report['requested']}",
        f"- 总耗时：{report['elapsed_seconds']} 秒",
        "- 范围声明：只通过 DOI/OpenAlex 查询合法开放版本；没有正文时不会把摘要导读标成全文。", "",
    ]
    for item in report["items"]:
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
        lines.extend([
            f"## {item['rank']}. {item.get('title') or item['doi']}", "",
            f"- 方向：{item['field']}", f"- IEEE Xplore：{item['xplore_url']}",
            f"- DOI：{item['doi_url']}", f"- 期刊：{item.get('venue') or '未知'}",
            f"- 证据范围：{result.get('scope_label') or '失败'}",
            f"- 开放状态：{item.get('oa_status') or '未知'}",
            f"- 提取字符：{result.get('extracted_chars', 0)}",
            f"- 耗时：{item.get('elapsed_seconds', 0)} 秒", "",
        ])
        if item.get("error"):
            lines.extend(["### 失败原因", "", str(item["error"]), ""])
            continue
        for heading, key in (("研究问题", "problem"), ("创新与价值", "novelty"), ("方法", "method")):
            value = str(analysis.get(key) or "").strip()
            if value:
                lines.extend([f"### {heading}", "", value, ""])
        findings = analysis.get("findings") if isinstance(analysis.get("findings"), list) else []
        if findings:
            lines.extend(["### 主要发现与证据", ""])
            for finding in findings:
                state = "已核验" if finding.get("grounded") else "未通过核验"
                lines.append(f"- {finding.get('claim', '')}〔{state}〕")
                if finding.get("evidence"):
                    lines.append(f"  - 依据：{finding['evidence']}")
            lines.append("")
        limitations = analysis.get("limitations") if isinstance(analysis.get("limitations"), list) else []
        if limitations:
            lines.extend(["### 局限与阅读边界", ""])
            lines.extend(f"- {value}" for value in limitations)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    report["completed"] = sum(1 for item in report["items"] if item.get("result"))
    report["fulltext"] = sum(1 for item in report["items"] if (item.get("result") or {}).get("scope") == "open_fulltext")
    report["abstract_only"] = sum(1 for item in report["items"] if (item.get("result") or {}).get("scope") == "abstract")
    report["elapsed_seconds"] = round(time.monotonic() - report["_started"], 2)
    payload = {key: value for key, value in report.items() if not key.startswith("_")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    selected = SELECTIONS[: max(1, min(5, args.limit))]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"ieee-biology-{len(selected)}-{stamp}.json"
    md_path = json_path.with_suffix(".md")
    report: dict[str, Any] = {
        "suite": "ieee_xplore_biology_guided_reading", "requested": len(selected),
        "completed": 0, "fulltext": 0, "abstract_only": 0,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": 0, "items": [], "_started": time.monotonic(),
    }
    store = GardenStore()
    for rank, selection in enumerate(selected, 1):
        started = time.monotonic()
        item: dict[str, Any] = {
            "rank": rank, "field": selection["field"], "doi": selection["doi"],
            "doi_url": f"https://doi.org/{selection['doi']}",
            "xplore_url": f"https://ieeexplore.ieee.org/document/{selection['xplore_id']}",
        }
        try:
            article = openalex_by_doi(selection)
            item.update({key: article.get(key) for key in ("title", "venue", "publication_date", "open_access", "oa_status", "repository_url")})
            item["result"] = deep_read_paper(store, article, force=True)
        except Exception as exc:
            item["error"] = f"{exc.__class__.__name__}: {exc}"
        item["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["items"].append(item)
        save(report, json_path, md_path)
        scope = (item.get("result") or {}).get("scope", "error")
        print(f"PROGRESS {rank}/{len(selected)} {scope} {item['elapsed_seconds']}s", flush=True)
    print(f"REPORT_JSON={json_path}", flush=True)
    print(f"REPORT_MD={md_path}", flush=True)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
