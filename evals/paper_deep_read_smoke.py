from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import ROOT
from core.paper_reader import _connect_local_knowledge, deep_read_paper
from core.storage import GardenStore


ARTICLES: list[dict[str, Any]] = [
    {
        "field": "机器学习 × 应用数学",
        "title": "Principled approaches for extending neural architectures to function spaces for operator learning",
        "url": "https://www.nature.com/articles/s42256-026-01267-z",
        "pdf_url": "https://www.nature.com/articles/s42256-026-01267-z.pdf",
        "venue": "Nature Machine Intelligence",
        "year": "2026",
        "abstract": (
            "Deep learning commonly maps finite-dimensional representations, while many scientific problems "
            "governed by partial differential equations live on infinite-dimensional function spaces. The paper "
            "distils principles for neural operators and proposes a recipe for converting popular neural "
            "architectures into discretization-agnostic operators with minimal modifications."
        ),
    },
    {
        "field": "材料科学 × 机器学习",
        "title": "DFT-based machine-learning for the rational design of magnetocaloric high-entropy alloys",
        "url": "https://www.nature.com/articles/s41467-026-77123-w",
        "pdf_url": "https://www.nature.com/articles/s41467-026-77123-w_reference.pdf",
        "venue": "Nature Communications",
        "year": "2026",
        "abstract": (
            "The work combines a minimal-supercell principle with physics-informed machine learning to map the "
            "large composition space of magnetocaloric high-entropy alloys. It reports interpretable electronic "
            "and magnetic descriptors, a stability-performance trade-off, and a hierarchical tuning strategy."
        ),
    },
    {
        "field": "量子物理",
        "title": "Subjective nature of path information in quantum mechanics",
        "url": "https://www.nature.com/articles/s41467-026-69034-7",
        "pdf_url": "https://www.nature.com/articles/s41467-026-69034-7.pdf",
        "venue": "Nature Communications",
        "year": "2026",
        "abstract": (
            "The experiment studies complementarity between path distinguishability and interference visibility "
            "using three sources that emit into identical modes. Different groupings show that a definite physical "
            "origin cannot always be assigned even when full path information is available."
        ),
    },
    {
        "field": "计算生物物理 × 生成模型",
        "title": "Accurate predictions of disordered protein ensembles with STARLING",
        "url": "https://www.nature.com/articles/s41586-026-10141-2",
        "pdf_url": "https://www.nature.com/articles/s41586-026-10141-2.pdf",
        "venue": "Nature",
        "year": "2026",
        "abstract": (
            "STARLING combines physics-based force fields and multimodal generative deep learning to rapidly "
            "generate ensembles and representations for intrinsically disordered protein regions. It supports "
            "environmental conditioning, experimental reweighting, similarity search and sequence design."
        ),
    },
]


def _row(result: dict[str, Any], field: str, elapsed: float) -> dict[str, Any]:
    analysis = result.get("analysis") or {}
    findings = analysis.get("findings") or []
    grounded = [item for item in findings if item.get("grounded")]
    pipeline = result.get("reading_pipeline") or {}
    return {
        "field": field,
        "title": result.get("title"),
        "scope": result.get("scope"),
        "scope_label": result.get("scope_label"),
        "source_note": result.get("source_note"),
        "fulltext_error": result.get("fulltext_error"),
        "extracted_chars": result.get("extracted_chars"),
        "confidence": analysis.get("confidence"),
        "findings": len(findings),
        "grounded_findings": len(grounded),
        "local_candidates": pipeline.get("local_candidate_count", 0),
        "accepted_connections": pipeline.get("accepted_connection_count", 0),
        "elapsed_seconds": round(elapsed, 2),
        "result": result,
    }


def _markdown(rows: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# 多领域论文深读冒烟测试",
        "",
        f"生成时间：{generated_at}",
        "",
        "| 领域 | 证据范围 | 置信度 | 落地发现 | 本地连接 | 耗时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['field']} | {row['scope_label']} | {float(row['confidence'] or 0):.0%} | "
            f"{row['grounded_findings']}/{row['findings']} | {row['accepted_connections']}/"
            f"{row['local_candidates']} | {row['elapsed_seconds']}s |"
        )
    for row in rows:
        result = row["result"]
        analysis = result.get("analysis") or {}
        lines.extend([
            "", f"## {row['field']}：{row['title']}", "",
            f"- 范围：{row['scope_label']}；{result.get('source_note', '')}",
            f"- 研究问题：{analysis.get('problem', '')}",
            f"- 创新：{analysis.get('novelty', '')}",
            f"- 方法：{analysis.get('method', '')}",
            f"- 本地连接审计：{analysis.get('local_connection_note', '')}",
        ])
        if result.get("fulltext_error"):
            lines.append(f"- 全文读取回退原因：{result['fulltext_error']}")
        for finding in analysis.get("findings") or []:
            state = "已核验" if finding.get("grounded") else "未通过核验"
            lines.append(f"- 发现（{state}）：{finding.get('claim', '')}")
        for connection in analysis.get("local_connections") or []:
            learned = connection.get("mastery")
            learned_text = f"；掌握证据={learned.get('stage')}" if learned else "；仅本地资料命中"
            lines.append(
                f"- 本地连接：{connection.get('title')} → {connection.get('bridge')}"
                f"（{float(connection.get('confidence') or 0):.0%}{learned_text}）"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(ARTICLES))
    parser.add_argument("--reconnect-only", action="store_true")
    args = parser.parse_args()
    store = GardenStore()
    report_dir = Path(ROOT) / "evals" / "reports"
    json_path = report_dir / "paper_deep_read_multidomain.json"
    md_path = report_dir / "paper_deep_read_multidomain.md"
    if args.reconnect_only:
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        rows = existing.get("rows") or []
        for row in rows:
            result = row.get("result") or {}
            analysis = result.get("analysis") or {}
            bridge = _connect_local_knowledge(store, str(row.get("title") or ""), analysis)
            analysis["local_connections"] = bridge["connections"]
            analysis["local_connection_note"] = bridge["note"]
            pipeline = result.setdefault("reading_pipeline", {})
            pipeline["local_candidate_count"] = bridge["candidate_count"]
            pipeline["accepted_connection_count"] = len(bridge["connections"])
            row["local_candidates"] = bridge["candidate_count"]
            row["accepted_connections"] = len(bridge["connections"])
        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        json_path.write_text(
            json.dumps({"generated_at": generated_at, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(_markdown(rows, generated_at), encoding="utf-8")
        print(json_path)
        print(md_path)
        return
    rows: list[dict[str, Any]] = []
    for article in ARTICLES[: max(1, min(len(ARTICLES), args.limit))]:
        started = time.perf_counter()
        try:
            result = deep_read_paper(store, article, force=True)
            rows.append(_row(result, str(article["field"]), time.perf_counter() - started))
            print(
                f"[{article['field']}] {result['scope_label']} "
                f"connections={result['reading_pipeline']['accepted_connection_count']} "
                f"elapsed={rows[-1]['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            rows.append({
                "field": article["field"], "title": article["title"], "scope_label": "失败",
                "confidence": 0, "findings": 0, "grounded_findings": 0,
                "local_candidates": 0, "accepted_connections": 0,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "error": f"{exc.__class__.__name__}: {exc}", "result": {},
            })
            print(f"[{article['field']}] ERROR {exc.__class__.__name__}: {exc}", flush=True)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"generated_at": generated_at, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(rows, generated_at), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
