from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import DB_PATH
from core.retrieval import _textbook_navigation_weight
from core.storage import GardenStore
from evals.adapter import load_cases


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "evals" / "datasets" / "zhili_college_54_v1.jsonl"
REPORT_DIR = ROOT / "evals" / "reports"


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def matched_evidence_groups(text: str, groups: list[list[str]]) -> set[int]:
    corpus = compact_text(text)
    return {
        index for index, alternatives in enumerate(groups)
        if any(compact_text(term) in corpus for term in alternatives if compact_text(term))
    }


def _allowed_book(note: dict[str, Any], patterns: list[str]) -> bool:
    if not patterns:
        return True
    if note.get("kind") in {"textbook", "course"}:
        corpus = compact_text(str(note.get("title", "")))
    else:
        corpus = compact_text(f"{note.get('title', '')}\n{note.get('content', '')}")
    return any(compact_text(pattern) in corpus for pattern in patterns)


def classify_case(case: dict[str, Any], notes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [
        [str(term).strip() for term in group if str(term).strip()]
        for group in case.get("evidence_terms", []) if isinstance(group, list)
    ]
    patterns = [str(item) for item in case.get("book_patterns", [])]
    candidates: list[tuple[tuple[float, int, int], dict[str, Any], set[int]]] = []
    for note in notes:
        if note.get("kind") not in {"textbook", "course", "concept", "knowledge"}:
            continue
        if not _allowed_book(note, patterns):
            continue
        text = f"{note.get('title', '')}\n{note.get('content', '')}"
        matches = matched_evidence_groups(text, groups)
        if not matches or 0 not in matches:
            continue
        textbook = int(note.get("kind") in {"textbook", "course"})
        navigation = _textbook_navigation_weight(note) if textbook else 1.0
        matches_required = min(2, len(groups))
        score = (
            round(len(matches) * navigation + textbook * 0.3, 4),
            int(len(matches) >= matches_required),
            min(len(str(note.get("content", ""))), 3000),
        )
        candidates.append((score, note, matches))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[dict[str, Any], set[int]]] = []
    covered: set[int] = set()
    for _, note, matches in candidates:
        if len(chosen) >= 8:
            break
        if not chosen or matches - covered or len(chosen) < 3:
            chosen.append((note, matches))
            covered.update(matches)
    all_groups = set(range(len(groups)))
    textbook_chosen = any(note.get("kind") in {"textbook", "course"} for note, _ in chosen)
    strong_page = any(len(matches) >= min(2, len(groups)) for _, matches in chosen)
    if all_groups and covered == all_groups and strong_page:
        coverage = "textbook_grounded" if textbook_chosen else "local_note_grounded"
    elif chosen:
        coverage = "partial_local_evidence"
    else:
        coverage = "no_local_evidence"
    matches_required = min(2, len(groups))
    reference_titles = list(dict.fromkeys(
        str(note["title"])
        for _, note, matches in candidates
        if len(matches) >= matches_required
        and _textbook_navigation_weight(note) >= 0.5
    ))
    if not reference_titles:
        reference_titles = [str(note["title"]) for note, _ in chosen]
    return {
        **case,
        "category": str(case.get("section", "")),
        "reference_titles": reference_titles,
        "should_abstain": coverage == "no_local_evidence",
        "coverage_status": coverage,
        "evidence_group_coverage": round(len(covered) / max(1, len(all_groups)), 4),
        "covered_evidence_groups": sorted(covered),
        "requires_online_completion": bool(
            case.get("freshness_required") or coverage not in {"textbook_grounded", "local_note_grounded"}
        ),
        "gold_source_kinds": [str(note.get("kind", "")) for note, _ in chosen],
    }


def materialize_cases(cases: list[dict[str, Any]], notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [classify_case(case, notes) for case in cases]


def write_coverage_report(cases: list[dict[str, Any]], *, label: str = "zhili-54-coverage") -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_path = REPORT_DIR / f"{label}-{stamp}.jsonl"
    report_path = REPORT_DIR / f"{label}-{stamp}.md"
    dataset_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8",
    )
    status_names = {
        "textbook_grounded": "已有完整教材依据",
        "local_note_grounded": "已有本地知识笔记依据",
        "partial_local_evidence": "本地只有部分依据，需要补充",
        "no_local_evidence": "本地暂时没有直接依据",
    }
    counts = Counter(str(case.get("coverage_status")) for case in cases)
    lines = [
        "# 致理书院 54 题：本地证据覆盖与三层评测分流", "",
        f"- 题目总数：{len(cases)}",
        *[f"- {status_names.get(status, status)}：{count} 题" for status, count in counts.items()],
        f"- 需要联网补足或前沿验证：{sum(bool(case.get('requires_online_completion')) for case in cases)} 题",
        "", "说明：这是教材当前时刻的快照。扫描版 OCR 在后台持续写入后，可再次运行更新覆盖情况。", "",
    ]
    for index, case in enumerate(cases, 1):
        lines.extend([
            f"## {index}. {case['id']} · {case.get('discipline', '')}", "",
            f"- 问题：{case['question']}",
            f"- 板块：{case.get('section', '')}",
            f"- 本地证据：{status_names.get(str(case.get('coverage_status')), '')}",
            f"- 核心要点覆盖：{float(case.get('evidence_group_coverage', 0)):.0%}",
            f"- 是否需要联网补足：{'是' if case.get('requires_online_completion') else '否'}",
            f"- 已识别的相关教材/笔记页：{len(case.get('reference_titles', []))}",
            *[f"- 候选教材/笔记：{title}" for title in case.get("reference_titles", [])[:4]],
            f"- 参考答案：{case.get('reference', '')}", "",
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return dataset_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare textbook-grounded three-layer evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--database", type=Path, default=DB_PATH)
    args = parser.parse_args()
    store = GardenStore(args.database)
    cases = materialize_cases(load_cases(args.dataset), store.list_notes(limit=50_000))
    dataset, report = write_coverage_report(cases)
    print(json.dumps({
        "cases": len(cases), "coverage": dict(Counter(row["coverage_status"] for row in cases)),
        "requires_online_completion": sum(bool(row["requires_online_completion"]) for row in cases),
        "materialized_dataset": str(dataset), "report": str(report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
