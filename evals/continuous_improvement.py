from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.config import DATA_DIR, ROOT
from core.multiuser import AuthRegistry
from core.storage import utc_now


REPORTS = ROOT / "evals" / "reports"
RUBRIC = {
    "version": "continuous-rubric-v1",
    "dimensions": [
        "correctness", "answers_the_question", "derivation_rigor",
        "evidence_boundary", "teaching_fit", "naturalness",
    ],
    "failure_types": [
        "router", "evidence", "generation", "false_refusal", "teaching_style", "timeout", "other",
    ],
}
RUBRIC_HASH = hashlib.sha256(
    json.dumps(RUBRIC, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()


def deterministic_review(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    refusal_markers = ("这次先不补写答案", "证据不足", "无法回答", "不能回答", "没有足够证据")
    false_refusal = bool(question and any(marker in answer for marker in refusal_markers))
    empty = len(answer) < 12
    leaked_secret = bool(re.search(r"(?i)\b(?:sk|ak)-[A-Za-z0-9_-]{12,}\b", answer))
    passed = not (empty or leaked_secret or false_refusal)
    issues = []
    if empty:
        issues.append("回答为空或过短")
    if leaked_secret:
        issues.append("回答疑似包含未脱敏密钥")
    if false_refusal:
        issues.append("普通用户问题疑似被错误拒答")
    return {
        "judge": "deterministic-v1", "verdict": "pass" if passed else "fail",
        "score": 5.0 if passed else 1.0, "issues": issues,
        "failure_type": "false_refusal" if false_refusal else ("generation" if empty else "other"),
        "rubric_hash": RUBRIC_HASH,
    }


def model_judge(row: dict[str, Any], *, prefix: str, label: str) -> dict[str, Any] | None:
    key = os.getenv(f"{prefix}_API_KEY", "").strip()
    if not key:
        return None
    from openai import OpenAI

    base_url = os.getenv(f"{prefix}_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv(f"{prefix}_MODEL", "").strip()
    if not model:
        raise RuntimeError(f"{prefix}_MODEL 未配置")
    payload = {
        "question": row["question"], "answer": row["answer"],
        "explicit_user_feedback": row.get("explicit_feedback") or "",
        "rubric": RUBRIC, "rubric_hash": RUBRIC_HASH,
    }
    prompt = (
        "你是与回答模型隔离的盲审裁判。不得补写答案，不得猜测用户身份。"
        "按冻结 rubric 独立评分，并只输出 JSON："
        "verdict(pass|warn|fail), score(1-5), failure_type, strengths, issues, suggestion,"
        "dimensions{rubric 中每个维度为1-5}。若证据不足以判定事实正确性，应标 warn 而不是臆测。\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    response = OpenAI(api_key=key, base_url=base_url, timeout=120).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content or "{}")
    data.update({"judge": label, "model": model, "rubric_hash": RUBRIC_HASH})
    return data


def save_review(registry: AuthRegistry, candidate_id: str, review: dict[str, Any]) -> None:
    judge_name = str(review.get("judge") or "unknown")
    judge_version = str(review.get("model") or review.get("judge") or "v1")
    with registry.connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO judge_reviews(
                   review_id,candidate_id,judge_name,judge_version,verdict,score,findings_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                f"review-{uuid4().hex}", candidate_id, judge_name, judge_version,
                str(review.get("verdict") or "warn"),
                float(review["score"]) if review.get("score") is not None else None,
                json.dumps(review, ensure_ascii=False), utc_now(),
            ),
        )


def run(limit: int) -> dict[str, Any]:
    registry = AuthRegistry()
    with registry.connect() as conn:
        rows = [dict(row) for row in conn.execute(
            """SELECT * FROM interaction_candidates
               WHERE dataset_split='development' AND status IN ('pending','retry')
               ORDER BY created_at LIMIT ?""",
            (max(1, min(limit, 500)),),
        )]
    results = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] judging {row['candidate_id']}")
        reviews = [deterministic_review(row)]
        for prefix, label in (("JUDGE", "primary-independent"), ("SECONDARY_JUDGE", "secondary-independent")):
            try:
                review = model_judge(row, prefix=prefix, label=label)
                if review:
                    reviews.append(review)
            except Exception as exc:
                reviews.append({"judge": label, "verdict": "error", "issues": [str(exc)]})
        for review in reviews:
            save_review(registry, row["candidate_id"], review)
        model_verdicts = [item.get("verdict") for item in reviews if item.get("judge") != "deterministic-v1" and item.get("verdict") != "error"]
        deterministic_failed = reviews[0]["verdict"] == "fail"
        judges_agree = len(model_verdicts) >= 2 and len(set(model_verdicts)) == 1
        if deterministic_failed or any(value == "fail" for value in model_verdicts):
            status = "actionable_failure" if judges_agree else "human_review"
        elif judges_agree:
            status = "reviewed"
        else:
            status = "human_review"
        with registry.connect() as conn:
            conn.execute(
                "UPDATE interaction_candidates SET status=? WHERE candidate_id=?",
                (status, row["candidate_id"]),
            )
        results.append({
            "candidate_id": row["candidate_id"], "status": status,
            "question": row["question"], "reviews": reviews,
        })

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": f"continuous-{stamp}", "rubric": RUBRIC,
        "rubric_hash": RUBRIC_HASH, "processed": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "safety": {
            "holdout_read": False, "production_modified": False,
            "requires_human_approval_before_release": True,
        },
        "results": results,
    }
    json_path = REPORTS / f"continuous-improvement-{stamp}.json"
    md_path = REPORTS / f"continuous-improvement-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 持续改进裁判报告", "",
        f"- 已处理：{len(results)}", f"- Rubric：`{RUBRIC_HASH}`",
        "- 固定保留集：未读取", "- 生产代码/提示词：未自动修改", "",
        "## 需要处理的案例", "",
    ]
    for item in results:
        if item["status"] in {"actionable_failure", "human_review"}:
            lines.extend([
                f"### {item['candidate_id']} · {item['status']}", "",
                item["question"], "",
                *[f"- {review.get('judge')}：{review.get('verdict')} · {review.get('issues') or review.get('suggestion') or ''}" for review in item["reviews"]],
                "",
            ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    report["reports"] = [str(json_path), str(md_path)]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge opted-in development interactions without touching the sealed holdout.")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
