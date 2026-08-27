from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import DB_PATH, TEMP_DIR, llm_config
from core.context_builder import ContextBuilder
from core.gardener_graph import run_gardener_graph
from core.learning_memory import LearningMemoryService
from core.llm import LLMError, chat_json
from core.storage import GardenStore


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "evals" / "reports"
PREFERENCE = "先用几何直觉建立图景，再给代数定义和推导，并配一个具体例子。"
PERSONA = (
    "你是一名有高等数学基础、正在学习基础理科的本科生。你稳定偏好先用空间或几何直觉建立图景，"
    "再给严格定义和逐步推导，最后用一个具体例子检验；你不喜欢只堆公式、先抛抽象术语或省略中间步骤。"
)

TURNS = [
    ("PREF-DEFINE-1", "define-session", "请解释矩阵的秩。"),
    ("PREF-DEFINE-2", "define-session", "什么是特征值？"),
    ("PREF-DEFINE-3", "define-session", "什么是线性变换的核与像？"),
    ("PREF-MECHANISM-1", "mechanism-session", "为什么特征向量在线性变换后方向不变？"),
    ("PREF-MECHANISM-2", "mechanism-session", "为什么矩阵的秩等于其列空间的维数？"),
    ("PREF-MECHANISM-3", "mechanism-session", "为什么可逆矩阵不能把非零向量压到零向量？"),
    ("PREF-COMPARE-1", "compare-session", "矩阵的秩和行列式有什么区别？"),
    ("PREF-COMPARE-2", "compare-session", "特征值与奇异值有什么区别？"),
    ("PREF-COMPARE-3", "compare-session", "可逆与可对角化有什么区别？"),
]


def _reset_memory(store: GardenStore) -> None:
    """Remove copied learner state while preserving textbooks and application settings."""
    tables = (
        "memory_claim_evidence", "memory_claims", "event_concepts", "learning_events",
        "concept_mastery", "session_messages", "sessions",
    )
    with store.connect() as conn:
        for table in tables:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
            ).fetchone()
            if exists:
                conn.execute(f"DELETE FROM {table}")
    store.set_setting("teaching_preferences", [])
    store.set_setting("memory.last_reflection_at", "")


def _observable_preference_gaps(answer: str) -> list[str]:
    """Block judge false positives when the answer plainly did not execute the preference."""
    gaps: list[str] = []
    if re.search(r"先不补写答案|证据不足|没有取得足够|无法回答", answer):
        gaps.append("回答没有实际讲解知识内容")
    if not re.search(r"几何|空间|直觉|图景|直观|方向|伸缩|拉伸|压缩", answer):
        gaps.append("缺少几何或空间直觉")
    if not re.search(r"定义|是指|称为|意味着", answer):
        gaps.append("缺少定义或概念澄清")
    if not re.search(r"由此|因此|所以|于是|得到|推出|⇒|→|=", answer):
        gaps.append("缺少推导或中间关系")
    if not re.search(r"例如|举例|比如|例子|具体来看", answer):
        gaps.append("缺少具体例子")
    return gaps


def _simulate_user(answer: str, personalization: dict[str, Any]) -> tuple[dict[str, Any], str]:
    del personalization  # The learner judges the delivered answer, never hidden plan metadata.
    local_gaps = _observable_preference_gaps(answer)
    system = (
        f"{PERSONA}\n你现在只扮演学习者，不回答原知识问题。判断给出的讲解方式是否适合你。"
        "只能根据答案正文判断，不能根据系统声称采用了什么计划来加分。若答案拒绝讲解、只有证据缺口，"
        "或没有实际完成‘几何/空间直觉→定义与推导→具体例子’，helpful 必须为 false。"
        "只输出JSON：helpful（布尔）、reason（中文）、preference_instruction、learning_advice、"
        "observed_fit（字符串数组）。preference_instruction 必须原样输出："
        f"{PREFERENCE}"
    )
    user = json.dumps({"answer_to_judge": answer}, ensure_ascii=False)
    try:
        result = chat_json(system, user, timeout=45, max_retries=1)
        if isinstance(result, dict) and isinstance(result.get("helpful"), bool):
            if result["helpful"] and local_gaps:
                result["helpful"] = False
                result["reason"] = "本地可观察检查否决模型假阳性：" + "；".join(local_gaps)
            result["preference_instruction"] = PREFERENCE
            result["observable_gaps"] = local_gaps
            return result, f"glm:{llm_config().model}"
    except LLMError as exc:
        error = f"{type(exc).__name__}: {str(exc)[:180]}"
    else:
        error = "GLM未返回有效布尔反馈"
    return {
        "helpful": not local_gaps,
        "reason": (
            "GLM模拟失败，使用可审计的完整偏好检查。"
            + ("缺口：" + "；".join(local_gaps) if local_gaps else "全部可观察要求均满足。")
        ),
        "preference_instruction": PREFERENCE,
        "learning_advice": "保留稳定偏好并在下一轮检查是否被显式应用。",
        "observed_fit": [],
        "observable_gaps": local_gaps,
        "error": error,
    }, "deterministic-fallback"


def _run_turn(
    store: GardenStore,
    memory: LearningMemoryService,
    question: str,
    session_id: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    turn = memory.begin_turn(question, session_id)
    context = ContextBuilder(store).build(
        question,
        history,
        session_id=turn["session_id"],
        request_id=turn["request_id"],
        message_id=turn["message_id"],
    )
    started = time.perf_counter()
    result = run_gardener_graph(store, context, include_evaluation_context=True)
    memory_result = memory.complete_turn(context, result)
    result.pop("evaluation_context", None)
    return {
        "result": result,
        "memory_result": memory_result,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _claims_snapshot(store: GardenStore) -> list[dict[str, Any]]:
    with store.connect() as conn:
        rows = conn.execute(
            """SELECT claim_id,layer,dimension,scope_type,scope_key,claim_text,
                      confidence,status,source_kind
               FROM memory_claims
               WHERE dimension='teaching_preference'
               ORDER BY layer,scope_key,created_at"""
        ).fetchall()
    return [dict(row) for row in rows]


def run(database: Path) -> dict[str, Any]:
    histories: dict[str, list[dict[str, str]]] = {}
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="garden-personalization-eval-", dir=TEMP_DIR,
    ) as folder:
        target = Path(folder) / "garden-eval.db"
        shutil.copy2(database, target)
        store = GardenStore(target)
        _reset_memory(store)
        memory = LearningMemoryService(store)
        for index, (case_id, session_id, question) in enumerate(TURNS, 1):
            history = histories.setdefault(session_id, [])
            turn = _run_turn(store, memory, question, session_id, history)
            result = turn["result"]
            simulation, provider = _simulate_user(
                str(result.get("answer") or ""), dict(result.get("personalization") or {}),
            )
            feedback = memory.record_personalization_feedback(
                request_id=str(result["request_id"]),
                helpful=bool(simulation["helpful"]),
                feedback_note=str(simulation["preference_instruction"]),
            )
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(result.get("answer") or "")},
            ])
            plan = dict(result.get("personalization") or {})
            rows.append({
                "id": case_id,
                "turn": index,
                "session_id": session_id,
                "question": question,
                "answer": result.get("answer", ""),
                "latency_ms": turn["latency_ms"],
                "evidence_layer": result.get("evidence_layer", "none"),
                "personalization": plan,
                "personalization_applied": plan.get("status") == "applied",
                "applied_claim_ids": plan.get("applied_claim_ids", []),
                "simulated_user": simulation,
                "simulator_provider": provider,
                "feedback_write": feedback,
                "memory_claims_after_turn": _claims_snapshot(store),
            })
            print(
                f"[{index}/{len(TURNS)} {case_id}] status={plan.get('status')} "
                f"applied={len(plan.get('applied_claim_ids', []))} "
                f"helpful={simulation.get('helpful')} provider={provider}",
                flush=True,
            )
        reflection = memory.reflect(force=True, min_events=0)
        l3_graph = memory.l3_profile_graph()
        final_claims = _claims_snapshot(store)

    adapted_rows = [row for row in rows if row["turn"] not in {1, 4, 7}]
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    p90_index = max(0, min(len(latencies) - 1, (9 * len(latencies) + 9) // 10 - 1))
    applied_rows = [row for row in adapted_rows if row["personalization_applied"]]
    summary = {
        "turns": len(rows),
        "glm_simulated_feedback_turns": sum(row["simulator_provider"].startswith("glm:") for row in rows),
        "baseline_turns": 3,
        "post_feedback_turns": len(adapted_rows),
        "post_feedback_personalization_applied": sum(row["personalization_applied"] for row in adapted_rows),
        "post_feedback_helpful": sum(bool(row["simulated_user"].get("helpful")) for row in adapted_rows),
        "applied_helpful": sum(bool(row["simulated_user"].get("helpful")) for row in applied_rows),
        "applied_helpful_rate": round(
            sum(bool(row["simulated_user"].get("helpful")) for row in applied_rows)
            / max(1, len(applied_rows)), 4,
        ),
        "mean_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "p90_latency_ms": round(latencies[p90_index], 2),
        "l2_claims": sum(int(row.get("layer") or 0) == 2 for row in final_claims),
        "l3_claims": sum(int(row.get("layer") or 0) == 3 for row in final_claims),
        "reflection": reflection,
    }
    return {
        "persona": PERSONA,
        "stable_preference": PREFERENCE,
        "summary": summary,
        "final_claims": final_claims,
        "l3_profile_graph": l3_graph,
        "rows": rows,
    }


def write_report(payload: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORT_DIR / f"personalization-adoption-glm-{stamp}.json"
    md_path = REPORT_DIR / f"personalization-adoption-glm-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = payload["summary"]
    lines = [
        "# GLM 模拟学习者：个性化采纳评测", "",
        f"- 稳定偏好：{payload['stable_preference']}",
        f"- 总轮数：{summary['turns']}；GLM 实际反馈：{summary['glm_simulated_feedback_turns']}",
        f"- 反馈后轮次个性化生效：{summary['post_feedback_personalization_applied']}/{summary['post_feedback_turns']}",
        f"- 反馈后 GLM 认为适合：{summary['post_feedback_helpful']}/{summary['post_feedback_turns']}",
        f"- 真正采用偏好后的适合率：{summary['applied_helpful']}/{summary['post_feedback_personalization_applied']}（{summary['applied_helpful_rate']:.1%}）",
        f"- 回答延迟：平均 {summary['mean_latency_ms'] / 1000:.2f}s；P90 {summary['p90_latency_ms'] / 1000:.2f}s",
        f"- 最终 L2：{summary['l2_claims']}；L3：{summary['l3_claims']}", "",
        "| 轮次 | 问题 | 门控状态 | 采用偏好 | GLM反馈 | 延迟 | 反馈理由 |", "|---:|---|---|---:|---|---:|---|",
    ]
    for row in payload["rows"]:
        sim = row["simulated_user"]
        lines.append(
            f"| {row['turn']} | {row['question']} | {row['personalization'].get('status')} | "
            f"{len(row['applied_claim_ids'])} | {'适合' if sim.get('helpful') else '不适合'} | "
            f"{row['latency_ms'] / 1000:.2f}s | "
            f"{str(sim.get('reason', '')).replace('|', '｜')} |"
        )
    lines.extend(["", "## 最终教学偏好声明", ""])
    for claim in payload["final_claims"]:
        lines.append(
            f"- L{claim['layer']} `{claim['scope_type']}:{claim['scope_key'] or 'all'}` "
            f"{claim['confidence']:.2f} / {claim['status']}：{claim['claim_text']}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Use GLM as a preference-stable simulated learner")
    parser.add_argument("--database", type=Path, default=DB_PATH)
    args = parser.parse_args()
    os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    payload = run(args.database)
    json_path, md_path = write_report(payload)
    print(json.dumps({
        "summary": payload["summary"],
        "report_json": str(json_path),
        "report_markdown": str(md_path),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
