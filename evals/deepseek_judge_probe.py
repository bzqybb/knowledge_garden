from __future__ import annotations

import json
import time

from openai import OpenAI

from evals.judge_config import judge_api_key, judge_base_url


def main() -> None:
    key = judge_api_key("deepseek-v4-pro")
    base_url = judge_base_url("deepseek-v4-pro")
    for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
        started = time.perf_counter()
        client = OpenAI(api_key=key, base_url=base_url, timeout=25, max_retries=0)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": '只输出 JSON：{"ok": true}'}],
                response_format={"type": "json_object"},
                max_tokens=30,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            outcome = {
                "model": model,
                "ok": True,
                "latency_seconds": round(time.perf_counter() - started, 2),
                "content": str(response.choices[0].message.content or ""),
            }
        except Exception as exc:
            outcome = {
                "model": model,
                "ok": False,
                "latency_seconds": round(time.perf_counter() - started, 2),
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:400],
            }
        print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
