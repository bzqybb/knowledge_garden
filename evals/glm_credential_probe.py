from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI

from core.credentials import load_secret


DEFAULT_PATHS = (
    Path("data/runtime/glm-generator-api-key.dpapi"),
    Path("data/runtime/glm-generator-api-key.before-paratera-20260829.dpapi"),
    Path("data/runtime/glm-eval-api-key.dpapi"),
)


def probe(path: Path) -> tuple[bool, str]:
    """Send a data-free health prompt without ever printing the credential."""

    try:
        api_key = load_secret(path).strip()
        client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            timeout=30,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model="glm-5.2",
            max_tokens=8,
            messages=[{"role": "user", "content": "只回复 OK"}],
            extra_body={
                "thinking": {"type": "disabled"},
                "reasoning_effort": "none",
            },
        )
        content = str(response.choices[0].message.content or "").strip()
        return True, content[:40]
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"[:220]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    for path in args.paths or DEFAULT_PATHS:
        ok, detail = probe(path)
        print(f"{path.name}\t{'OK' if ok else 'FAIL'}\t{detail}")


if __name__ == "__main__":
    main()
