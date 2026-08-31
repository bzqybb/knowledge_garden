from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.credentials import load_secret


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start an isolated, authenticated Knowledge Garden test server.",
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--credential-path", type=Path, required=True)
    parser.add_argument(
        "--base-url", default="https://open.bigmodel.cn/api/coding/paas/v4",
    )
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--desktop-installer-path", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    credential_path = args.credential_path.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_secret(credential_path).strip()
    if not api_key:
        raise RuntimeError("The selected encrypted model credential is empty.")

    desktop_installer = (
        args.desktop_installer_path.expanduser().resolve()
        if args.desktop_installer_path else None
    )
    if desktop_installer and not desktop_installer.is_file():
        raise FileNotFoundError(f"Desktop installer not found: {desktop_installer}")

    os.environ.update({
        "GARDEN_AUTH_REQUIRED": "true",
        "GARDEN_ALLOW_SIGNUP": "true",
        "GARDEN_COOKIE_SECURE": "true",
        "GARDEN_DATA_DIR": str(data_dir),
        "GARDEN_API_KEY": api_key,
        "GARDEN_DISABLE_SAVED_API_KEY": "1",
        "GARDEN_BASE_URL": args.base_url.rstrip("/"),
        "GARDEN_MODEL": args.model,
        "GARDEN_RELEASE_VERSION": "public-beta-v2",
        "GARDEN_FEED_INTERVAL_MINUTES": "1440",
        "GARDEN_MEMORY_INTERVAL_MINUTES": "1440",
        "GARDEN_DESKTOP_DOWNLOAD_URL": "/downloads/windows" if desktop_installer else "",
        "GARDEN_DESKTOP_INSTALLER_PATH": str(desktop_installer) if desktop_installer else "",
    })

    # app.py and core.config read their deployment flags at import time, so all
    # public-mode environment variables must be set before importing the app.
    sys.argv = ["app.py", "--host", "127.0.0.1", "--port", str(args.port)]
    from app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
