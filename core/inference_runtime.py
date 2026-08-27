from __future__ import annotations

import os
import threading
from typing import Any


_LOCK = threading.Lock()
_CONFIGURED = False


def configure_local_inference(torch_module: Any) -> int:
    """Keep concurrent CPU embedding/reranking from oversubscribing every core."""
    global _CONFIGURED

    raw_threads = os.getenv("GARDEN_LOCAL_MODEL_THREADS", "4").strip()
    try:
        threads = max(1, min(int(raw_threads), max(1, os.cpu_count() or 1)))
    except ValueError:
        threads = min(4, max(1, os.cpu_count() or 1))
    with _LOCK:
        if _CONFIGURED:
            return int(torch_module.get_num_threads())
        torch_module.set_num_threads(threads)
        try:
            torch_module.set_num_interop_threads(min(2, threads))
        except RuntimeError:
            # PyTorch forbids changing inter-op threads after work has started.
            # The intra-op limit still prevents the major oversubscription issue.
            pass
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _CONFIGURED = True
        return threads
