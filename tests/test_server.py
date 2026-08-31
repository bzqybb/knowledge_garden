from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from app import GardenHTTPServer, GardenHandler


class ServerSafetyTests(unittest.TestCase):
    def test_windows_style_exclusive_port_blocks_second_server(self) -> None:
        first = GardenHTTPServer(("127.0.0.1", 0), GardenHandler)
        second = None
        try:
            with self.assertRaises(OSError):
                second = GardenHTTPServer(("127.0.0.1", first.server_address[1]), GardenHandler)
        finally:
            first.server_close()
            if second is not None:
                second.server_close()

    def test_desktop_cors_allows_only_tauri_origin(self) -> None:
        server = GardenHTTPServer(("127.0.0.1", 0), GardenHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/api/auth/status"
        try:
            with patch.dict(os.environ, {"GARDEN_DESKTOP_INSTANCE_ID": "test-desktop"}):
                allowed = Request(base_url, headers={"Origin": "http://tauri.localhost"})
                with urlopen(allowed, timeout=5) as response:
                    self.assertEqual(
                        response.headers.get("Access-Control-Allow-Origin"),
                        "http://tauri.localhost",
                    )
                    self.assertEqual(
                        response.headers.get("Access-Control-Allow-Private-Network"),
                        "true",
                    )

                rejected = Request(base_url, headers={"Origin": "https://untrusted.example"})
                with urlopen(rejected, timeout=5) as response:
                    self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
