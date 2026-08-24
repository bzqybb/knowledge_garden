from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
