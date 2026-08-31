from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multiuser import AuthRegistry, TenantGardenStore


class MultiUserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.auth = AuthRegistry(self.root / "auth.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accounts_sessions_and_tenant_databases_are_isolated(self):
        user_a, token_a = self.auth.register("a@example.com", "password-a-123")
        # Registration is closed after the first account by default, so use a
        # second isolated registry row to exercise the tenant router directly.
        with self.auth.connect() as conn:
            first = conn.execute("SELECT password_salt,password_hash FROM users LIMIT 1").fetchone()
            conn.execute(
                """INSERT INTO users(user_id,email,password_salt,password_hash,created_at)
                   VALUES('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb','b@example.com',?,?,datetime('now'))""",
                (first["password_salt"], first["password_hash"]),
            )
            conn.execute(
                """INSERT INTO improvement_preferences(user_id,consent,updated_at)
                   VALUES('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',0,datetime('now'))"""
            )
        user_b = {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "email": "b@example.com"}

        self.assertEqual(self.auth.user_for_token(token_a), user_a)
        router = TenantGardenStore(self.root / "local.db", self.root / "users")
        with router.using_user(user_a["id"]):
            router.set_setting("private_marker", "A")
            path_a = router.path
        with router.using_user(user_b["id"]):
            self.assertEqual(router.setting("private_marker", "missing"), "missing")
            router.set_setting("private_marker", "B")
            path_b = router.path
        with router.using_user(user_a["id"]):
            self.assertEqual(router.setting("private_marker", ""), "A")

        self.assertNotEqual(path_a, path_b)
        self.assertTrue(path_a.is_file())
        self.assertTrue(path_b.is_file())

    def test_only_consented_interactions_enter_sanitized_candidate_pool(self):
        user, _ = self.auth.register("tester@example.com", "password-test-123")
        ignored = self.auth.capture_interaction(
            user_id=user["id"], request_id="request-1", surface="gardener_chat",
            question="联系 me@example.com", answer="token sk-secretvalue123456", metadata={},
        )
        self.assertFalse(ignored["captured"])

        self.auth.set_consent(user["id"], True)
        captured = self.auth.capture_interaction(
            user_id=user["id"], request_id="request-1", surface="gardener_chat",
            question="联系 me@example.com 或 13800138000",
            answer="token sk-secretvalue123456",
            metadata={"routing_target": "FACT_RETRIEVAL_ONLY"},
        )
        self.assertTrue(captured["captured"])
        with self.auth.connect() as conn:
            row = conn.execute("SELECT * FROM interaction_candidates").fetchone()
        self.assertNotIn("me@example.com", row["question"])
        self.assertNotIn("13800138000", row["question"])
        self.assertNotIn("sk-secretvalue123456", row["answer"])
        self.assertIn(row["dataset_split"], {"development", "holdout"})
        self.assertEqual(row["status"], "sealed" if row["dataset_split"] == "holdout" else "pending")
        metadata = json.loads(row["metadata_json"])
        self.assertEqual(metadata["routing_target"], "FACT_RETRIEVAL_ONLY")

    def test_logout_revokes_session(self):
        user, token = self.auth.register("logout@example.com", "password-logout-123")
        self.assertEqual(self.auth.user_for_token(token)["id"], user["id"])
        self.auth.logout(token)
        self.assertIsNone(self.auth.user_for_token(token))


if __name__ == "__main__":
    unittest.main()
