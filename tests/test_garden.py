import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from core.agent import answer_from_wiki, briefing, daily_digest, patrol_vault, save_agent_insight, update_agents_manifest
from core.compiler import ingest_raw, validate_links
from core.context import ChatMessage, GardenContext, KnowledgeScope, LearnerSettings, ToolPolicy
from core.context_builder import ContextBuilder
from core.deepdiagram_adapter import DiagramSpec, build_local_diagram, validate_diagram
from core.engine import (
    add_interest,
    analyze_frontier,
    analyze_material_structure,
    article_preview_metadata,
    evaluate_review,
    extract_concepts,
)
from core.gardener_graph import (
    _fallback_planner,
    _fallback_wechat_lookup,
    _requests_wechat_history,
    _wechat_time_params,
    audit_evidence,
    generate_answer,
    generate_visualization,
    repair_outputs,
    review_answer,
)
from core.inspiration import explore_inspiration, save_inspiration_seed
from core.mindmap import build_mindmap
from core.learning_memory import LearningMemoryService
from core.obsidian import sync_vault
from core.retrieval import classify_textbook_structure, ingest_pdf_directory, rebuild_domain_map, search_notes
from core.storage import GardenStore
from core.taxonomy import classify_unmounted_concepts, rebuild_concept_hierarchy
from core.tracememo import TraceMemoConfig, TraceMemoClient, normalize_message, tracememo_config
from core.web_research import _WeChatArticleParser


class GardenTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("GARDEN_API_KEY", None)
        os.environ["GARDEN_DISABLE_SAVED_API_KEY"] = "1"
        os.environ["GARDEN_DISABLE_SAVED_TRACEMEMO_TOKEN"] = "1"
        os.environ["GARDEN_DISABLE_NETWORK"] = "1"
        os.environ.pop("TRACEMEMO_API_TOKEN", None)
        os.environ.pop("WECHATEXPLORER_API_TOKEN", None)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = GardenStore(self.root / "garden.db")

    def tearDown(self):
        os.environ.pop("TRACEMEMO_API_TOKEN", None)
        os.environ.pop("WECHATEXPLORER_API_TOKEN", None)
        self.temp.cleanup()

    def test_official_article_preview_has_domain_summary_knowledge_and_reading_time(self):
        body = ("这篇文章讨论宫廷文化、古代礼仪与历史叙事之间的关系。"
                "作者结合文物材料解释宫廷宴饮如何塑造文化记忆。") * 50
        preview = article_preview_metadata("宫廷宴饮与文化记忆", body)
        self.assertEqual(preview["domain"], "历史与文化")
        self.assertTrue(preview["summary"])
        self.assertIn("宫廷", preview["knowledge"])
        self.assertGreaterEqual(preview["reading_minutes"], 1)

    def test_garden_context_is_immutable_and_separates_runtime_state(self):
        history = ChatMessage(
            message_id="msg-history",
            role="assistant",
            content="我们刚才讨论了统一上下文。",
        )
        current = ChatMessage(
            message_id="msg-current",
            role="user",
            content="请结合我的知识库解释证据链。",
        )
        context = GardenContext(
            request_id="req-1",
            session_id="session-1",
            current_message=current,
            conversation_history=(history,),
            learner_settings=LearnerSettings(
                declared_level="本科入门",
                explicit_interests=("知识管理",),
            ),
            knowledge_scope=KnowledgeScope(selected_note_ids=(1, 2)),
            tool_policy=ToolPolicy(
                user_enabled=("local_wiki",),
                allowed=("local_wiki",),
                mounted=("local_wiki",),
            ),
        )

        self.assertEqual(context.current_message.role, "user")
        self.assertEqual(context.created_at.utcoffset().total_seconds(), 0)
        self.assertFalse(hasattr(context, "retrieved_knowledge"))
        self.assertFalse(hasattr(context, "teaching_strategy"))
        with self.assertRaises(ValidationError):
            context.active_capability = "review"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            ToolPolicy(allowed=("local_wiki",), mounted=("public_web",))
        with self.assertRaises(ValidationError):
            GardenContext(
                request_id="req-2",
                session_id="session-1",
                current_message=history,
            )

    def test_learning_memory_migration_creates_nine_tables_once(self):
        expected = {
            "schema_migrations",
            "sessions",
            "session_messages",
            "learning_events",
            "event_concepts",
            "memory_claims",
            "memory_claim_evidence",
            "concept_mastery",
            "concept_mastery_evidence",
        }
        with self.store.connect() as conn:
            actual = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            migrations = conn.execute(
                "SELECT version,name FROM schema_migrations ORDER BY version"
            ).fetchall()

        self.assertEqual(actual & expected, expected)
        self.assertEqual(len(expected), 9)
        self.assertEqual(foreign_keys, 1)
        self.assertEqual([(row["version"], row["name"]) for row in migrations], [
            (1, "001_learning_memory.sql"),
            (2, "002_learning_evolution.sql"),
            (3, "003_wechat_gateway.sql"),
        ])

        GardenStore(self.root / "garden.db")
        with self.store.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(count, 3)

    def test_tracememo_gateway_only_allows_loopback_and_normalizes_messages(self):
        with self.assertRaises(ValueError):
            tracememo_config("https://example.com")
        config = TraceMemoConfig("http://127.0.0.1:6131", "token", False)
        self.assertEqual(TraceMemoClient(config).config.base_url, "http://127.0.0.1:6131")
        message = normalize_message({
            "msgSvrId": "wx-1", "senderName": "小园", "createTime": 123,
            "strContent": "讨论预测编码", "type": 1,
        })
        self.assertEqual(message["source_message_id"], "wx-1")
        self.assertEqual(message["sender"], "小园")
        self.assertEqual(message["content"], "讨论预测编码")

    def test_tracememo_normalizes_official_account_cards_and_filters_accounts(self):
        message = normalize_message({
            "id": "article-1", "from": "user", "type": "公众号链接", "createTime": 1756798668,
            "contentData": {
                "type": "share", "title": "一篇文章", "des": "文章摘要",
                "url": "https://mp.weixin.qq.com/s/example", "appname": "示例公众号",
            },
        })
        self.assertEqual(message["content"], "一篇文章\n文章摘要")
        self.assertEqual(message["sender"], "示例公众号")
        self.assertEqual(message["article"]["title"], "一篇文章")
        self.assertFalse(message["is_system"])

        client = TraceMemoClient(TraceMemoConfig("http://127.0.0.1:6131", "token", False))
        with patch.object(client, "contacts", return_value={"contacts": [
            {"m_nsUsrName": "123@chatroom", "m_nsNickName": "普通群", "isOfficialAccount": False},
            {"m_nsUsrName": "gh_abc", "m_nsNickName": "知识号", "isOfficialAccount": True},
        ]}):
            accounts = client.official_accounts()
        self.assertEqual(accounts["count"], 1)
        self.assertEqual(accounts["items"][0]["m_nsNickName"], "知识号")

    def test_tracememo_bounded_chatlog_keeps_newest_messages(self):
        client = TraceMemoClient(TraceMemoConfig("http://127.0.0.1:6131", "token", False))
        payload = {"messages": [
            {"id": str(index), "content": f"message-{index}", "createTime": 1700000000 + index}
            for index in range(305)
        ]}
        with patch.object(client, "_get", return_value=payload):
            result = client.chatlog("测试群")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["messages"]), 300)
        self.assertEqual(result["messages"][0]["source_message_id"], "5")
        self.assertEqual(result["messages"][-1]["source_message_id"], "304")

    def test_wechat_article_parser_reads_only_js_content(self):
        parser = _WeChatArticleParser()
        parser.feed('<html><body>导航<div id="js_content"><h2>标题</h2><p>第一段</p><script>bad()</script><p>第二段</p></div>页脚</body></html>')
        text = "".join(parser.parts)
        self.assertIn("标题", text)
        self.assertIn("第一段", text)
        self.assertIn("第二段", text)
        self.assertNotIn("导航", text)
        self.assertNotIn("bad", text)

    def test_gardener_only_mounts_wechat_for_explicit_personal_history_requests(self):
        self.assertTrue(_requests_wechat_history("帮我总结昨天在技术交流群里的微信讨论"))
        self.assertFalse(_requests_wechat_history("微信公众号为什么采用订阅机制？"))
        self.assertFalse(_requests_wechat_history("请解释群体心理学"))
        lookup = _fallback_wechat_lookup("帮我总结昨天在技术交流群里的微信讨论")
        self.assertTrue(lookup.requested)
        self.assertEqual(lookup.talker, "技术交流群")
        self.assertEqual(lookup.time_hint, "昨天")
        self.assertFalse(lookup.needs_clarification)

    def test_wechat_relative_time_uses_tracememo_clock(self):
        params = _wechat_time_params("昨天", {"now": "2026-08-24T09:30:00+08:00"})
        start = datetime.fromtimestamp(int(params["start_time"]), tz=timezone(timedelta(hours=8)))
        end = datetime.fromtimestamp(int(params["end_time"]), tz=timezone(timedelta(hours=8)))
        self.assertEqual(start.isoformat(), "2026-08-23T00:00:00+08:00")
        self.assertEqual(end.isoformat(), "2026-08-24T00:00:00+08:00")

    def test_tracememo_legacy_token_is_read_but_new_name_has_priority(self):
        os.environ["WECHATEXPLORER_API_TOKEN"] = "legacy"
        self.assertEqual(tracememo_config().token, "legacy")
        os.environ["TRACEMEMO_API_TOKEN"] = "current"
        self.assertEqual(tracememo_config().token, "current")

    def test_wechat_candidate_is_l1_until_explicit_review(self):
        candidate = self.store.create_wechat_candidate(
            title="预测编码讨论",
            talker="技术交流群",
            time_range="2026-08-23",
            contact={"name": "技术交流群"},
            query={"time": "2026-08-23"},
            messages=[{
                "source_message_id": "wx-1", "sender": "同学甲", "sent_at": "10:00",
                "content": "预测误差会推动模型更新。", "message_type": "text", "source": {"id": "wx-1"},
            }],
        )
        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["messages"][0]["source_message_id"], "wx-1")
        with self.store.connect() as conn:
            event = conn.execute(
                "SELECT event_type,source_kind FROM learning_events WHERE surface='wechat_import'"
            ).fetchone()
            notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        self.assertEqual((event["event_type"], event["source_kind"]), ("wechat_candidate_created", "explicit"))
        self.assertEqual(notes, 0)
        reviewed = self.store.review_wechat_candidate(candidate["candidate_id"], False)
        self.assertEqual(reviewed["status"], "rejected")

    def test_memory_and_mastery_records_require_relational_evidence(self):
        note_id, _ = self.store.upsert_note({
            "path": "wiki/01-概念底座/证据链.md",
            "title": "证据链",
            "kind": "concept",
            "content": "长期判断必须能追溯到学习事件。",
            "content_hash": "evidence-chain",
        })
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id,title) VALUES(?,?)",
                ("session-1", "统一上下文讨论"),
            )
            conn.execute(
                """INSERT INTO session_messages(
                       message_id,session_id,request_id,role,content
                   ) VALUES(?,?,?,?,?)""",
                ("message-1", "session-1", "request-1", "user", "我更容易从图中理解层级。"),
            )
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,session_id,message_id,surface,event_type,source_kind,payload_json
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    "event-1", "session-1", "message-1", "gardener_chat",
                    "teaching_preference_observed", "observed", '{"preference":"visual"}',
                ),
            )
            conn.execute(
                "INSERT INTO event_concepts(event_id,concept_key,concept_note_id) VALUES(?,?,?)",
                ("event-1", "证据链", note_id),
            )
            conn.execute(
                """INSERT INTO memory_claims(
                       claim_id,layer,dimension,claim_text,source_kind,confidence,status
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    "claim-1", 2, "teaching_preference",
                    "用户在层级关系复杂时偏好视觉结构", "observed", 0.62, "candidate",
                ),
            )
            conn.execute(
                "INSERT INTO memory_claim_evidence(claim_id,event_id,weight) VALUES(?,?,?)",
                ("claim-1", "event-1", 0.8),
            )
            conn.execute(
                """INSERT INTO concept_mastery(
                       concept_key,concept_note_id,stage,confidence,last_evidence_at
                   ) VALUES(?,?,?,?,?)""",
                ("证据链", note_id, "explains", 0.58, "2026-08-22T00:00:00Z"),
            )
            conn.execute(
                """INSERT INTO concept_mastery_evidence(
                       concept_key,event_id,dimension,outcome,weight,stage_after
                   ) VALUES(?,?,?,?,?,?)""",
                ("证据链", "event-1", "explanation", "supports", 0.7, "explains"),
            )

            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO memory_claim_evidence(claim_id,weight) VALUES(?,?)",
                    ("claim-1", 1.0),
                )

        with self.store.connect() as conn:
            evidence = conn.execute(
                "SELECT event_id,relation,weight FROM memory_claim_evidence WHERE claim_id=?",
                ("claim-1",),
            ).fetchone()
            mastery = conn.execute(
                "SELECT stage,confidence FROM concept_mastery WHERE concept_key=?",
                ("证据链",),
            ).fetchone()
        self.assertEqual(evidence["event_id"], "event-1")
        self.assertEqual(evidence["relation"], "supports")
        self.assertEqual(mastery["stage"], "explains")

    def test_context_builder_uses_explicit_settings_not_inferred_memory(self):
        self.store.set_setting("learning_level", "本科进阶")
        self.store.set_setting("interests", ["心理学", "人工智能"])
        self.store.set_setting("teaching_preferences", ["先讲机制，再给例子"])
        context = ContextBuilder(self.store).build(
            "为什么需要证据链？",
            [{"role": "assistant", "content": "刚才讨论了长期记忆。"}],
            session_id="session-context",
            request_id="request-context",
            message_id="message-context",
        )

        self.assertEqual(context.learner_settings.declared_level, "本科进阶")
        self.assertEqual(context.learner_settings.explicit_interests, ("心理学", "人工智能"))
        self.assertIn("local_wiki", context.tool_policy.mounted)
        self.assertFalse(hasattr(context, "active_memory_claims"))
        self.assertFalse(hasattr(context, "intent"))

    def test_personalization_memory_is_scope_gated_and_keeps_evidence(self):
        with self.store.connect() as conn:
            conn.execute("INSERT INTO sessions(session_id,title) VALUES('scope-session','范围门控')")
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,session_id,surface,event_type,source_kind,payload_json
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    "scope-event", "scope-session", "gardener_chat",
                    "teaching_preference_observed", "explicit",
                    json.dumps({"observation": "讨论心理测量时，我希望先看结构再看细节。"}, ensure_ascii=False),
                ),
            )
            conn.execute(
                """INSERT INTO memory_claims(
                       claim_id,layer,dimension,scope_type,scope_key,claim_text,
                       source_kind,confidence,status
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    "scope-claim", 2, "teaching_preference", "concept", "心理测量",
                    "讨论心理测量时先给结构可能更有效", "explicit", 0.9, "active",
                ),
            )
            conn.execute(
                "INSERT INTO memory_claim_evidence(claim_id,event_id,weight) VALUES(?,?,?)",
                ("scope-claim", "scope-event", 1.0),
            )
        unrelated = LearningMemoryService(self.store).active_memory_context(
            ["微积分"], task_keys=["explain_mechanism"]
        )
        related = LearningMemoryService(self.store).active_memory_context(
            ["心理测量"], task_keys=["explain_mechanism"]
        )
        self.assertEqual(unrelated["claims"], [])
        self.assertEqual(related["claims"][0]["claim_id"], "scope-claim")
        self.assertIn("先看结构", related["claims"][0]["evidence"][0]["observation"])

    def test_negative_personalization_feedback_adds_counterevidence(self):
        with self.store.connect() as conn:
            conn.execute("INSERT INTO sessions(session_id,title) VALUES('feedback-session','反馈')")
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,session_id,surface,event_type,source_kind,payload_json
                   ) VALUES('feedback-seed','feedback-session','gardener_chat',
                            'teaching_preference_observed','explicit','{}')"""
            )
            conn.execute(
                """INSERT INTO memory_claims(
                       claim_id,layer,dimension,scope_type,scope_key,claim_text,
                       source_kind,confidence,status
                   ) VALUES('feedback-claim',2,'teaching_preference','task','explain_mechanism',
                            '机制问题先使用结构图','explicit',0.75,'active')"""
            )
            conn.execute(
                "INSERT INTO memory_claim_evidence(claim_id,event_id,weight) VALUES('feedback-claim','feedback-seed',1.0)"
            )
            metadata = json.dumps({"personalization": {
                "status": "applied", "task_key": "explain_mechanism",
                "strategy_summary": "先结构后机制", "applied_claim_ids": ["feedback-claim"],
            }}, ensure_ascii=False)
            conn.execute(
                """INSERT INTO session_messages(
                       message_id,session_id,request_id,role,content,metadata_json
                   ) VALUES(?,?,?,?,?,?)""",
                ("feedback-answer", "feedback-session", "feedback-request", "assistant", "回答", metadata),
            )
        memory = LearningMemoryService(self.store)
        result = memory.record_personalization_feedback(
            request_id="feedback-request", helpful=False, feedback_note="我想直接看推导"
        )
        duplicate = memory.record_personalization_feedback(
            request_id="feedback-request", helpful=False
        )
        with self.store.connect() as conn:
            claim = conn.execute(
                "SELECT confidence,status FROM memory_claims WHERE claim_id='feedback-claim'"
            ).fetchone()
            relation = conn.execute(
                """SELECT relation FROM memory_claim_evidence
                   WHERE claim_id='feedback-claim' AND relation='contradicts'"""
            ).fetchone()
        self.assertTrue(result["recorded"])
        self.assertFalse(duplicate["recorded"])
        self.assertAlmostEqual(claim["confidence"], 0.47)
        self.assertEqual(claim["status"], "candidate")
        self.assertEqual(relation["relation"], "contradicts")

    def test_gardener_turn_persists_one_session_messages_and_l1_event(self):
        self.store.upsert_note({
            "path": "wiki/01-概念底座/证据链.md", "title": "证据链", "kind": "concept",
            "content": "证据链让长期判断可以追溯到原始观察。", "content_hash": "turn-evidence",
        })
        first = answer_from_wiki(self.store, "证据链是什么？", session_id="stable-session")
        second = answer_from_wiki(
            self.store,
            "那它为什么重要？",
            [{"role": "user", "content": "证据链是什么？"},
             {"role": "assistant", "content": first["answer"]}],
            session_id="stable-session",
        )
        with self.store.connect() as conn:
            roles = [row["role"] for row in conn.execute(
                "SELECT role FROM session_messages WHERE session_id=? ORDER BY created_at,rowid",
                ("stable-session",),
            )]
            event_count = conn.execute(
                "SELECT COUNT(*) n FROM learning_events WHERE session_id=? AND event_type='question_asked'",
                ("stable-session",),
            ).fetchone()["n"]
        self.assertEqual(first["session_id"], "stable-session")
        self.assertEqual(second["session_id"], "stable-session")
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertEqual(event_count, 2)

    def test_reflection_requires_repetition_across_sessions_before_activating_l2(self):
        memory = LearningMemoryService(self.store)
        session_a = memory.begin_turn("问题 A", "reflection-a")["session_id"]
        session_b = memory.begin_turn("问题 B", "reflection-b")["session_id"]
        for index in range(5):
            memory.record_event(
                surface="gardener_chat",
                event_type="question_asked",
                source_kind="observed",
                session_id=session_a if index < 3 else session_b,
                payload={"primary_intent": "explain_mechanism"},
            )
        reflection = memory.reflect(force=True)
        with self.store.connect() as conn:
            claims = [dict(row) for row in conn.execute(
                "SELECT * FROM memory_claims WHERE dimension='question_pattern'"
            )]
            evidence_count = conn.execute(
                """SELECT COUNT(*) n FROM memory_claim_evidence e
                   JOIN memory_claims c ON c.claim_id=e.claim_id
                   WHERE c.dimension='question_pattern'"""
            ).fetchone()["n"]
        self.assertTrue(reflection["triggered"])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["layer"], 2)
        self.assertEqual(claims[0]["status"], "active")
        self.assertEqual(evidence_count, 5)
        self.assertEqual(reflection["l3_created"], 0)

    def test_knowledge_access_changes_value_without_automatic_deletion(self):
        memory = LearningMemoryService(self.store)
        active_id, _ = self.store.upsert_note({
            "path": "wiki/01-概念底座/活跃证据.md", "title": "活跃证据", "kind": "concept",
            "content": "证据用于支持可以检查的知识判断。", "content_hash": "active",
        })
        quiet_id, _ = self.store.upsert_note({
            "path": "wiki/01-概念底座/安静证据.md", "title": "安静证据", "kind": "concept",
            "content": "证据用于支持可以检查的知识判断。", "content_hash": "quiet",
        })
        for _ in range(3):
            memory.record_knowledge_access([active_id], reason="test")
        hits = search_notes(self.store, "证据支持知识判断", kinds={"concept"}, limit=10)
        report = memory.refresh_knowledge_weights()
        by_id = {item["id"]: item for item in hits}
        self.assertGreater(by_id[active_id]["knowledge_value"], by_id[quiet_id]["knowledge_value"])
        self.assertLess(hits.index(by_id[active_id]), hits.index(by_id[quiet_id]))
        self.assertIsNotNone(self.store.get_note(quiet_id))
        self.assertEqual(report["auto_compressed"], 0)

    def test_multidimensional_mastery_keeps_raw_evidence_and_decays_projection(self):
        memory = LearningMemoryService(self.store)
        task = {
            "id": 7,
            "task_type": "quiz",
            "concept": "证据链",
            "payload": {"question": "哪一项属于可追溯证据？"},
        }
        for answer_index in range(2):
            plan = memory.plan_mastery_update(task, 3)
            snapshot = memory.apply_mastery_update(
                plan, answer=f"第 {answer_index + 1} 次能够识别证据", task_id=7,
            )
        now_score = snapshot["effective_scores"]["recognition"]
        future = memory.mastery_for(
            "证据链", now=datetime.now(timezone.utc) + timedelta(days=30)
        )
        with self.store.connect() as conn:
            evidence_count = conn.execute(
                "SELECT COUNT(*) n FROM concept_mastery_evidence WHERE concept_key='证据链'"
            ).fetchone()["n"]
        self.assertEqual(snapshot["stage"], "recognizes")
        self.assertGreater(snapshot["recognition_score"], snapshot["explanation_score"])
        self.assertLess(future["effective_scores"]["recognition"], now_score)
        self.assertEqual(evidence_count, 2)

    def test_feynman_probe_waits_for_explanation_evidence_and_does_not_duplicate(self):
        memory = LearningMemoryService(self.store)
        task = {
            "id": 11,
            "task_type": "reflection",
            "concept": "代理梯度",
            "payload": {"question": "请解释代理梯度为什么能帮助训练。"},
        }
        first = memory.plan_mastery_update(task, 3, 3)
        memory.apply_mastery_update(first, answer="第一次机制解释", task_id=11)
        self.assertEqual(
            [item for item in self.store.list_tasks() if item["task_type"] == "feynman"], []
        )
        for index in range(2):
            plan = memory.plan_mastery_update(task, 3, 3)
            memory.apply_mastery_update(plan, answer=f"补充机制解释 {index}", task_id=11)
        probes = [item for item in self.store.list_tasks() if item["task_type"] == "feynman"]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["payload"]["mastery_dimension"], "transfer")
        self.assertIn("新情境", probes[0]["payload"]["question"])

    def test_gardener_graph_hands_structured_state_between_agents(self):
        self.store.upsert_note({
            "path": "wiki/01-概念底座/审美距离.md", "title": "审美距离", "kind": "concept",
            "content": "审美距离让人接近情绪内容，同时保留反思空间。\n来源：https://example.edu/aesthetic-distance",
            "source_url": "https://example.edu/aesthetic-distance", "content_hash": "graph-local",
        })

        def fake_chat_json(system, user, **kwargs):
            if "交互诊断 Agent" in system:
                return {
                    "primary_intent": "explain_mechanism", "secondary_intents": [], "concepts": ["审美距离"],
                    "task_demand": "analyze", "possible_obstacle": "causal_gap", "needs_clarification": False,
                    "clarification_question": "", "evidence": "用户使用了‘为什么’。",
                }
            if "来源规划 Agent" in system:
                return {
                    "local_first": True, "source_types": ["local_wiki", "review"],
                    "search_query": "aesthetic distance emotion mechanism", "recency_needed": False,
                    "rationale": "机制问题需要本地知识和综述。",
                }
            if "证据审查 Agent" in system:
                return {
                    "accepted_ids": ["L1"], "rejected": [], "usable_claims": ["审美距离提供反思空间"],
                    "gaps": [], "sufficient": True, "rationale": "本地概念页直接相关。",
                }
            if "教学策略 Agent" in system:
                return {
                    "teaching_move": "repair_causal_chain", "explanation_order": ["结论", "机制", "边界"],
                    "use_analogy": False, "analogy_basis": "", "rigor": "conceptual",
                    "personalization_basis": "用户正在追问因果机制。", "avoid": ["强行类比"],
                    "success_criterion": "能复述因果链。", "rationale": "修复因果断点。",
                }
            if "教学回答 Agent" in system:
                return {
                    "answer": "关键不是审美会生成新能量，而是它可能通过审美距离改变情绪加工方式。[L1]",
                    "followup": "这条因果链哪一步还不清楚？", "discussion_prompts": ["何时失效？", "如何验证？"],
                }
            if "回答质量审查 Agent" in system:
                return {
                    "passed": True, "answered_question": True, "evidence_bounded": True,
                    "personalization_natural": True, "issues": [], "revised_answer": "", "rationale": "通过。",
                }
            raise AssertionError(system)

        with patch("core.gardener_graph.chat_json", side_effect=fake_chat_json):
            result = answer_from_wiki(self.store, "为什么审美能影响情绪？")
        self.assertEqual(result["intent"]["primary_intent"], "explain_mechanism")
        self.assertEqual(result["teaching_strategy"]["teaching_move"], "repair_causal_chain")
        self.assertEqual(result["citations"][0]["title"], "审美距离")
        self.assertEqual(
            [step["node"] for step in result["agent_trace"]],
            ["planner_intake", "understand_question", "planner_plan", "load_learner_memory", "gate_personalization",
             "plan_sources", "retrieve_sources", "audit_evidence", "choose_teaching_strategy",
             "planner_select_delivery", "build_content_blueprint", "generate_answer",
             "join_deliverables", "reflect_outputs", "assemble_result"],
        )
        self.assertEqual(result["planner"]["visual_kind"], "none")
        self.assertEqual(result["planner"]["primary_modality"], "text")

    def test_deepdiagram_adapter_rejects_code_and_binds_audited_sources(self):
        payload = {
            "title": "机制图", "design_concept": "只画因果链",
            "nodes": [
                {"id": "start", "label": "前提", "role": "step", "evidence_ids": ["L1", "BAD"]},
                {"id": "result", "label": "结果", "role": "step", "evidence_ids": ["L1"]},
                {"id": "x", "label": "<script>alert(1)</script>", "role": "unknown", "evidence_ids": []},
            ],
            "edges": [{"source": "start", "target": "result", "label": "导致"}],
        }
        diagram = validate_diagram(payload, requested_kind="flowchart", allowed_source_ids={"L1"})
        self.assertEqual(diagram["status"], "ready")
        self.assertEqual(diagram["source_ids"], ["L1"])
        self.assertNotIn("BAD", json.dumps(diagram))
        self.assertNotIn("script", json.dumps(diagram))

    def test_evidence_gate_blocks_factual_generation_without_direct_source(self):
        state = {
            "evidence_review": {"sufficient": False, "gaps": ["没有直接证据"]},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        with patch("core.gardener_graph.chat_json") as model:
            result = generate_answer(state)
        model.assert_not_called()
        self.assertIn("证据不足", result["answer"])
        self.assertIn("硬门控", result["trace"][-1]["summary"])

    def test_audit_does_not_promote_same_domain_textbook_without_object_match(self):
        state = {
            "question": "为什么特征向量在线性变换后方向不变？",
            "intent": {
                "primary_intent": "explain_mechanism",
                "research_object": "特征向量",
                "core_question": "特征向量在线性变换后的方向",
                "claim_to_verify": "",
                "concepts": ["特征向量", "线性变换"],
                "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "特征向量 线性变换"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "L1", "title": "梯度与方向导数",
                "text": "梯度给出多元函数增长最快的方向，并可定义方向导数。" * 8,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"semantic_score": 0.8},
            }],
            "trace": [],
        }
        result = audit_evidence(state)
        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["L1"], "prerequisite")

    def test_planner_selects_spatial_concept_diagram_for_vector_question(self):
        plan = _fallback_planner(
            "为什么特征向量在线性变换后方向不变？请配图。",
            {"primary_intent": "explain_mechanism", "core_question": "特征向量在线性变换中的方向", "concepts": ["特征向量"]},
        )
        self.assertEqual(plan.relation_type, "spatial_geometric")
        self.assertEqual(plan.visual_kind, "concept")
        self.assertEqual(plan.primary_modality, "text_visual")

    def test_full_deepdiagram_failure_falls_back_to_local_diagram(self):
        state = {
            "planner_decision": {"visual_kind": "concept", "visual_request": "画出向量变换前后的共线关系"},
            "accepted_sources": [{"source_id": "L1"}],
            "content_blueprint": {
                "research_object": "特征向量", "core_question": "为什么方向保持共线",
                "direct_source_ids": ["L1"],
                "evidence_items": [{"source_id": "L1", "title": "线性代数", "excerpt": "特征向量经过变换后是原向量的标量倍"}],
                "usable_claims": ["变换结果与原向量共线"], "gaps": [],
            },
            "trace": [],
        }
        with patch("core.gardener_graph.generate_with_full_service", side_effect=OSError("connection refused")):
            result = generate_visualization(state)
        diagram = result["visualization"]
        self.assertEqual(diagram["status"], "ready")
        self.assertEqual(diagram["provider"], "local-deterministic-adapter")
        self.assertIn("connection refused", diagram["warning"])

    def test_low_risk_reflector_uses_local_hard_checks_only(self):
        state = {
            "question": "什么是矩阵？", "intent": {"primary_intent": "define", "response_mode": "standard"},
            "planner_decision": {"complexity": "simple", "primary_modality": "text", "max_revisions": 1},
            "evidence_review": {"sufficient": True, "source_roles": {"L1": "direct_evidence"}},
            "accepted_sources": [{"source_id": "L1"}],
            "answer": "矩阵是按行和列排列的数表，用于表示线性关系。[L1]",
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "teaching_strategy": {}, "trace": [],
        }
        with patch("core.gardener_graph.chat_json") as model:
            result = review_answer(state)
        model.assert_not_called()
        self.assertTrue(result["quality_review"]["passed"])
        self.assertEqual(result["trace"][-1]["data"]["mode"], "local_hard_checks")

    def test_repair_rejects_revised_answer_that_drops_direct_citation(self):
        original = "特征向量经线性变换后仍是原向量的标量倍，因此两者共线。[L1]"
        state = {
            "question": "为什么特征向量方向不变？",
            "answer": original,
            "quality_review": {
                "repair_target": "text",
                "issues": ["需要改善措辞"],
                "revised_answer": "因为定义就是这样，所以方向不变。",
            },
            "evidence_review": {
                "sufficient": True,
                "source_roles": {"L1": "direct_evidence"},
            },
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "trace": [],
        }
        with patch("core.gardener_graph.chat_json", return_value={
            "answer": "换一种说法，变换后的向量仍与原向量共线。"
        }):
            result = repair_outputs(state)
        self.assertEqual(result["answer"], original)
        self.assertIn("[L1]", result["answer"])

    def test_unmounted_concept_enters_evidence_based_classification_lifecycle(self):
        self.store.upsert_note({
            "path": "wiki/04-主题索引/心理学.md", "title": "心理学", "kind": "moc",
            "content": "心理学主题索引，研究行为与心理过程。", "content_hash": "moc-psych",
        })
        evidence = "参照群体效应会改变跨文化问卷分数的比较基准"
        self.store.upsert_note({
            "path": "wiki/01-概念底座/参照群体效应.md", "title": "参照群体效应", "kind": "concept",
            "content": (evidence + "，因而同一自评量表在不同文化群体中可能具有不同解释。") * 3,
            "content_hash": "concept-reference-group",
        })
        with patch("core.taxonomy.chat_json", return_value={
            "status": "classified", "target_moc": "心理学", "confidence": 0.88,
            "evidence": evidence, "reason": "正文研究跨文化测量中的心理参照机制。",
        }):
            result = classify_unmounted_concepts(self.store)
        self.assertEqual(result["classified"], 1)
        tree = build_mindmap(self.store)["tree"]
        psychology = next(item for item in tree["children"] if item["title"] == "心理学")
        self.assertIn("参照群体效应", [item["title"] for item in psychology["children"]])

    def test_obsidian_sync_parses_frontmatter_tags_and_wikilinks(self):
        vault = self.root / "vault"
        note_dir = vault / "wiki" / "01-概念底座"
        note_dir.mkdir(parents=True)
        (note_dir / "注意力机制.md").write_text(
            "---\ntags: [AI, 课程]\n---\n# 注意力机制\n连接到 [[Transformer]]。 #深度学习",
            encoding="utf-8",
        )
        result = sync_vault(vault, self.store)
        self.assertEqual(result["scanned"], 1)
        note = self.store.list_notes()[0]
        self.assertEqual(note["kind"], "concept")
        self.assertIn("深度学习", note["tags"])
        self.assertEqual(self.store.graph()["edges"], [])

    def test_local_search_prefers_matching_knowledge(self):
        self.store.upsert_note({"path": "wiki/01-概念底座/梯度下降.md", "title": "梯度下降", "kind": "concept", "content": "梯度下降使用损失函数的梯度迭代更新参数。", "content_hash": "a"})
        self.store.upsert_note({"path": "wiki/01-概念底座/傅里叶变换.md", "title": "傅里叶变换", "kind": "concept", "content": "将时域信号分解到频域。", "content_hash": "b"})
        hits = search_notes(self.store, "损失函数怎样用梯度更新参数", kinds={"concept"})
        self.assertEqual(hits[0]["title"], "梯度下降")

    def test_retrieval_rejects_generic_word_overlap_across_domains(self):
        self.store.upsert_note({
            "path": "wiki/微积分.md", "title": "高等微积分", "kind": "textbook",
            "content": "为什么需要研究函数极限与积分基础。", "source": "pdf", "content_hash": "calc",
        })
        self.store.upsert_note({
            "path": "wiki/历史唯物主义.md", "title": "经济基础与上层建筑", "kind": "textbook",
            "content": "经济基础制约上层建筑，上层建筑也会反作用于经济基础。", "source": "pdf", "content_hash": "marx",
        })
        hits = search_notes(self.store, "为什么经济基础决定上层建筑", kinds={"textbook"})
        self.assertEqual([item["title"] for item in hits], ["经济基础与上层建筑"])

    def test_placeholder_page_cannot_become_factual_citation(self):
        self.store.upsert_note({
            "path": "wiki/占位概念.md", "title": "审美距离", "kind": "concept",
            "content": "此页由 Ingest 自动建立，等待后续资料继续充实。", "content_hash": "placeholder",
        })
        hits = search_notes(self.store, "审美距离是什么", kinds={"concept"})
        self.assertEqual(hits[0]["knowledge_status"], "placeholder")

    def test_ingest_is_idempotent_and_links_are_closed(self):
        vault = self.root / "vault"
        raw = vault / "raw"
        raw.mkdir(parents=True)
        (raw / "sample.md").write_text(
            "# 脉冲神经网络的代理梯度\n\n- Category: 神经网络 / 类脑计算\n\n脉冲神经网络使用代理梯度算法处理阶跃函数不可微问题。",
            encoding="utf-8",
        )
        first = ingest_raw(vault, "raw/sample.md", self.store)
        second = ingest_raw(vault, "raw/sample.md", self.store)
        self.assertTrue(first["created"])
        self.assertEqual(second["created"], [])
        self.assertTrue((vault / "wiki" / "02-降维对照").is_dir())
        self.assertEqual(validate_links(vault)["unresolved_count"], 0)
        concept_page = vault / "wiki" / "01-概念底座" / "脉冲神经网络.md"
        self.assertIn("## 核心定义", concept_page.read_text(encoding="utf-8"))
        self.assertNotIn("等待后续资料继续充实", concept_page.read_text(encoding="utf-8"))

    def test_frontier_analysis_creates_card_and_review_tasks(self):
        self.store.upsert_note({"path": "wiki/01-概念底座/反向传播.md", "title": "反向传播", "kind": "concept", "content": "反向传播用链式法则计算损失函数对参数的梯度。", "content_hash": "x"})
        result = analyze_frontier(self.store, "残差网络", "残差网络通过快捷连接改善深层网络的梯度传播。")
        self.assertGreaterEqual(len(result["cards"]), 1)
        self.assertGreaterEqual(len(self.store.list_tasks()), 2)

    def test_psychology_article_becomes_nested_knowledge_branch(self):
        vault = self.root / "vault"
        raw = vault / "raw"
        raw.mkdir(parents=True)
        article = (
            "# 被误解的集体主义文化\n\n"
            "集体主义文化中的人可能通过特质-情境匹配进行主动情境选择。"
            "文化差异的测量还会受到参照群体效应影响。人格与行为的关系具有跨文化一致性。"
        )
        (raw / "culture.md").write_text(article, encoding="utf-8")
        result = ingest_raw(vault, "raw/culture.md", self.store)
        self.assertEqual(result["discipline"], "心理学")
        self.assertEqual(result["topic"], "社会与文化心理学")
        self.assertIn("参照群体效应", result["concepts"])
        sync_vault(vault, self.store)
        tree = build_mindmap(self.store)["tree"]
        psychology = next(node for node in tree["children"] if node["title"] == "心理学")
        branch = next(node for node in psychology["children"] if node["title"] == "社会与文化心理学")
        self.assertIn("参照群体效应", {node["title"] for node in branch["children"]})

    def test_reanalysis_replaces_cards_and_tasks_without_duplicates(self):
        text = "文化差异的测量受到参照群体效应影响，个体还会进行主动情境选择。"
        first = analyze_frontier(self.store, "被误解的集体主义文化", text)
        second = analyze_frontier(self.store, "被误解的集体主义文化", text)
        self.assertEqual(len(self.store.list_cards()), len(second["cards"]))
        self.assertEqual(len(self.store.list_tasks()), len(second["cards"]) * 2)
        self.assertEqual(second["removed"]["cards"], len(first["cards"]))

    def test_agent_patrol_updates_manifest_and_briefing(self):
        vault = self.root / "vault"
        raw = vault / "raw"
        raw.mkdir(parents=True)
        (vault / "AGENTS.md").write_text("# 原有规则\n\n请保留这段。", encoding="utf-8")
        (raw / "culture.md").write_text(
            "# 集体主义文化\n\n文化差异的测量受到参照群体效应影响，并涉及主动情境选择。",
            encoding="utf-8",
        )
        result = patrol_vault(vault, self.store)
        manifest = (vault / "AGENTS.md").read_text(encoding="utf-8")
        self.assertEqual(len(result["ingested"]), 1)
        self.assertIn("请保留这段", manifest)
        self.assertIn("Agent 的地位", manifest)
        self.assertIn("sources/culture", manifest)
        self.assertTrue(briefing(self.store)["question"])

    def test_agent_manifest_repairs_double_markdown_extension(self):
        vault = self.root / "vault"
        vault.mkdir()
        (vault / "AGENTS.md.md").write_text("# 原有 Agent 规则", encoding="utf-8")
        result = update_agents_manifest(vault)
        self.assertEqual(Path(result["path"]).name, "AGENTS.md")
        self.assertTrue((vault / "AGENTS.md").is_file())
        self.assertFalse((vault / "AGENTS.md.md").exists())

    def test_agent_query_prefers_compiled_wiki(self):
        self.store.upsert_note({
            "path": "wiki/01-概念底座/参照群体效应.md", "title": "参照群体效应", "kind": "concept",
            "content": "参照群体效应会让不同文化中的相同问卷分数具有不同含义。\n来源：https://example.edu/reference-group-effect",
            "source_url": "https://example.edu/reference-group-effect", "content_hash": "agent-wiki",
        })
        answer = answer_from_wiki(self.store, "为什么跨文化问卷分数不能直接比较？")
        self.assertEqual(answer["evidence_layer"], "wiki")
        self.assertEqual(answer["citations"][0]["title"], "参照群体效应")
        self.assertEqual(answer["web_sources"], [])
        self.assertTrue(answer["offer_save"])

    def test_agent_followup_accepts_conversation_history(self):
        self.store.upsert_note({
            "path": "wiki/01-概念底座/审美距离.md", "title": "审美距离", "kind": "concept",
            "content": "审美距离让人接近情绪内容，同时保留反思空间。\n来源：https://example.edu/aesthetic-distance",
            "source_url": "https://example.edu/aesthetic-distance", "content_hash": "dialogue-wiki",
        })
        answer = answer_from_wiki(self.store, "那它什么时候会失效？", [
            {"role": "user", "content": "审美为什么能调节情绪？"},
            {"role": "assistant", "content": "审美距离可能提供安全边界。"},
        ])
        self.assertEqual(answer["citations"][0]["title"], "审美距离")
        self.assertTrue(answer["followup"])

    def test_agent_writeback_requires_confirmation_then_creates_spark(self):
        vault = self.root / "vault"
        concept_dir = vault / "wiki" / "01-概念底座"
        concept_dir.mkdir(parents=True)
        concept = concept_dir / "审美体验.md"
        concept.write_text("# 审美体验\n\n审美体验涉及注意、情绪与意义建构。", encoding="utf-8")
        sync_vault(vault, self.store)
        self.store.set_setting("vault_path", str(vault))
        citation = next(note for note in self.store.list_notes() if note["title"] == "审美体验")
        result = save_agent_insight(
            self.store,
            "为什么审美体验会改变情绪？",
            "审美距离可以让人接近情绪，同时保留安全边界。",
            [{"id": citation["id"], "title": citation["title"], "path": citation["path"]}],
            [{"title": "Aesthetic emotions", "url": "https://doi.org/10.1/example", "year": 2024, "venue": "Test Journal"}],
            "这种解释在哪些情绪上会失效？",
        )
        output = Path(result["path"])
        self.assertTrue(output.is_file())
        text = output.read_text(encoding="utf-8")
        self.assertIn("[[审美体验]]", text)
        self.assertIn("https://doi.org/10.1/example", text)
        self.assertIn(result["title"], concept.read_text(encoding="utf-8"))
        self.assertTrue(Path(result["concept_path"]).is_file())

    def test_review_tutor_understands_partial_answer_before_followup(self):
        task = {
            "id": 1, "title": "回忆：虚构概念", "task_type": "recall", "concept": "虚构概念",
            "payload": {"question": "它如何起作用？"},
        }
        assessment = {
            "quality": 1, "correct": None, "understood": "你已经说明它会改变比较基准。",
            "feedback": "还需要澄清基准怎样影响结果。", "followup": "这个基准具体改变了哪个判断？",
            "needs_followup": True,
        }
        with patch("core.engine.chat_json", return_value=assessment):
            result = evaluate_review(task, "它会让大家用不同标准看同一件事", 2, [], "参考知识")
        self.assertTrue(result["needs_followup"])
        self.assertIn("已经说明", result["understood"])
        self.assertNotIn("错误", result["feedback"])

    def test_daily_digest_changes_with_interest_profile(self):
        self.store.set_setting("learning_level", "本科入门")
        self.store.set_setting("interests", ["音乐"])
        article = {
            "title": "Music and emotion", "url": "https://doi.org/example", "year": 2025,
            "authors": ["A"], "venue": "Journal", "source": "OpenAlex", "abstract": "音乐与情绪研究摘要。",
        }
        with patch("core.agent.search_academic_articles", return_value=[article]):
            music = daily_digest(self.store, force=True)
        self.assertEqual(music["items"][0]["interest"], "音乐")
        self.assertIn("本科入门", music["items"][0]["why"])
        self.assertIn("before_reading", music["items"][0]["reading_guide"])
        self.assertEqual(len(music["items"][0]["reading_guide"]["checkpoints"]), 3)
        self.assertIn("domain_match", music["items"][0]["scores"])
        self.assertIn("不要一次把答案全部讲完", music["items"][0]["prompt"])
        self.store.set_setting("interests", ["电子电路"])
        with patch("core.agent.search_academic_articles", return_value=[article]):
            circuit = daily_digest(self.store, force=False)
        self.assertEqual(circuit["items"][0]["interest"], "电子电路")

    def test_graph_merges_normalized_duplicate_titles_without_deleting_notes(self):
        first_id, _ = self.store.upsert_note({
            "path": "interest::1", "title": "高速摄影 为什么要舍弃大部分时间", "kind": "interest",
            "content": "兴趣碎片", "content_hash": "same-a",
        })
        second_id, _ = self.store.upsert_note({
            "path": "wiki/03-交叉火花/高速摄影为什么要舍弃大部分时间.md",
            "title": "高速摄影为什么要舍弃大部分时间", "kind": "spark",
            "content": "已经沉淀的解释", "content_hash": "same-b",
        })
        matching = [node for node in self.store.graph()["nodes"] if "高速摄影" in node["title"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["kind"], "spark")
        self.assertEqual(set(matching[0]["merged_ids"]), {first_id, second_id})
        self.assertEqual(len([note for note in self.store.list_notes() if "高速摄影" in note["title"]]), 2)

    def test_mindmap_merges_bilingual_aliases_and_adds_evidenced_concept_level(self):
        moc_id, _ = self.store.upsert_note({
            "path": "wiki/04-主题索引/类脑计算与 SNN.md", "title": "类脑计算与 SNN", "kind": "moc",
            "content": "# 类脑计算与 SNN", "content_hash": "moc-snn",
        })
        titles = ["脉冲神经网络", "脉冲神经网络（SNN）", "代理梯度", "代理梯度（Surrogate Gradient）", "反向传播"]
        ids = {}
        for index, title in enumerate(titles):
            note_id, _ = self.store.upsert_note({
                "path": f"wiki/01-概念底座/{title}.md", "title": title, "kind": "concept",
                "content": f"# {title}\n\n概念说明。", "content_hash": f"alias-{index}",
            })
            ids[title] = note_id
        self.store.replace_wikilinks(moc_id, titles)
        self.store.resolve_links()
        branch = build_mindmap(self.store)["tree"]["children"][0]
        self.assertEqual(branch["title"], "类脑计算与 SNN")
        self.assertEqual(len(branch["children"]), 1)
        anchor = branch["children"][0]
        self.assertEqual(anchor["title"], "脉冲神经网络")
        child_titles = {child["title"] for child in anchor["children"]}
        self.assertEqual(child_titles, {"代理梯度", "反向传播"})

    def test_future_unknown_subject_uses_model_relations_not_fixed_vocabulary(self):
        moc_id, _ = self.store.upsert_note({
            "path": "wiki/04-主题索引/虚构交叉学科.md", "title": "虚构交叉学科", "kind": "moc",
            "content": "# 虚构交叉学科", "content_hash": "future-moc",
        })
        titles = ["泽塔母体", "蓝桥机制", "月纹测量法"]
        for index, title in enumerate(titles):
            self.store.upsert_note({
                "path": f"wiki/01-概念底座/{title}.md", "title": title, "kind": "concept",
                "content": f"# {title}\n\n这是未来新增、规则库从未见过的概念 {index}。", "content_hash": f"future-{index}",
            })
        self.store.replace_wikilinks(moc_id, titles)
        self.store.resolve_links()
        planned = {
            "relations": [
                {"parent": "泽塔母体", "child": "蓝桥机制", "reason": "机制属于该对象", "confidence": 0.91},
                {"parent": "蓝桥机制", "child": "月纹测量法", "reason": "测量法用于检验机制", "confidence": 0.82},
            ],
            "unresolved": [],
        }
        with patch("core.taxonomy.chat_json", return_value=planned):
            result = rebuild_concept_hierarchy(self.store, force=True)
        self.assertEqual(result["method"], "langchain")
        branch = build_mindmap(self.store)["tree"]["children"][0]
        self.assertEqual(branch["children"][0]["title"], "泽塔母体")
        self.assertEqual(branch["children"][0]["children"][0]["title"], "蓝桥机制")
        self.assertEqual(branch["children"][0]["children"][0]["children"][0]["title"], "月纹测量法")

    def test_interest_capture_discovers_professional_link(self):
        self.store.upsert_note({"path": "wiki/01-概念底座/注意力机制.md", "title": "注意力机制", "kind": "concept", "content": "注意力机制给不同信息分配不同权重，聚焦关键部分。", "content_hash": "z"})
        result = add_interest(self.store, "摄影聚光", "摄影用聚光突出注意力和关键信息", ["摄影"])
        self.assertIsNotNone(result["link"])
        self.assertGreaterEqual(self.store.stats()["links"], 1)
        semantic = next(edge for edge in self.store.graph()["edges"] if edge["relation"] == "semantic")
        self.assertEqual(semantic["status"], "proposed")
        self.assertTrue(self.store.review_link(semantic["id"], True))
        reviewed = next(edge for edge in self.store.graph()["edges"] if edge["id"] == semantic["id"])
        self.assertEqual(reviewed["status"], "accepted")

    def test_web_content_writes_to_vault_without_duplicate_node(self):
        vault = self.root / "vault"
        vault.mkdir()
        self.store.set_setting("vault_path", str(vault))
        result = add_interest(self.store, "音乐里的周期", "鼓点通过重复形成周期结构", ["音乐", "周期"])
        output = vault / "wiki" / "03-交叉火花" / "音乐里的周期.md"
        self.assertTrue(output.is_file())
        matching = [note for note in self.store.list_notes() if note["title"] == "音乐里的周期"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(result["note_id"], matching[0]["id"])

    def test_vault_change_and_delete_are_reflected_in_garden(self):
        vault = self.root / "vault"
        vault.mkdir()
        note = vault / "外部更新.md"
        note.write_text("# 外部更新\n\n第一版", encoding="utf-8")
        sync_vault(vault, self.store)
        self.assertEqual(self.store.list_notes()[0]["content"], "# 外部更新\n\n第一版")
        note.write_text("# 外部更新\n\n第二版", encoding="utf-8")
        sync_vault(vault, self.store)
        self.assertIn("第二版", self.store.list_notes()[0]["content"])
        note.unlink()
        result = sync_vault(vault, self.store)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(self.store.list_notes(), [])

    def test_pdf_pages_are_hidden_and_replaced_by_math_branches(self):
        self.store.upsert_note({
            "path": "pdf::calculus#page=1", "title": "高等微积分讲义 · 第 1 页", "kind": "textbook",
            "content": "极限与连续、导数和微分、中值定理、定积分与无穷级数。",
            "source": "pdf", "source_url": "calculus.pdf", "content_hash": "pdf-1",
        })
        count = rebuild_domain_map(self.store)
        titles = {node["title"] for node in self.store.graph()["nodes"]}
        self.assertGreaterEqual(count, 3)
        self.assertIn("数学", titles)
        self.assertIn("数学分析与微积分", titles)
        self.assertNotIn("高等微积分讲义 · 第 1 页", titles)
        mindmap = build_mindmap(self.store)["tree"]
        self.assertEqual(mindmap["title"], "我的知识花园")
        math_branch = next(child for child in mindmap["children"] if child["title"] == "数学")
        self.assertEqual(math_branch["children"][0]["title"], "数学分析与微积分")

        def all_titles(node):
            return [node["title"], *[title for child in node.get("children", []) for title in all_titles(child)]]

        self.assertFalse(any("第 1 页" in title for title in all_titles(mindmap)))

    def test_textbook_directory_skips_unchanged_pdf_on_later_scans(self):
        from pypdf import PdfWriter

        textbook_dir = self.root / "all-textbooks" / "math"
        textbook_dir.mkdir(parents=True)
        pdf_path = textbook_dir / "sample.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        first = ingest_pdf_directory(self.root / "all-textbooks", self.store)
        second = ingest_pdf_directory(self.root / "all-textbooks", self.store)
        self.assertEqual(first["files"], 1)
        self.assertEqual(first["processed"], 1)
        self.assertEqual(second["processed"], 0)
        self.assertEqual(second["skipped"], 1)

    def test_review_requires_answer_and_schedules_next_interval(self):
        result = analyze_frontier(self.store, "梯度下降", "梯度下降通过损失函数的梯度更新模型参数。")
        quiz = next(task for task in self.store.list_tasks() if task["task_type"] == "quiz")
        assessment = evaluate_review(quiz, quiz["payload"]["answer"])
        review = self.store.record_review(quiz["id"], assessment["quality"], assessment["feedback"], str(quiz["payload"]["answer"]))
        self.assertTrue(assessment["correct"])
        self.assertGreaterEqual(review["next_interval_days"], 4)
        original = self.store.get_task(quiz["id"])
        self.assertEqual(original["status"], "done")
        next_reviews = [
            task for task in self.store.list_tasks()
            if task["task_type"] == "quiz" and task["title"] == quiz["title"]
        ]
        self.assertEqual(len(next_reviews), 1)

    def test_inspiration_is_l1_only_and_unanchored_fact_is_downgraded(self):
        generated = {
            "primary_type": "counterfactual", "secondary_types": [],
            "acknowledgement": "这是一个值得展开的假设。", "assumptions": ["人类可以光合作用"],
            "claims": [
                {"status": "fact", "text": "没有来源支持的现实断言", "anchor_index": None},
                {"status": "imagination", "text": "城市可能围绕日照设计", "anchor_index": None},
            ],
            "counter_view": "能量效率可能限制这个设定。",
            "branches": [{"title": "机制", "question": "最小机制是什么？"}],
        }
        with patch("core.inspiration.chat_json", return_value=generated):
            result = explore_inspiration(self.store, "如果人类能光合作用会怎样？")
        self.assertEqual(result["claims"][0]["status"], "uncertain")
        with self.store.connect() as conn:
            event = conn.execute(
                "SELECT surface,event_type,payload_json FROM learning_events WHERE event_id=?",
                (result["event_id"],),
            ).fetchone()
            mastery_count = conn.execute("SELECT COUNT(*) n FROM concept_mastery").fetchone()["n"]
        self.assertEqual(event["surface"], "inspiration")
        self.assertEqual(event["event_type"], "inspiration_turn")
        self.assertEqual(mastery_count, 0)
        reflected = LearningMemoryService(self.store).reflect(force=True, min_events=0)
        self.assertEqual(reflected["l2_created"], 0)
        self.assertEqual(reflected["l3_created"], 0)

    def test_inspiration_is_not_written_until_explicit_save(self):
        before = len(self.store.list_notes())
        result = explore_inspiration(self.store, "我感觉颜色和情绪之间有某种映射")
        self.assertEqual(len(self.store.list_notes()), before)
        saved = save_inspiration_seed(
            self.store, "颜色与情绪映射",
            [{"role": "user", "content": "我感觉颜色和情绪之间有某种映射"}], result,
        )
        note = self.store.get_note(saved["note_id"])
        self.assertIn("eligible_for_factual_retrieval: false", note["content"])
        self.assertIn("未核验", note["tags"])

    def test_frontier_guided_reading_carries_source_material_into_gardener(self):
        question = (
            "请带我导读这篇神经科学与诗歌研究。\n\n<frontier_material>\n"
            "title: A neuroscientific stance on poetry\n"
            "url: https://doi.org/10.1000/example\n"
            "authors: A Researcher\nvenue: Test Journal\nyear: 2026\n"
            "access_scope: abstract\nabstract:\n"
            "This neuroscience study discusses poetry, emotion, clinical applications and evidence boundaries.\n"
            "</frontier_material>"
        )
        result = answer_from_wiki(self.store, question)
        self.assertEqual(result["evidence_layer"], "authority")
        self.assertEqual(result["web_sources"][0]["title"], "A neuroscientific stance on poetry")
        self.assertEqual(result["web_sources"][0]["access_scope"], "abstract")
        self.assertIn("[M1]", result["answer"])

    def test_material_classification_requires_body_evidence(self):
        body = (
            "参照群体效应会改变跨文化问卷分数的比较基准，因此相同分数未必具有相同心理含义。"
            "跨文化心理学需要区分真实的群体差异与测量过程产生的差异。研究者通常还会检查量表的测量不变性，"
            "并结合行为数据与访谈材料判断问卷分数代表的实际心理过程。这个机制提醒我们，比较之前必须先说明"
            "参照标准、适用人群和测量边界，不能把一个量表数字直接当成脱离文化语境的心理事实。"
        )
        generated = {
            "main_claim": "比较基准会影响跨文化测量解释",
            "discipline": "心理学", "topic": "社会与文化心理学",
            "classification_evidence": ["参照群体效应会改变跨文化问卷分数的比较基准"],
            "concepts": [{
                "name": "参照群体效应",
                "evidence": "参照群体效应会改变跨文化问卷分数的比较基准",
                "role": "mechanism",
            }],
            "confidence": 0.91,
        }
        with patch("core.engine.chat_json", return_value=generated):
            result = analyze_material_structure("被误解的集体主义文化", body)
        self.assertEqual(result["discipline"], "心理学")
        self.assertEqual(result["topic"], "社会与文化心理学")
        self.assertEqual(result["method"], "langchain")

        generated["classification_evidence"] = ["正文里根本不存在的证据"]
        with patch("core.engine.chat_json", return_value=generated):
            rejected = analyze_material_structure("随机公众号", body)
        self.assertNotEqual(rejected["method"], "langchain")

    def test_textbook_classifier_adds_new_disciplines_without_hardcoded_map(self):
        structure = classify_textbook_structure(
            "基础工程电路分析",
            "This chapter introduces circuit voltage current and Kirchhoff laws. 后续讨论交流阻抗。",
        )
        self.assertEqual(structure["disciplines"][0]["name"], "电气与电子工程")
        self.assertEqual(structure["disciplines"][0]["branches"][0]["name"], "电路与系统")


if __name__ == "__main__":
    unittest.main()
