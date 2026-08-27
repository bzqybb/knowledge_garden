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
from core.deepdiagram_service import _parse_mermaid, _structure_comparison_graph
from core.engine import (
    add_interest,
    analyze_frontier,
    analyze_material_structure,
    article_preview_metadata,
    evaluate_review,
    extract_concepts,
)
from core.gardener_graph import (
    _comparison_subjects,
    _discussion_depth_guidance,
    _deterministic_exact_iff_proof,
    _ensure_readable_paragraphs,
    _explicit_academic_concepts,
    _fallback_planner,
    _fallback_wechat_lookup,
    _question_subject,
    _response_profile,
    _requires_claim_level_audit,
    _requests_wechat_history,
    _scientific_premise_guard,
    _wechat_time_params,
    audit_evidence,
    choose_teaching_strategy,
    generate_answer,
    generate_deliverables,
    generate_visualization,
    plan_sources,
    planner_plan,
    retrieve_sources,
    route_after_planner,
    understand_question,
    repair_outputs,
    review_answer,
)
from core.inspiration import explore_inspiration, save_inspiration_seed
from core.mindmap import build_mindmap
from core.learning_memory import LearningMemoryService
from core.llm import LLMError
from core.obsidian import sync_vault
from core.query_understanding import build_query_plan
from core.retrieval import classify_textbook_structure, ingest_pdf_directory, rebuild_domain_map, search_notes
from core.storage import GardenStore
from core.taxonomy import classify_unmounted_concepts, rebuild_concept_hierarchy
from core.tracememo import TraceMemoConfig, TraceMemoClient, normalize_message, tracememo_config
from core.web_research import _WeChatArticleParser, search_academic_articles, search_public_web


class GardenTests(unittest.TestCase):
    def test_diagnose_reasoning_error_is_not_misclassified_as_medical(self):
        self.assertEqual(_response_profile("请诊断这个热力学推导的错误。"), "grounded_knowledge")
        self.assertEqual(_response_profile("请诊断患者持续疼痛的原因。"), "health_guidance")
        wrapped = (
            "【致理结构调试·develop】规则提醒：患者症状与疼痛需要医疗边界。\n"
            "题目：\n错误回答：‘未观察到反应，所以 ΔG 一定大于零。’请诊断并修正。"
        )
        self.assertEqual(_response_profile(wrapped), "grounded_knowledge")

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
            {"m_nsUsrName": "gh_abc", "m_nsNickName": "知识号", "name": "错误名称", "isOfficialAccount": True},
            {"m_nsUsrName": "gh_6ac216e2b856@app", "m_nsNickName": "gh_6ac216e2b856@app", "isOfficialAccount": True},
        ]}):
            accounts = client.official_accounts()
        self.assertEqual(accounts["count"], 1)
        self.assertEqual(accounts["unresolved_count"], 1)
        self.assertEqual(accounts["items"][0]["m_nsNickName"], "知识号")
        self.assertEqual(accounts["items"][0]["display_name"], "知识号")
        self.assertEqual(accounts["items"][0]["account_id"], "gh_abc")

    def test_tracememo_official_articles_preserve_real_account_nickname(self):
        client = TraceMemoClient(TraceMemoConfig("http://127.0.0.1:6131", "token", False))
        message = normalize_message({
            "id": "article-1", "from": "gh_abc", "createTime": 1756798668,
            "contentData": {
                "title": "一篇文章", "url": "https://mp.weixin.qq.com/s/example",
                "appname": "gh_abc",
            },
        })
        with patch.object(client, "current_time", return_value={"now": "2026-08-25T09:30:00+08:00"}), \
             patch.object(client, "chatlog", return_value={"messages": [message], "truncated": False}):
            result = client.official_articles("gh_abc", contact={
                "m_nsUsrName": "gh_abc", "m_nsNickName": "中文公众号", "name": "不正确的名称",
            })
        self.assertEqual(result["account_name"], "中文公众号")
        self.assertEqual(result["articles"][0]["sender"], "中文公众号")
        self.assertEqual(result["articles"][0]["article"]["publisher"], "中文公众号")
        self.assertEqual(result["articles"][0]["article"]["account_name"], "中文公众号")

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
            correction = conn.execute(
                """SELECT claim_id,confidence,status FROM memory_claims
                   WHERE dimension='teaching_preference' AND scope_type='task'
                     AND scope_key='explain_mechanism' AND claim_text='我想直接看推导'"""
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
        self.assertIsNotNone(correction)
        self.assertAlmostEqual(correction["confidence"], 0.86)
        self.assertEqual(correction["status"], "active")
        self.assertEqual(result["created_claim_id"], correction["claim_id"])

    def test_feedback_on_standard_answer_is_applied_to_next_same_task(self):
        with self.store.connect() as conn:
            conn.execute("INSERT INTO sessions(session_id,title) VALUES('standard-session','标准讲解')")
            metadata = json.dumps({"personalization": {
                "status": "standard", "task_key": "define",
                "strategy_summary": "标准讲解（没有足够个性化证据）", "applied_claim_ids": [],
            }}, ensure_ascii=False)
            conn.execute(
                """INSERT INTO session_messages(
                       message_id,session_id,request_id,role,content,metadata_json
                   ) VALUES(?,?,?,?,?,?)""",
                ("standard-answer", "standard-session", "standard-request", "assistant", "回答", metadata),
            )
        memory = LearningMemoryService(self.store)
        feedback = memory.record_personalization_feedback(
            request_id="standard-request", helpful=False,
            feedback_note="先用几何直觉建立图景，再给代数定义和推导，并配一个具体例子。",
        )
        recalled = memory.active_memory_context([], task_keys=["define"])

        self.assertTrue(feedback["recorded"])
        self.assertIsNotNone(feedback["created_claim_id"])
        self.assertEqual(len(recalled["claims"]), 1)
        self.assertIn("几何直觉", recalled["claims"][0]["claim_text"])
        self.assertGreaterEqual(recalled["claims"][0]["effective_confidence"], 0.8)

    def test_confirmed_preference_survives_teaching_strategy_agent(self):
        state = {
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "learner_context": {"concept_mastery": []},
            "personalization_plan": {
                "status": "applied", "task_key": "define", "confidence": 0.86,
                "hypotheses": [{
                    "claim": "先用几何直觉建立图景，再给代数定义和推导，并配一个具体例子。",
                    "claim_id": "claim-geometry", "evidence_ids": ["event-geometry"],
                }],
                "evidence": [{"evidence_id": "event-geometry"}],
                "applied_claim_ids": ["claim-geometry"],
                "allowed_adjustments": ["调整解释顺序"],
            },
            "planner_decision": {"complexity": "simple"},
            "dialogue": "", "evidence_review": {"gaps": []}, "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value=None):
            result = choose_teaching_strategy(state)
        strategy = result["teaching_strategy"]
        self.assertEqual(strategy["preference_directives"], [
            "先用几何直觉建立图景，再给代数定义和推导，并配一个具体例子。"
        ])
        self.assertEqual(strategy["explanation_order"], [
            "几何或空间直觉", "严格定义", "逐步推导", "具体例子检验",
        ])
        self.assertEqual(strategy["applied_evidence_ids"], ["event-geometry"])

    def test_reflector_flags_unexecuted_confirmed_preference(self):
        state = {
            "question": "什么是矩阵的秩？",
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "planner_decision": {
                "complexity": "simple", "primary_modality": "text", "max_revisions": 1,
            },
            "evidence_review": {
                "sufficient": True, "source_roles": {"L1": "direct_evidence"},
            },
            "accepted_sources": [{"source_id": "L1"}],
            "answer": "矩阵的秩是其列空间的维数，它可以通过初等行变换计算。[L1]",
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "teaching_strategy": {"preference_directives": [
                "先用几何直觉建立图景，再给代数定义和推导，并配一个具体例子。"
            ]},
            "trace": [],
        }
        with patch("core.gardener_graph.chat_json", return_value=None):
            result = review_answer(state)
        review = result["quality_review"]
        self.assertFalse(review["passed"])
        self.assertFalse(review["personalization_natural"])
        self.assertTrue(any("具体例子" in issue for issue in review["issues"]))

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

    def test_l3_profile_is_an_evidence_graph_but_does_not_bias_problem_understanding(self):
        with self.store.connect() as conn:
            for index, scope in enumerate(("define", "apply", "compare"), start=1):
                session_id = f"l3-session-{index}"
                event_id = f"l3-event-{index}"
                l2_id = f"l2-source-{index}"
                conn.execute("INSERT INTO sessions(session_id,title) VALUES(?,?)", (session_id, scope))
                conn.execute(
                    """INSERT INTO learning_events(
                           event_id,session_id,surface,event_type,source_kind,payload_json
                       ) VALUES(?,?,?,?,?,?)""",
                    (event_id, session_id, "gardener_chat", "personalization_feedback", "explicit", "{}"),
                )
                conn.execute(
                    """INSERT INTO memory_claims(
                           claim_id,layer,dimension,scope_type,scope_key,claim_text,
                           source_kind,confidence,status
                       ) VALUES(?,2,'teaching_preference','task',?,?,'explicit',0.9,'active')""",
                    (l2_id, scope, "复杂问题先给结构再展开机制"),
                )
                conn.execute(
                    "INSERT INTO memory_claim_evidence(claim_id,event_id,weight) VALUES(?,?,1.0)",
                    (l2_id, event_id),
                )
            conn.execute(
                """INSERT INTO memory_claims(
                       claim_id,layer,dimension,scope_type,scope_key,claim_text,
                       source_kind,confidence,status
                   ) VALUES('l3-structure',3,'teaching_preference','global','',
                            '复杂问题先给结构再展开机制','inferred',0.86,'active')"""
            )
            for index in range(1, 4):
                conn.execute(
                    """INSERT INTO memory_claim_evidence(
                           claim_id,source_claim_id,relation,weight
                       ) VALUES('l3-structure',?,'supports',1.0)""",
                    (f"l2-source-{index}",),
                )

        graph = LearningMemoryService(self.store).l3_profile_graph()
        node_types = {item["type"] for item in graph["nodes"]}
        relations = {item["relation"] for item in graph["edges"]}
        self.assertIn("l3_pattern", node_types)
        self.assertIn("l2_hypothesis", node_types)
        self.assertIn("context", node_types)
        self.assertIn("generalized_from", relations)
        self.assertEqual(graph["applicable_patterns"][0]["claim_id"], "l3-structure")

        captured = {}

        def fake_understanding_agent_json(system, user, **kwargs):
            captured["system"] = system
            captured["prompt"] = user
            return ({
                "primary_intent": "explain_mechanism", "concepts": ["注意力机制"],
                "task_demand": "analyze", "possible_obstacle": "causal_gap",
                "evidence": "当前问题要求解释机制。", "core_question": "为什么注意力机制有效？",
                "profile_graph_claim_ids_used": ["l3-structure", "invented-claim"],
                "profile_graph_rationale": "用已验证的结构偏好消解讲解顺序。",
            }, "glm:test")

        with patch("core.gardener_graph._understanding_agent_json", side_effect=fake_understanding_agent_json):
            understood = understand_question({
                "store": self.store, "question": "为什么注意力机制有效？",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertNotIn("l3-structure", captured["prompt"])
        self.assertIn("不读取或推断学习画像", captured["system"])
        self.assertIn("例：", captured["system"])
        self.assertEqual(understood["intent"]["profile_graph_claim_ids_used"], [])
        self.assertEqual(understood["trace"][-1]["data"]["understanding_provider"], "glm:test")
        self.assertEqual(understood["profile_graph"]["applicable_patterns"][0]["claim_id"], "l3-structure")

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

    def test_full_deepdiagram_mermaid_parser_keeps_declared_labels(self):
        code = '''flowchart LR
            CQ["Compare Alpha and Beta<br/>[core question]"]
            CQ --> A["Uses Method One [W1]"]
            CQ --> B["Uses Method Two [W2]"]
        </code'''
        nodes, edges = _parse_mermaid(code)
        labels = {item["id"]: item["label"] for item in nodes}
        self.assertEqual(labels["A"], "Uses Method One")
        self.assertEqual(labels["B"], "Uses Method Two")
        evidence = {item["id"]: item["evidence_ids"] for item in nodes}
        self.assertEqual(evidence["A"], ["W1"])
        self.assertEqual(evidence["B"], ["W2"])
        self.assertIn("Compare Alpha and Beta", labels["CQ"])
        self.assertEqual({item["source"] for item in edges}, {"CQ"})

    def test_full_comparison_graph_restores_two_subject_anchors(self):
        nodes, edges = _structure_comparison_graph(
            [
                {"id": "CQ", "label": "Compare Alpha and Beta · [core question]", "role": "concept", "evidence_ids": []},
                {"id": "A", "label": "Uses Method One", "role": "concept", "evidence_ids": ["W1"]},
                {"id": "B", "label": "Uses Method Two", "role": "concept", "evidence_ids": ["W2"]},
            ],
            [{"source": "CQ", "target": "A", "label": ""}],
            {
                "core_question": "Compare Alpha and Beta",
                "comparison_subjects": ["Alpha", "Beta"],
                "evidence_items": [
                    {"source_id": "W1", "excerpt": "Alpha uses Method One"},
                    {"source_id": "W2", "excerpt": "Beta uses Method Two"},
                ],
            },
        )
        self.assertEqual([item["label"] for item in nodes[:2]], ["Alpha", "Beta"])
        self.assertEqual([item["role"] for item in nodes[:2]], ["anchor", "anchor"])
        self.assertTrue(all(item["label"] != "Compare Alpha and Beta · [core question]" for item in nodes))
        self.assertEqual({item["label"] for item in edges}, {"比较维度"})

    def test_full_comparison_graph_reuses_existing_subject_nodes(self):
        nodes, edges = _parse_mermaid('''graph LR
            Q["Compare Synthetic Gamma &amp; Delta"]
            Q --> G["Gamma"]
            Q --> D["Delta"]
            G --> GP["Pattern One [W1]"]
            D --> DP["Pattern Two [W2]"]
        </code''')
        nodes, edges = _structure_comparison_graph(
            nodes, edges,
            {
                "core_question": "Compare synthetic Gamma and Delta",
                "comparison_subjects": ["Gamma", "Delta"],
                "evidence_items": [
                    {"source_id": "W1", "excerpt": "Gamma uses Pattern One"},
                    {"source_id": "W2", "excerpt": "Delta uses Pattern Two"},
                ],
            },
        )
        self.assertEqual([item["label"] for item in nodes].count("Gamma"), 1)
        self.assertEqual([item["label"] for item in nodes].count("Delta"), 1)
        self.assertNotIn("Compare Synthetic Gamma & Delta", [item["label"] for item in nodes])
        self.assertIn("W1", next(item for item in nodes if item["label"] == "Pattern One")["evidence_ids"])

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

    def test_subjective_question_can_be_discussed_without_fabricated_citations(self):
        state = {
            "question": "为什么我来到清华之后反而更迷茫了？",
            "intent": {"primary_intent": "evaluate", "response_mode": "standard"},
            "evidence_review": {"sufficient": False, "gaps": ["没有直接证据"]},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "进入新的环境后感到迷茫，并不等于你选错了路。选择变多、比较压力变强，都可能让原来确定的目标重新摇晃。",
            "followup": "你觉得最明显的变化是什么？", "discussion_prompts": [],
        }) as model:
            result = generate_answer(state)
        self.assertIn("进入新的环境", result["answer"])
        self.assertNotIn("证据不足", result["answer"])
        self.assertEqual(result["generation_sources"], [])
        self.assertIn("不要强制套用统一框架", model.call_args.args[0])
        self.assertIn("由本题内容决定的加粗小标题", model.call_args.args[0])
        self.assertEqual(result["trace"][-1]["data"]["response_profile"], "reflective_discussion")

    def test_health_question_keeps_common_sense_and_adds_medical_boundary(self):
        state = {
            "question": "为什么我运动完之后右腿比左腿疼得更厉害？",
            "intent": {"primary_intent": "explain_mechanism", "response_mode": "standard"},
            "evidence_review": {"sufficient": False, "gaps": ["没有直接证据"]},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "左右腿负荷、发力习惯和运动前状态不同，都可能导致运动后的酸痛程度不一致。",
        }):
            result = generate_answer(state)
        self.assertIn("不能替代医生面诊", result["answer"])
        self.assertIn("及时就医", result["answer"])
        self.assertNotIn("一定是", result["answer"])

    def test_open_discussion_removes_unsourced_statistics_and_invented_nicknames(self):
        state = {
            "question": "猫和狗谁更聪明？",
            "intent": {"primary_intent": "evaluate", "response_mode": "standard"},
            "evidence_review": {"sufficient": False, "gaps": ["没有直接证据"]},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "猫和狗适应环境的方式不同。研究表明狗有5.3亿个神经元。不能用一种能力给所有动物排名。",
        }):
            result = generate_answer(state)
        self.assertIn("适应环境", result["answer"])
        self.assertNotIn("5.3亿", result["answer"])
        self.assertNotIn("研究表明", result["answer"])

    def test_discussion_depth_expands_ambiguous_concepts_without_imposing_theory_everywhere(self):
        reflective = _discussion_depth_guidance("清华大学真的能给人带来幸福吗？", "reflective_discussion")
        playful = _discussion_depth_guidance("猫和狗谁更聪明？", "reflective_discussion")
        self.assertIn("生活满意度", reflective)
        self.assertIn("意义感", reflective)
        self.assertIn("3~5 段", reflective)
        self.assertIn("不为凑字数或学术感额外塞理论", playful)

    def test_long_discussion_is_split_into_natural_paragraphs_without_template_headings(self):
        sentence = "幸福既可以指当下的情绪体验，也可以指对整体生活状况的满意，还可以涉及个人是否觉得自己的行动有意义。"
        answer = sentence * 5
        formatted = _ensure_readable_paragraphs(answer)
        self.assertIn("\n\n", formatted)
        self.assertNotIn("## 结论", formatted)
        self.assertEqual(formatted.replace("\n\n", ""), answer)

    def test_scientific_premise_guard_corrects_wrong_free_energy_and_missing_time_term(self):
        thermo = _scientific_premise_guard("推导恒温恒容条件下反应自发进行的判据 ΔG < 0。")
        poisson = _scientific_premise_guard("证明对于任意物理量 A，有 dA/dt={A,H}，请用泊松括号说明。")
        commuting = _scientific_premise_guard(
            "如果A和B是可交换的矩阵，那么它们是否一定有公共特征向量？请证明或给出反例。"
        )
        self.assertEqual(thermo["kind"], "thermodynamic_constraints")
        self.assertIn("亥姆霍兹", thermo["correction"])
        self.assertEqual(poisson["kind"], "explicit_time_dependence")
        self.assertIn("∂A/∂t", poisson["correction"])
        self.assertEqual(commuting["kind"], "commuting_matrices_field_condition")
        self.assertIn("复数域", commuting["correction"])
        self.assertIn("90°", commuting["correction"])
        self.assertIn("旋转矩阵", commuting["correction"])

    def test_scientific_premise_guard_does_not_reject_correctly_qualified_statement(self):
        self.assertIsNone(_scientific_premise_guard(
            "证明复数域上矩阵可对角化，当且仅当最小多项式无重根。",
        ))
        self.assertIsNone(_scientific_premise_guard(
            "对于不显含时间的可观测量 A，请用泊松括号证明 dA/dt={A,H}。",
        ))

    def test_scientific_premise_guard_identifies_units_rank_and_quantum_postulates(self):
        coulomb = _scientific_premise_guard("推导氢原子中电子在库仑势V(r) = -e²/r下的能量本征值Eₙ。")
        regression = _scientific_premise_guard(
            "给出线性回归的损失函数，并推导其闭式解θ = (XᵀX)⁻¹Xᵀy。",
        )
        quantum = _scientific_premise_guard(
            "推导：在量子力学中，一个自由粒子的波函数由薛定谔方程描述。请写出含时薛定谔方程。",
        )
        genetics = _scientific_premise_guard(
            "推导：在自然选择作用下，等位基因频率的变化率由什么决定？",
        )
        entropy = _scientific_premise_guard(
            "用信息论中的熵概念解释为什么生命系统可以维持低熵状态。",
        )
        self.assertEqual(coulomb["kind"], "coulomb_unit_convention")
        self.assertIn("4πε₀", coulomb["correction"])
        self.assertIn("幂级数截断", coulomb["correction"])
        self.assertEqual(regression["kind"], "least_squares_invertibility")
        self.assertIn("满列秩", regression["correction"])
        self.assertIn("正规方程", regression["correction"])
        self.assertEqual(quantum["kind"], "quantum_dynamics_postulate")
        self.assertIn("公设", quantum["correction"])
        self.assertIn("iℏ∂ψ/∂t", quantum["correction"])
        self.assertEqual(genetics["kind"], "population_genetics_model_assumptions")
        self.assertIn("1−sq²", genetics["correction"])
        self.assertEqual(entropy["kind"], "information_and_thermodynamic_entropy")
        self.assertIn("香农信息熵", entropy["correction"])

    def test_scientific_premise_guard_preserves_already_qualified_statements(self):
        self.assertIsNone(_scientific_premise_guard(
            "在高斯单位制中，推导氢原子的库仑势 V(r)=-e²/r。",
        ))
        self.assertIsNone(_scientific_premise_guard(
            "设 X 满列秩，推导线性回归最小二乘解 θ=(XᵀX)⁻¹Xᵀy。",
        ))
        self.assertIsNone(_scientific_premise_guard(
            "将含时薛定谔方程作为基本假设，推导自由粒子的波函数。",
        ))

    def test_missing_sources_still_correct_formal_question_premise(self):
        state = {
            "question": "推导恒温恒容条件下反应自发进行的判据 ΔG < 0。",
            "intent": {"primary_intent": "explain_mechanism", "response_mode": "standard"},
            "evidence_review": {"sufficient": False, "gaps": ["没有可靠教材正文"]},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        result = generate_answer(state)
        self.assertIn("亥姆霍兹", result["answer"])
        self.assertIn("恒温恒压", result["answer"])
        self.assertIn("证据不足", result["answer"])

    def test_comparison_subjects_ignore_followup_instructions(self):
        subjects = _comparison_subjects(
            {
                "primary_intent": "compare",
                "core_question": "可逆矩阵和可对角化矩阵是同一个概念吗？如果不同，请给出各自定义并说明它们的区别。",
                "concepts": ["可逆矩阵", "可对角化矩阵", "矩阵"],
            },
            "",
        )
        self.assertEqual(subjects, ["可逆矩阵", "可对角化矩阵"])

    def test_campus_food_answer_does_not_invent_famous_specific_dishes(self):
        state = {
            "question": "清华的食堂和北大的食堂哪个更好吃？",
            "intent": {"primary_intent": "evaluate", "response_mode": "standard"},
            "evidence_review": {"sufficient": False, "gaps": []},
            "retrieval_errors": [], "accepted_sources": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "口味没有统一标准。学一食堂的麻辣烫很有名。更适合亲自体验后再比较。",
        }):
            result = generate_answer(state)
        self.assertIn("没有统一标准", result["answer"])
        self.assertNotIn("麻辣烫", result["answer"])

    def test_open_discussion_rejects_unrelated_textbook_even_when_audit_overaccepts(self):
        state = {
            "question": "为什么我来到清华之后反而更迷茫了？",
            "intent": {"primary_intent": "explain_mechanism", "response_mode": "standard"},
            "evidence_review": {
                "sufficient": True, "source_roles": {"L1": "direct_evidence"},
                "usable_claims": ["高等代数中的线性关系"], "gaps": [],
            },
            "accepted_sources": [{
                "source_id": "L1", "title": "高等代数 · 第 37 页",
                "source_type": "textbook", "text": "向量空间中的线性关系。",
            }],
            "retrieval_errors": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "进入更大的环境后感到迷茫，可能来自目标变化、比较压力与选择变多。",
        }):
            result = generate_answer(state)
        self.assertNotIn("高等代数", result["answer"])
        self.assertNotIn("[L1]", result["answer"])
        self.assertEqual(result["accepted_sources"], [])
        self.assertFalse(result["evidence_review"]["sufficient"])

    def test_delivery_join_preserves_open_discussion_evidence_downgrade(self):
        state = {
            "question": "为什么我来到清华之后反而更迷茫了？",
            "intent": {"primary_intent": "explain_mechanism", "response_mode": "standard"},
            "planner_decision": {"primary_modality": "text"},
            "evidence_review": {
                "sufficient": True, "source_roles": {"L1": "direct_evidence"},
                "usable_claims": ["高等代数中的线性关系"], "gaps": [],
            },
            "accepted_sources": [{
                "source_id": "L1", "title": "高等代数 · 第 37 页",
                "source_type": "textbook", "text": "向量空间中的线性关系。",
            }],
            "retrieval_errors": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "进入更大的环境后感到迷茫，可能来自目标变化、比较压力与选择变多。",
        }):
            result = generate_deliverables(state)
        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertEqual(result["accepted_sources"], [])

    def test_open_discussion_planner_skips_extra_model_and_visualization(self):
        state = {
            "question": "内卷到底是谁的错？",
            "intent": {
                "primary_intent": "evaluate", "response_mode": "standard",
                "core_question": "内卷到底是谁的错", "secondary_intents": [],
            },
            "trace": [],
        }
        with patch("core.gardener_graph._agent_json") as model:
            result = planner_plan(state)
        model.assert_not_called()
        self.assertEqual(result["planner_decision"]["complexity"], "moderate")
        self.assertEqual(result["planner_decision"]["primary_modality"], "text")
        self.assertFalse(result["planner_decision"]["online_research"])
        self.assertEqual(result["trace"][-1]["data"]["planning_mode"], "bounded_discussion_fast_path")

    def test_open_discussion_skips_irrelevant_textbook_retrieval(self):
        question = "清华大学真的能给人带来幸福吗？"
        context = ContextBuilder(self.store).build(
            question, [], session_id="boundary-fast", request_id="boundary-request", message_id="boundary-message",
        )
        state = {
            "store": self.store,
            "context": context,
            "question": question,
            "intent": {"primary_intent": "evaluate"},
            "source_plan": {"source_types": ["local_wiki", "textbook"]},
            "planner_decision": {"online_research": False},
            "wechat_lookup": {"requested": False},
            "trace": [],
        }
        with patch("core.gardener_graph.search_notes") as search:
            result = retrieve_sources(state)
        search.assert_not_called()
        self.assertEqual(result["candidate_sources"], [])
        self.assertEqual(result["trace"][-1]["data"]["retrieval_mode"], "bounded_discussion_fast_path")

    def test_clear_campus_question_does_not_trigger_remote_understanding_or_clarification(self):
        state = {
            "store": self.store,
            "question": "在致理书院，怎么判断一个同学是真的喜欢学习还是在卷？",
            "dialogue": "", "history": [], "trace": [],
        }
        with patch("core.gardener_graph._understanding_agent_json") as model:
            result = understand_question(state)
        model.assert_not_called()
        self.assertFalse(result["intent"]["needs_clarification"])
        self.assertEqual(result["intent"]["primary_intent"], "evaluate")
        self.assertEqual(result["trace"][-1]["data"]["understanding_provider"], "deterministic-bounded-discussion")

    def test_plain_explain_concept_uses_definition_fast_path(self):
        state = {
            "store": self.store, "question": "请解释矩阵的秩。",
            "dialogue": "", "history": [], "trace": [],
        }
        with patch("core.gardener_graph._understanding_agent_json") as model:
            result = understand_question(state)
        model.assert_not_called()
        self.assertEqual(result["intent"]["primary_intent"], "define")
        self.assertEqual(result["intent"]["research_object"], "矩阵的秩")
        self.assertEqual(
            result["trace"][-1]["data"]["understanding_provider"],
            "deterministic-simple-definition",
        )

    def test_formal_proof_uses_deterministic_understanding_fast_path(self):
        state = {
            "store": self.store,
            "question": "证明矩阵可逆的充要条件是其行列式不为零。",
            "dialogue": "", "history": [], "trace": [],
        }
        with patch("core.gardener_graph._understanding_agent_json") as model:
            result = understand_question(state)
        model.assert_not_called()
        self.assertEqual(result["intent"]["primary_intent"], "apply")
        self.assertEqual(result["intent"]["task_demand"], "analyze")
        self.assertEqual(
            result["trace"][-1]["data"]["understanding_provider"],
            "deterministic-formal-operation",
        )

    def test_exact_invertibility_theorem_has_complete_verified_proof_fallback(self):
        answer = _deterministic_exact_iff_proof(
            "证明矩阵可逆的充要条件是其行列式不为零。",
            [{"source_id": "L1", "title": "高等代数", "text": "定理原文"}],
        )
        self.assertIn("[L1]", answer)
        self.assertIn("必要性", answer)
        self.assertIn("充分性", answer)
        self.assertIn(r"\det(A)\det(A^{-1})", answer)
        self.assertIn(r"\operatorname{adj}(A)", answer)

    def test_reflector_accepts_bounded_open_discussion_without_fake_sources(self):
        state = {
            "question": "清华大学真的能给人带来幸福吗？",
            "intent": {"primary_intent": "evaluate", "response_mode": "standard"},
            "planner_decision": {"complexity": "moderate", "primary_modality": "text", "max_revisions": 1},
            "evidence_review": {"sufficient": False, "source_roles": {}},
            "accepted_sources": [],
            "answer": "学校可以提供成长机会，却不能替任何人保证幸福；你的关系、选择空间和生活节奏同样重要。",
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "teaching_strategy": {}, "trace": [],
        }
        with patch("core.gardener_graph.chat_json") as model:
            result = review_answer(state)
        model.assert_not_called()
        self.assertTrue(result["quality_review"]["passed"])
        self.assertTrue(result["quality_review"]["boundary_appropriate"])
        self.assertTrue(result["quality_review"]["expression_natural"])

    def test_reflector_flags_mechanical_repeated_answer_framework(self):
        state = {
            "question": "什么是矩阵？",
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "planner_decision": {"complexity": "simple", "primary_modality": "text", "max_revisions": 1},
            "evidence_review": {"sufficient": True, "source_roles": {"L1": "direct_evidence"}},
            "accepted_sources": [{"source_id": "L1"}],
            "answer": "## 先说结论\n矩阵是一张数表。[L1]\n\n## 为什么\n它用行列组织数据。\n\n## 成立边界\n需要明确维度。",
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "teaching_strategy": {}, "trace": [],
        }
        with patch("core.gardener_graph.chat_json", return_value=None):
            result = review_answer(state)
        self.assertFalse(result["quality_review"]["passed"])
        self.assertFalse(result["quality_review"]["expression_natural"])
        self.assertTrue(any("机械" in issue for issue in result["quality_review"]["issues"]))

    def test_reflector_rejects_citation_identifier_that_was_never_retrieved(self):
        state = {
            "question": "什么是矩阵？",
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "planner_decision": {"complexity": "simple", "primary_modality": "text", "max_revisions": 1},
            "evidence_review": {"sufficient": True, "source_roles": {"L1": "direct_evidence"}},
            "accepted_sources": [{"source_id": "L1"}],
            "answer": "矩阵是按行和列排列的数表。[L1] 另一个不存在的来源也这样说。[P1]",
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "teaching_strategy": {}, "trace": [],
        }
        with patch("core.gardener_graph.chat_json", return_value=None):
            result = review_answer(state)
        self.assertFalse(result["quality_review"]["passed"])
        self.assertFalse(result["quality_review"]["evidence_bounded"])
        self.assertTrue(any("P1" in issue for issue in result["quality_review"]["issues"]))

    def test_factual_definition_cannot_be_silently_downgraded_to_local_only(self):
        state = {
            "question": "人因学是不是新兴学科？请说明它的定义和发展历史。",
            "intent": {
                "primary_intent": "define", "research_object": "人因学",
                "core_question": "人因学的定义和发展历史", "concepts": ["人因学"],
                "longitudinal_questions": ["人因学如何形成并发展？"],
                "response_mode": "standard",
            },
            "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "goal": "解释人因学", "complexity": "moderate", "required_steps": [],
            "primary_modality": "text", "relation_type": "none", "visual_kind": "none",
            "modality_reason": "文字即可", "visual_request": "", "online_research": False,
            "reflection_required": True, "max_revisions": 1, "stop_condition": "完成回答",
        }):
            planned = planner_plan(state)["planner_decision"]
        self.assertTrue(planned["online_research"])
        source_state = {**state, "planner_decision": planned, "history": []}
        source_plan = plan_sources(source_state)["source_plan"]
        self.assertIn("encyclopedia", source_plan["source_types"])
        self.assertIn("review", source_plan["source_types"])
        self.assertIn("research_paper", source_plan["source_types"])

    def test_question_subject_reduces_a_full_chinese_question_to_the_term(self):
        self.assertEqual(
            _question_subject("人因学是不是新兴学科？请说明它的定义、历史与核心议题。"),
            "人因学",
        )
        self.assertEqual(_question_subject("什么是人因学"), "人因学")
        self.assertEqual(_question_subject("艾颖华是谁"), "艾颖华")
        self.assertEqual(_question_subject("谁是艾颖华"), "艾颖华")

    def test_named_person_is_searched_before_possible_namesake_clarification(self):
        payload = {
            "primary_intent": "clarify",
            "research_object": "艾颖华",
            "target_kind": "person",
            "concepts": ["艾颖华"],
            "needs_clarification": True,
            "clarification_question": "您指的是哪个领域的艾颖华？",
            "ambiguities": ["可能存在同名人物"],
            "confidence": 0.82,
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            understood = understand_question({
                "store": self.store, "question": "艾颖华是谁",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertEqual(understood["intent"]["primary_intent"], "define")
        self.assertEqual(understood["intent"]["research_object"], "艾颖华")
        self.assertEqual(understood["intent"]["target_kind"], "person")
        self.assertFalse(understood["intent"]["needs_clarification"])
        self.assertEqual(route_after_planner({"intent": understood["intent"]}), "load_learner_memory")

    def test_clarification_answer_preserves_original_person_and_scope(self):
        history = [
            {"role": "user", "content": "艾颖华是谁"},
            {
                "role": "assistant",
                "content": "您指的是哪个领域的艾颖华？",
                "evidence_layer": "clarification",
            },
        ]
        payload = {
            "primary_intent": "clarify",
            "research_object": "学术界",
            "target_kind": "person",
            "core_question": "学术界中的哪位艾颖华",
            "concepts": ["学术界"],
            "needs_clarification": True,
            "clarification_question": "请进一步提供机构或研究领域。",
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            understood = understand_question({
                "store": self.store, "question": "学术界",
                "dialogue": "用户：艾颖华是谁\n园丁：您指的是哪个领域的艾颖华？",
                "history": history, "trace": [],
            })
        intent = understood["intent"]
        self.assertEqual(intent["research_object"], "艾颖华")
        self.assertIn("艾颖华", intent["core_question"])
        self.assertIn("学术界", intent["core_question"])
        self.assertIn("学术界", intent["explicit_constraints"])
        self.assertFalse(intent["needs_clarification"])
        self.assertIn("艾颖华", intent["query_plan"]["resolved"])
        self.assertIn("学术界", intent["query_plan"]["resolved"])

    def test_missing_referent_still_requests_genuine_clarification(self):
        payload = {
            "primary_intent": "clarify",
            "research_object": "",
            "target_kind": "unknown",
            "concepts": [],
            "needs_clarification": True,
            "clarification_question": "你说的是哪一个对象？",
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            understood = understand_question({
                "store": self.store, "question": "那个为什么这样",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertTrue(understood["intent"]["needs_clarification"])
        self.assertEqual(route_after_planner({"intent": understood["intent"]}), "clarify")

    def test_complete_personal_decision_is_not_misclassified_as_missing_referent(self):
        payload = {
            "primary_intent": "clarify",
            "research_object": "专业方向",
            "target_kind": "unknown",
            "concepts": [],
            "needs_clarification": True,
            "clarification_question": "请补充更多信息。",
        }
        question = (
            "我轮转了数学、化学和生物实验室，感觉每个方向都挺有意思，"
            "反而更不知道该选什么了。你能不能根据我这句话判断我最适合哪个方向？"
        )
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            understood = understand_question({
                "store": self.store, "question": question,
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertEqual(understood["reasoning_profile"]["key"], "decision_analysis")
        self.assertFalse(understood["intent"]["needs_clarification"])
        self.assertEqual(understood["intent"]["primary_intent"], "evaluate")
        self.assertEqual(route_after_planner({"intent": understood["intent"]}), "load_learner_memory")

    def test_public_search_ranks_institutional_pages_before_general_results(self):
        payload = b'''<?xml version="1.0" encoding="utf-8"?>
            <rss><channel>
              <item><title>General profile</title>
                <link>https://example.com/person</link>
                <description>Ada Example studies linear algebra.</description></item>
              <item><title>Ada Example - university faculty</title>
                <link>https://math.example.edu.cn/faculty/ada</link>
                <description>Ada Example is a mathematics professor.</description></item>
            </channel></rss>'''
        with patch("core.web_research.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = payload
            results = search_public_web("Ada Example", limit=4)
        self.assertEqual(results[0]["source_type"], "official_docs")
        self.assertTrue(results[0]["official"])
        self.assertEqual(results[1]["source_type"], "public_web")

    def test_person_lookup_retrieves_and_audits_official_public_page(self):
        question = "艾颖华是谁"
        intent = {
            "primary_intent": "define", "secondary_intents": [],
            "research_object": "艾颖华", "target_kind": "person",
            "core_question": question, "concepts": ["艾颖华"],
            "explicit_constraints": ["学术界"], "claim_to_verify": "",
            "response_mode": "standard",
        }
        with patch.dict(os.environ, {"GARDEN_DISABLE_NETWORK": "0"}):
            context = ContextBuilder(self.store).build(
                question, [], session_id="person-session", request_id="person-request",
                message_id="person-message",
            )
            state = {
                "store": self.store, "context": context, "question": question,
                "intent": intent, "history": [], "trace": [],
                "planner_decision": {"online_research": True, "complexity": "simple"},
            }
            source_state = {**state, **plan_sources(state)}
            self.assertIn("public_web", source_state["source_plan"]["source_types"])
            with patch("core.gardener_graph.search_wikipedia", return_value=[]), patch(
                "core.gardener_graph.search_public_web", return_value=[{
                    "title": "艾颖华 - 数学学院教师主页",
                    "url": "https://math.example.edu.cn/faculty/ai",
                    "abstract": "艾颖华是数学学院教师，研究方向包括应用数学与科学计算。",
                    "year": None, "authors": [], "venue": "math.example.edu.cn",
                    "source": "机构官网", "source_type": "official_docs", "official": True,
                }],
            ) as public_search:
                retrieval = retrieve_sources(source_state)
        public_search.assert_called_once()
        self.assertEqual(public_search.call_args.args[0], "艾颖华 学术界")
        self.assertIn("公开网页 / 机构官网", retrieval["retrieval_attempts"])
        result = audit_evidence({**source_state, **retrieval})
        self.assertTrue(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["P1"], "direct_evidence")

    def test_person_clarification_followup_completes_full_cited_answer(self):
        history = [
            {"role": "user", "content": "艾颖华是谁"},
            {
                "role": "assistant", "content": "您指的是哪个领域的艾颖华？",
                "evidence_layer": "clarification",
            },
        ]
        understanding = {
            "primary_intent": "clarify", "research_object": "学术界",
            "target_kind": "person", "concepts": ["学术界"],
            "needs_clarification": True,
            "clarification_question": "请进一步提供研究方向。",
        }
        source = {
            "title": "艾颖华 - 数学学院教师主页",
            "url": "https://math.example.edu.cn/faculty/ai",
            "abstract": "艾颖华是数学学院教师，研究方向包括应用数学与科学计算。",
            "year": None, "authors": [], "venue": "math.example.edu.cn",
            "source": "机构官网", "source_type": "official_docs", "official": True,
        }
        with patch.dict(os.environ, {"GARDEN_DISABLE_NETWORK": "0"}), patch(
            "core.gardener_graph._understanding_agent_json",
            return_value=(understanding, "glm:test"),
        ), patch("core.gardener_graph.search_wikipedia", return_value=[]), patch(
            "core.gardener_graph.search_public_web", return_value=[source],
        ):
            result = answer_from_wiki(self.store, "学术界", history)
        self.assertEqual(result["intent"]["research_object"], "艾颖华")
        self.assertEqual(result["evidence_layer"], "authority")
        self.assertTrue(result["researched_online"])
        self.assertIn("[P1]", result["answer"])
        self.assertEqual(result["web_sources"][0]["url"], source["url"])

    def test_simple_definition_uses_planner_fast_path_without_second_model_call(self):
        state = {
            "question": "什么是人因学",
            "intent": {
                "primary_intent": "define", "secondary_intents": [],
                "research_object": "人因学", "core_question": "人因学是什么",
                "claim_to_verify": "", "response_mode": "standard", "concepts": ["人因学"],
            },
            "trace": [],
        }
        with patch("core.gardener_graph._agent_json") as model:
            result = planner_plan(state)
        model.assert_not_called()
        self.assertEqual(result["planner_decision"]["complexity"], "simple")
        self.assertTrue(result["planner_decision"]["online_research"])
        self.assertEqual(result["trace"][-1]["data"]["planning_mode"], "deterministic_fast_path")

    def test_unconfigured_glm_uses_safe_millisecond_definition_parser(self):
        with patch("core.gardener_graph._understanding_agent_json") as model:
            result = understand_question({
                "store": self.store, "question": "什么是人因学",
                "dialogue": "", "history": [], "trace": [],
            })
        model.assert_not_called()
        self.assertEqual(result["intent"]["research_object"], "人因学")
        self.assertEqual(
            result["trace"][-1]["data"]["understanding_provider"],
            "deterministic-simple-definition",
        )

    def test_understanding_fallback_routes_numeric_foundation_question_as_apply(self):
        with patch("core.gardener_graph._understanding_agent_json", return_value=(
            None, "deterministic-fallback-after-glm-unavailable",
        )):
            result = understand_question({
                "store": self.store,
                "question": "一个三节点网络需要列写多少个线性独立的KCL方程？",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertEqual(result["intent"]["primary_intent"], "apply")
        self.assertEqual(result["intent"]["query_plan"]["subject_mode"], "foundational")

    def test_glm_fallback_keeps_definition_when_application_contains_design(self):
        with patch("core.gardener_graph._understanding_agent_json", return_value=(
            None, "deterministic-fallback-after-glm-unavailable",
        )):
            result = understand_question({
                "store": self.store,
                "question": "什么是手性分子？它对药物设计有什么影响？",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertEqual(result["intent"]["primary_intent"], "define")
        self.assertEqual(result["intent"]["research_object"], "手性分子")
        self.assertEqual(result["intent"]["concepts"], ["手性分子", "药物设计"])

    def test_understanding_enum_drift_does_not_discard_valid_subject(self):
        payload = {
            "primary_intent": "获取定义", "research_object": "人因学",
            "core_question": "人因学的基本定义", "concepts": ["人因学", "用户未提到的概念"],
            "task_demand": "用户希望了解定义", "possible_obstacle": "概念可能混淆",
            "needs_clarification": False, "confidence": 0.95,
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            result = understand_question({
                "store": self.store, "question": "请介绍人因学的基本定义",
                "dialogue": "", "history": [], "trace": [],
            })
        self.assertEqual(result["intent"]["primary_intent"], "define")
        self.assertEqual(result["intent"]["task_demand"], "understand")
        self.assertEqual(result["intent"]["research_object"], "人因学")
        self.assertEqual(result["intent"]["concepts"], ["人因学"])

    def test_standalone_question_keeps_original_query_when_agent_subject_drifts(self):
        question = "一个三节点网络需要列写多少个线性独立的KCL方程？"
        payload = {
            "primary_intent": "define",
            "research_object": "一阶微分方程",
            "core_question": "一阶微分方程是什么",
            "concepts": [],
            "needs_clarification": False,
            "confidence": 0.7,
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            result = understand_question({
                "store": self.store, "question": question,
                "dialogue": "", "history": [], "trace": [],
            })
        query_plan = result["intent"]["query_plan"]
        self.assertEqual(query_plan["resolved"], question)
        self.assertEqual(query_plan["queries"][0]["text"], question)

    def test_understanding_normalizes_descriptive_clarification_string(self):
        payload = {
            "primary_intent": "compare",
            "research_object": "人因学与工程心理学",
            "core_question": "人因学和工程心理学有什么区别",
            "concepts": ["人因学", "工程心理学"],
            "needs_clarification": "不需要，最近上文已明确指出人因学",
            "confidence": "0.9",
        }
        with patch("core.gardener_graph._understanding_agent_json", return_value=(payload, "glm:test")):
            result = understand_question({
                "store": self.store, "question": "它和工程心理学有什么区别？",
                "dialogue": "用户：我正在了解人因学。", "history": [
                    {"role": "user", "content": "我正在了解人因学。"},
                ], "trace": [],
            })
        self.assertFalse(result["intent"]["needs_clarification"])
        self.assertEqual(result["intent"]["research_object"], "人因学与工程心理学")

    def test_glm_failure_uses_narrow_contextual_pronoun_fallback(self):
        with patch("core.gardener_graph._understanding_agent_json", return_value=(
            None, "deterministic-fallback-after-glm-unavailable",
        )):
            result = understand_question({
                "store": self.store, "question": "它和工程心理学有什么区别？",
                "dialogue": "用户：我正在了解人因学。", "history": [
                    {"role": "user", "content": "什么是人因学？"},
                ], "trace": [],
            })
        self.assertEqual(result["intent"]["primary_intent"], "compare")
        self.assertEqual(result["intent"]["research_object"], "人因学与工程心理学")
        self.assertEqual(result["intent"]["concepts"], ["人因学", "工程心理学"])
        self.assertFalse(result["intent"]["needs_clarification"])

    def test_top_encyclopedia_canonical_title_can_define_short_alias(self):
        state = {
            "question": "什么是人因学",
            "intent": {
                "primary_intent": "define", "research_object": "人因学",
                "core_question": "什么是人因学", "claim_to_verify": "",
                "concepts": ["人因学"], "response_mode": "standard",
            },
            "planner_decision": {"complexity": "simple"},
            "source_plan": {"search_query": "人因学"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "W1", "title": "人因工程学",
                "text": "人因工程学研究人与环境及系统之间的相互作用，并使系统适合人的能力和限制。",
                "source_type": "encyclopedia", "authority": "orientation",
                "local": False, "knowledge_status": "grounded", "access_scope": "abstract",
                "retrieval_rank": 1,
            }],
            "trace": [],
        }
        result = audit_evidence(state)
        self.assertTrue(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["W1"], "direct_evidence")

    def test_evidence_failure_reports_whether_online_lookup_was_attempted(self):
        state = {
            "evidence_review": {"sufficient": False, "gaps": ["没有直接证据"]},
            "retrieval_attempts": ["Wikipedia", "OpenAlex/Crossref"],
            "retrieval_errors": ["OpenAlex:RuntimeError"],
            "accepted_sources": [], "trace": [],
        }
        result = generate_answer(state)
        self.assertIn("已发出联网查询", result["answer"])
        self.assertNotIn("已完成本地知识、教材入口和可用权威来源的检索", result["answer"])

    def test_generation_prioritizes_claim_specific_textbook_and_repairs_its_citation(self):
        state = {
            "question": "导数处处为零，为什么函数必定是常数？",
            "intent": {
                "primary_intent": "explain_mechanism", "response_mode": "standard",
                "concepts": ["闭区间连续", "导数处处为零"],
                "query_plan": {"constraints": [], "aliases": []},
            },
            "teaching_strategy": {"teaching_move": "repair_causal_chain"},
            "evidence_review": {
                "sufficient": True,
                "source_roles": {"L1": "direct_evidence", "L2": "direct_evidence"},
                "usable_claims": [], "gaps": [],
            },
            "accepted_sources": [
                {"source_id": "L1", "title": "连续函数的整体性质", "source_type": "textbook",
                 "text": "闭区间上的连续函数具有介值性质。",
                 "note": {"reranker_score": 0.99, "fusion_score": 0.1}},
                {"source_id": "L2", "title": "微分中值定理", "source_type": "textbook",
                 "text": "若导数处处为 0，则函数在整个区间是常值函数。",
                 "note": {"reranker_score": 0.12, "fusion_score": 0.07,
                          "channel_consensus_bonus": 0.014}},
            ],
            "retrieval_errors": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "根据中值定理，导数处处为零的函数是常数。",
            "followup": "", "discussion_prompts": [],
        }):
            result = generate_answer(state)
        self.assertEqual(result["generation_sources"][0]["source_id"], "L2")
        self.assertIn("[L2]", result["answer"])

    def test_source_priority_prefers_constraint_and_aliases_in_same_argument(self):
        from core.gardener_graph import _source_argument_priority

        aliases = ["linearly independent", "node voltage", "reference node", "N-1"]
        background = {
            "title": "Nodal exercises",
            "text": "KCL applies to this circuit. " + ("background " * 40)
                    + "N-1 linearly independent node voltage reference node",
            "note": {"reranker_score": 0.99},
        }
        direct = {
            "title": "Nodal analysis",
            "text": "Exactly N-1 linearly independent KCL equations determine every node voltage "
                    "with respect to the reference node.",
            "note": {"reranker_score": 0.1},
        }
        self.assertGreater(
            _source_argument_priority(direct, ["KCL"], aliases, []),
            _source_argument_priority(background, ["KCL"], aliases, []),
        )

    def test_simple_answer_repairs_missing_direct_citation_without_reflector_retry(self):
        state = {
            "question": "什么是人因学",
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "teaching_strategy": {"teaching_move": "direct_definition"},
            "evidence_review": {
                "sufficient": True, "source_roles": {"W1": "direct_evidence"},
                "usable_claims": [], "gaps": [],
            },
            "accepted_sources": [{
                "source_id": "W1", "title": "人因工程学", "source_type": "encyclopedia",
                "text": "人因工程学研究人与系统的交互。",
            }],
            "retrieval_errors": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "人因学研究人与系统之间的相互作用。",
            "followup": "", "discussion_prompts": [],
        }):
            result = generate_answer(state)
        self.assertIn("[W1]", result["answer"])
        self.assertTrue(result["trace"][-1]["data"]["citation_binding_repaired"])

    def test_answer_generation_removes_invented_source_identifiers(self):
        state = {
            "question": "什么是人因学",
            "intent": {"primary_intent": "define", "response_mode": "standard"},
            "teaching_strategy": {"teaching_move": "direct_definition"},
            "evidence_review": {
                "sufficient": True, "source_roles": {"W1": "direct_evidence"},
                "usable_claims": [], "gaps": [],
            },
            "accepted_sources": [{
                "source_id": "W1", "title": "人因工程学", "source_type": "encyclopedia",
                "text": "人因工程学研究人与系统的交互。",
            }],
            "retrieval_errors": [], "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "人因学研究人与系统之间的相互作用。[P1]",
        }):
            result = generate_answer(state)
        self.assertNotIn("[P1]", result["answer"])
        self.assertIn("[W1]", result["answer"])
        self.assertEqual(result["trace"][-1]["data"]["removed_invalid_citations"], ["P1"])

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

    def test_pendulum_derivation_does_not_promote_unrelated_periodic_function_page(self):
        state = {
            "question": "推导单摆的小角度运动方程，并求出其周期公式。",
            "intent": {
                "primary_intent": "explain_mechanism",
                "research_object": "单摆小角度运动方程",
                "core_question": "推导单摆的小角度运动方程和周期公式",
                "claim_to_verify": "", "concepts": ["单摆", "运动方程", "周期公式"],
                "query_plan": {"aliases": ["simple pendulum", "pendulum"]},
                "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "单摆 周期 运动方程 pendulum"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "L1", "title": "复变函数 · 第 58 页",
                "text": "指数函数和三角函数具有周期，满足复变函数的 Cauchy-Riemann 方程。" * 5,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"reranker_score": 0.93, "reranker_rank": 1},
            }],
            "trace": [],
        }
        result = audit_evidence(state)
        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["L1"], "prerequisite")

    def test_high_risk_audit_requires_a_supported_usable_claim(self):
        state = {
            "question": "写出快速排序算法的伪代码，并分析其时间复杂度。",
            "intent": {
                "primary_intent": "apply", "research_object": "快速排序",
                "core_question": "快速排序算法与时间复杂度", "claim_to_verify": "",
                "concepts": ["快速排序"], "query_plan": {"aliases": ["quick sort"]},
                "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "快速排序 quick sort"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "L1", "title": "算法教材 · 快速排序",
                "text": "本页只列出章节标题，没有给出伪代码或复杂度推导。" * 4,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"reranker_score": 0.92, "reranker_rank": 1},
            }],
            "trace": [],
        }
        model_decision = {
            "accepted_ids": ["L1"], "rejected": [], "usable_claims": [], "gaps": [],
            "sufficient": True, "rationale": "只有主题重合。",
            "source_roles": {"L1": "direct_evidence"},
        }
        with patch("core.gardener_graph._agent_json", return_value=model_decision):
            result = audit_evidence(state)

        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertIn("未形成可逐项核对的来源论断", "；".join(result["evidence_review"]["gaps"]))

    def test_standard_calculation_questions_require_claim_level_evidence(self):
        questions = (
            "一质点沿x轴运动，求t=2s时的速度和加速度。",
            "半径为R的均匀带电球面，求球内外的电场强度分布。",
            "理想气体等温膨胀，求气体对外做的功和吸收的热量。",
            "由波函数求振幅、频率、波长和波速。",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(_requires_claim_level_audit(question))

    def test_formal_matrix_terms_are_extracted_without_model_inference(self):
        self.assertEqual(
            _explicit_academic_concepts("证明矩阵可逆的充要条件是行列式不为零。"),
            ["可逆矩阵", "行列式不为零"],
        )
        self.assertEqual(
            _explicit_academic_concepts("求矩阵的特征值和特征向量。"),
            ["特征值", "特征向量"],
        )
        self.assertTrue(_requires_claim_level_audit(
            "辨析可逆矩阵与可对角化矩阵，并各举一个反例。"
        ))

    def test_iff_proof_query_decomposes_both_directions(self):
        plan = build_query_plan("证明一个矩阵可逆的充要条件是其行列式不为零。")
        self.assertEqual(plan["question_type"], "proof_or_derivation")
        self.assertEqual(plan["strategy"], "decompose")
        self.assertEqual(len(plan["queries"]), 3)
        joined = " ".join(item["text"] for item in plan["queries"])
        self.assertIn("必要性", joined)
        self.assertIn("充分性", joined)
        self.assertIn("伴随矩阵", joined)

    def test_exact_textbook_iff_statement_survives_audit_provider_failure(self):
        state = {
            "question": "证明矩阵可逆的充要条件是其行列式不为零。",
            "intent": {
                "primary_intent": "apply", "research_object": "矩阵可逆性",
                "core_question": "证明矩阵可逆的充要条件是其行列式不为零。",
                "concepts": ["可逆矩阵", "行列式不为零"], "claim_to_verify": "",
                "query_plan": {"aliases": [], "constraints": []}, "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "矩阵可逆 行列式不为零"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "L1", "title": "高等代数 · 可逆矩阵",
                "text": "定理：n阶方阵可逆的充要条件是它的行列式 @ 关 0。",
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "explicitly_selected": True, "note": {},
            }],
            "trace": [],
        }
        with patch("core.gardener_graph._agent_json", side_effect=LLMError("timeout")):
            result = audit_evidence(state)
        review = result["evidence_review"]
        self.assertTrue(review["sufficient"])
        self.assertEqual(review["proof_anchor_mode"], "exact_textbook_iff_statement")
        self.assertTrue(review["usable_claims"])
        self.assertEqual(review["source_roles"]["L1"], "direct_evidence")

    def test_relationship_audit_rejects_single_word_cross_domain_collision(self):
        state = {
            "question": "我在学量子力学的‘态叠加原理’。它和线性代数有什么关系？",
            "intent": {
                "primary_intent": "explain_mechanism", "research_object": "态叠加原理",
                "core_question": "态叠加原理和线性代数的关系",
                "concepts": ["态叠加原理", "线性代数"], "claim_to_verify": "",
                "query_plan": {"aliases": ["superposition"], "constraints": []},
                "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "quantum superposition linear algebra"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "L1", "title": "Basic Engineering Circuit Analysis",
                "text": "Superposition can be applied to a circuit with independent sources." * 8,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"reranker_score": 0.95, "reranker_rank": 1},
            }],
            "trace": [],
        }
        mistaken_review = {
            "accepted_ids": ["L1"], "rejected": [],
            "usable_claims": ["Superposition is available in linear systems."],
            "gaps": [], "sufficient": True, "rationale": "命中叠加一词。",
            "source_roles": {"L1": "direct_evidence"},
        }
        with patch("core.gardener_graph._agent_json", return_value=mistaken_review):
            result = audit_evidence(state)
        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertIn("核心概念", "；".join(result["evidence_review"]["gaps"]))

    def test_specialist_mechanisms_do_not_promote_unrelated_textbook_or_wiki(self):
        examples = (
            (
                "推导：在自然选择作用下，等位基因频率的变化率由什么决定？",
                "等位基因频率", "主动情境选择｜教材—前沿对照",
                "主动情境选择反映个体通过选择环境影响后续情境和发生频率。", "local_wiki",
            ),
            (
                "解释DNA双螺旋结构中，碱基对的互补配对如何保证遗传信息的精确传递。",
                "DNA遗传信息精确传递", "普通化学原理 · 第 468 页",
                "DNA由四种碱基组成，氢键能够维持双螺旋分子结构稳定。", "textbook",
            ),
            (
                "为什么酶能降低反应的活化能？请用过渡态理论解释。",
                "酶降低活化能", "普通化学原理 · 第 157 页",
                "反应需要跨越过渡态，活化能决定化学反应的快慢。", "textbook",
            ),
            (
                "推导氢原子中电子在库仑势V(r)=-e²/r下的能量本征值。",
                "氢原子能级", "普通化学原理 · 第 77 页",
                "Bohr 模型中氢原子的第一电离能是 13.6 eV。", "textbook",
            ),
        )
        for question, subject, title, text, source_type in examples:
            with self.subTest(question=question):
                state = {
                    "question": question,
                    "intent": {
                        "primary_intent": "explain_mechanism", "research_object": subject,
                        "core_question": question, "claim_to_verify": "",
                        "concepts": [subject], "query_plan": {"aliases": []},
                        "response_mode": "standard",
                    },
                    "planner_decision": {"complexity": "moderate"},
                    "source_plan": {"search_query": subject},
                    "wechat_lookup": {"requested": False},
                    "candidate_sources": [{
                        "source_id": "L1", "title": title, "text": text * 4,
                        "source_type": source_type, "authority": "local_textbook",
                        "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                        "note": {"reranker_score": 0.95, "reranker_rank": 1},
                    }],
                    "trace": [],
                }
                result = audit_evidence(state)
                self.assertFalse(result["evidence_review"]["sufficient"])
                self.assertEqual(result["evidence_review"]["source_roles"]["L1"], "prerequisite")

    def test_audit_accepts_strong_reranked_textbook_for_compositional_foundation_question(self):
        state = {
            "question": "一个三节点网络需要列写多少个线性独立的KCL方程？",
            "intent": {
                "primary_intent": "apply",
                "research_object": "三节点网络的线性独立KCL方程数量",
                "core_question": "三节点网络需要多少个线性独立KCL方程",
                "claim_to_verify": "",
                "concepts": [],
                "response_mode": "standard",
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "三节点网络 线性独立 KCL 方程"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "T1", "title": "电路分析教材 · 第 109 页",
                "text": "对于包含 N 个节点的网络，只需要列写 N-1 个线性独立的节点方程。" * 4,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"reranker_score": 0.88, "reranker_rank": 4},
            }],
            "trace": [],
        }
        result = audit_evidence(state)
        self.assertTrue(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["T1"], "direct_evidence")

    def test_audit_rejects_strong_reranker_without_auditable_term_overlap(self):
        state = {
            "question": "节点1相对于节点2的电压是多少？",
            "intent": {
                "primary_intent": "apply", "research_object": "节点间电压",
                "core_question": "节点1相对于节点2的电压", "claim_to_verify": "",
                "concepts": [], "response_mode": "standard",
                "query_plan": {"aliases": ["node voltage", "reference node"]},
            },
            "planner_decision": {"complexity": "moderate"},
            "source_plan": {"search_query": "节点电压 node voltage"},
            "wechat_lookup": {"requested": False},
            "candidate_sources": [{
                "source_id": "T1", "title": "山岳文学 · 第 157 页",
                "text": "地图用阴影和等高线表示山脉、坡度与地形。" * 8,
                "source_type": "textbook", "authority": "local_textbook",
                "local": True, "knowledge_status": "grounded", "access_scope": "full_text",
                "note": {"reranker_score": 0.98, "reranker_rank": 1},
            }],
            "trace": [],
        }
        result = audit_evidence(state)
        self.assertFalse(result["evidence_review"]["sufficient"])
        self.assertEqual(result["evidence_review"]["source_roles"]["T1"], "prerequisite")

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

    def test_comparison_fallback_draws_two_subjects_and_removes_vault_markup(self):
        diagram = build_local_diagram(
            {
                "research_object": "人因学与技术伦理的核心区别",
                "core_question": "人因学与技术伦理的核心区别是什么？",
                "comparison_subjects": ["人因学", "技术伦理"],
                "direct_source_ids": ["W1", "W2"],
                "evidence_items": [
                    {
                        "source_id": "W1", "title": "人因学",
                        "excerpt": "人因学研究人、任务、技术与环境构成的系统。",
                    },
                    {
                        "source_id": "W2", "title": "技术伦理",
                        "excerpt": "技术伦理关注技术设计与使用中的价值、责任和风险。",
                    },
                    {
                        "source_id": "W2", "title": "坏的元数据",
                        "excerpt": "# 被误解的集体主义文化 | 降维对照 > 来源：旧页面.md",
                    },
                ],
                "usable_claims": [], "gaps": [],
            },
            requested_kind="comparison",
            allowed_source_ids={"W1", "W2"},
            fallback_reason="full service unavailable",
        )
        labels = [node["label"] for node in diagram["nodes"]]
        self.assertEqual(diagram["status"], "ready")
        self.assertIn("人因学", labels)
        self.assertIn("技术伦理", labels)
        self.assertGreaterEqual(len(labels), 4)
        serialized = json.dumps(diagram, ensure_ascii=False)
        self.assertNotIn("降维对照", serialized)
        self.assertNotIn("来源：", serialized)
        self.assertNotIn(".md", serialized)
        self.assertTrue(any(edge["label"] != "对照" for edge in diagram["edges"]))

    def test_reflector_rejects_shallow_comparison_and_decorative_diagram(self):
        state = {
            "question": "人因学与技术伦理的核心区别是什么？",
            "intent": {
                "primary_intent": "compare", "response_mode": "standard",
                "concepts": ["人因学", "技术伦理"],
            },
            "planner_decision": {
                "complexity": "moderate", "primary_modality": "text_visual",
                "visual_kind": "comparison", "max_revisions": 1,
            },
            "evidence_review": {
                "sufficient": True,
                "source_roles": {"W1": "direct_evidence", "W2": "direct_evidence"},
            },
            "accepted_sources": [{"source_id": "W1"}, {"source_id": "W2"}],
            "answer": "人因学研究人与系统，技术伦理研究价值问题。[W1][W2]",
            "visualization": {
                "status": "ready", "kind": "comparison", "provider": "local",
                "title": "比较", "design_concept": "", "source_ids": ["W1"],
                "nodes": [
                    {"id": "a", "label": "人因学与技术伦理", "role": "anchor", "evidence_ids": []},
                    {"id": "b", "label": "# 旧页 | 降维对照 > 来源：材料.md", "role": "concept", "evidence_ids": ["W1"]},
                ],
                "edges": [{"source": "a", "target": "b", "label": "对照"}],
            },
            "teaching_strategy": {}, "trace": [],
        }
        with patch("core.gardener_graph._agent_json", return_value=None):
            result = review_answer(state)
        review = result["quality_review"]
        self.assertFalse(review["passed"])
        self.assertEqual(review["repair_target"], "both")
        self.assertTrue(any("比较回答过浅" in issue for issue in review["issues"]))
        self.assertTrue(any("Markdown" in issue for issue in review["issues"]))

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

    def test_daily_digest_prioritizes_explicit_professional_focus_and_explains_basis(self):
        self.store.set_setting("learning_level", "本科进阶")
        self.store.set_setting("interests", ["音乐"])
        self.store.set_setting("frontier_focus", "材料物理")
        article = {
            "title": "Emerging quantum materials", "url": "https://doi.org/focus", "year": 2026,
            "authors": ["A"], "venue": "Journal", "source": "OpenAlex", "abstract": "Quantum materials research.",
        }
        with patch("core.agent.search_academic_articles", return_value=[article]):
            result = daily_digest(self.store, force=True)
        self.assertIn("材料物理", result["profile"]["explicit"])
        self.assertIn("你主动填写的专业/当前重点", result["profile"]["basis"][0])
        self.assertIn("材料物理", result["chosen_directions"])

    def test_academic_search_falls_back_to_crossref_with_diagnostics(self):
        crossref_payload = {"message": {"items": [{
            "DOI": "10.1234/garden.2026", "title": ["A useful fallback article"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "container-title": ["Garden Journal"], "abstract": "<jats:p>Useful abstract.</jats:p>",
            "published-online": {"date-parts": [[2026, 8, 20]]},
            "is-referenced-by-count": 7,
        }]}}
        diagnostics = {}
        with patch("core.web_research._request_json_with_retry", side_effect=[OSError("OpenAlex down"), crossref_payload]):
            articles = search_academic_articles("learning", limit=2, diagnostics=diagnostics)
        self.assertEqual(articles[0]["source"], "Crossref / DOI")
        self.assertEqual(articles[0]["year"], 2026)
        self.assertEqual(diagnostics["provider"], "Crossref")
        self.assertTrue(diagnostics["degraded"])
        self.assertIn("OpenAlex", diagnostics["errors"][0])

    def test_daily_digest_does_not_cache_transient_empty_failure(self):
        self.store.set_setting("learning_level", "本科入门")
        self.store.set_setting("interests", ["AI"])
        with patch("core.agent.search_academic_articles", side_effect=RuntimeError("all providers unavailable")) as search:
            first = daily_digest(self.store, force=False)
            second = daily_digest(self.store, force=False)
        self.assertEqual(first["items"], [])
        self.assertIn("RuntimeError", first["message"])
        self.assertEqual(second["items"], [])
        self.assertEqual(search.call_count, 2)

    def test_daily_digest_deduplicates_same_title_with_different_urls(self):
        self.store.set_setting("interests", ["AI"])
        base = {
            "title": "The Same Research Article", "year": 2026, "authors": ["A"],
            "venue": "Journal", "source": "OpenAlex", "abstract": "Artificial intelligence cognitive science.",
        }
        with patch("core.agent.search_academic_articles", return_value=[
            {**base, "url": "https://doi.org/10.1/a"},
            {**base, "url": "https://openalex.org/W123"},
        ]):
            result = daily_digest(self.store, force=True)
        self.assertEqual(len(result["items"]), 1)

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

    def test_inspiration_uses_expanded_answer_instead_of_fixed_status_labels(self):
        answer = (
            "你问的可能不只是‘虚名有什么用’，而是一个人为什么把外界认可看得比自己的判断更重。\n\n"
            "一种解释是，名声和他人的评价会被当作获得认可的捷径；另一种解释是，群体中的模仿"
            "能暂时降低做决定的风险。这两种动机不一定同时出现，也不能替具体的人下定论。\n\n"
            "再换个角度看：如果没有观众、没有排名，也没有可以模仿的对象，这种行为是否还会出现？"
        )
        generated = {
            "answer": answer, "primary_type": "open_exploration", "secondary_types": [],
            "acknowledgement": "你在追问声誉和从众背后的动机。",
            "assumptions": ["名声可能具有社会意义"],
            "claims": [{"status": "inference", "text": "外界评价可能影响行为", "anchor_index": None}],
            "counter_view": "不同人可能有不同的动机。",
            "branches": [{"title": "群体里的安全感", "question": "为什么模仿别人有时让人安心？"}],
        }
        with patch("core.inspiration.chat_json", return_value=generated) as model:
            result = explore_inspiration(self.store, "为什么有人这么在乎虚名，这么喜欢对别人亦步亦趋")
        self.assertEqual(result["answer"], answer)
        self.assertNotIn("[推测]", result["answer"])
        self.assertIn("450～850", model.call_args.args[0])
        self.assertIn("不要因为几个相同日常词", model.call_args.args[0])

    def test_inspiration_normalizes_string_and_alternative_followup_fields(self):
        generated = {
            "answer": "可以从认同需求、群体压力与社会评价分别理解。",
            "primary_type": "open_exploration", "claims": [],
            "branches": [
                "为什么被群体认可会让人产生安全感？",
                {"direction": "声誉的社会作用", "prompt": "虚名与真实能力之间有什么区别？"},
                {"主题": "模仿与学习", "追问": "模仿他人什么时候有帮助，什么时候会变成盲从？"},
                {"heading": "没有观众的时候", "detail": "如果没有旁观者，这种行为是否仍会出现？"},
            ],
        }
        with patch("core.inspiration.chat_json", return_value=generated) as model:
            result = explore_inspiration(self.store, "为什么有人很在乎虚名和模仿别人")
        self.assertEqual(len(result["branches"]), 4)
        self.assertTrue(all(branch["title"] and branch["question"] for branch in result["branches"]))
        self.assertEqual(result["branches"][1], {
            "title": "声誉的社会作用", "question": "虚名与真实能力之间有什么区别？",
        })
        self.assertEqual(result["branches"][2]["title"], "模仿与学习")
        self.assertEqual(result["branches"][3]["title"], "没有观众的时候")
        self.assertIn("每个对象必须同时包含 title", model.call_args.args[0])

    def test_inspiration_deduplicates_and_skips_empty_followup_options(self):
        generated = {
            "answer": "我们可以进一步拆解这个观察。",
            "primary_type": "open_exploration", "claims": [],
            "branches": [
                {},
                None,
                {"title": "", "question": ""},
                "心理动机：人为什么需要得到别人的认可？",
                {"title": "重复方向", "question": "人为什么需要得到别人的认可？"},
                {"name": "历史视角", "questions": ["不同年代的人追求名声的方式一样吗？"]},
            ],
        }
        with patch("core.inspiration.chat_json", return_value=generated):
            result = explore_inspiration(self.store, "为什么有人在乎名声")
        self.assertEqual(result["branches"], [
            {"title": "心理动机", "question": "人为什么需要得到别人的认可？"},
            {"title": "历史视角", "question": "不同年代的人追求名声的方式一样吗？"},
        ])

    def test_inspiration_rejects_textbook_matching_only_conversational_filler(self):
        self.store.upsert_note({
            "path": "books/mountains/page-123", "title": "念念远山 · 第 123 页", "kind": "textbook",
            "content": "我一直好奇他为什么喜欢这么高的山峰，别人觉得登山太辛苦。",
            "source": "pdf", "content_hash": "mountains-filler",
        })
        generated = {
            "answer": "这种行为可能和认可、群体归属以及模仿带来的安全感有关。",
            "primary_type": "open_exploration", "secondary_types": [],
            "acknowledgement": "这是关于声誉和从众的问题。", "assumptions": [],
            "claims": [{"status": "fact", "text": "教材证明人们都追求虚名", "anchor_index": 1}],
            "counter_view": "", "branches": [], "used_anchor_indexes": [1],
        }
        with patch("core.inspiration.chat_json", return_value=generated) as model:
            result = explore_inspiration(
                self.store, "为什么有人这么在乎虚名，这么喜欢对别人亦步亦趋",
            )
        self.assertEqual(result["anchors"], [])
        self.assertEqual(result["claims"][0]["status"], "uncertain")
        self.assertNotIn("念念远山", model.call_args.args[1])

    def test_inspiration_shows_only_directly_relevant_and_used_textbook_anchor(self):
        self.store.upsert_note({
            "path": "books/biology/photosynthesis", "title": "植物生理学 · 光合作用", "kind": "textbook",
            "content": "光合作用利用叶绿素吸收光能，并把光能转化为化学能。",
            "source": "pdf", "content_hash": "photosynthesis-anchor",
        })
        generated = {
            "answer": "光合作用利用叶绿素吸收光能。[1]\n\n不过，人类是否能靠它满足能量需求仍是设想。",
            "primary_type": "counterfactual", "secondary_types": [],
            "acknowledgement": "这是一个关于光能与人体代谢的思想实验。",
            "assumptions": ["人体具有可运行的光合作用结构"],
            "claims": [{
                "status": "fact", "text": "光合作用利用叶绿素吸收光能", "anchor_index": 1,
            }],
            "counter_view": "还需要估计能量转化效率。", "branches": [],
            "used_anchor_indexes": [1],
        }
        with patch("core.inspiration.chat_json", return_value=generated):
            result = explore_inspiration(self.store, "如果人类也能光合作用，会发生什么？")
        self.assertEqual(len(result["anchors"]), 1)
        self.assertEqual(result["anchors"][0]["title"], "植物生理学 · 光合作用")
        self.assertEqual(result["claims"][0]["status"], "fact")
        self.assertIn("[1]", result["answer"])

    def test_inspiration_does_not_display_unused_relevant_reference(self):
        self.store.upsert_note({
            "path": "books/biology/photosynthesis-unused", "title": "植物生理学 · 光合作用",
            "kind": "textbook", "content": "光合作用利用叶绿素吸收光能，并转化为化学能。",
            "source": "pdf", "content_hash": "photosynthesis-unused",
        })
        generated = {
            "answer": "可以先把这个问题当成一个思想实验，从能量与身体结构分别讨论。",
            "primary_type": "counterfactual", "secondary_types": [],
            "acknowledgement": "我们可以先推演。", "assumptions": [],
            "claims": [{"status": "imagination", "text": "想象人体改变能量来源", "anchor_index": None}],
            "counter_view": "", "branches": [], "used_anchor_indexes": [],
        }
        with patch("core.inspiration.chat_json", return_value=generated):
            result = explore_inspiration(self.store, "如果人类也能光合作用，会发生什么？")
        self.assertEqual(result["anchors"], [])

    def test_inspiration_prioritizes_current_topic_over_unrelated_old_history(self):
        self.store.upsert_note({
            "path": "books/math/eigenvectors", "title": "线性代数 · 特征向量", "kind": "textbook",
            "content": "矩阵的特征向量在线性变换后保持原来的方向。",
            "source": "pdf", "content_hash": "eigenvectors-old-topic",
        })
        with patch("core.inspiration.chat_json", return_value={
            "answer": "可以从社会期待与个体选择之间的张力来理解。",
            "primary_type": "open_exploration", "claims": [], "branches": [],
        }) as model:
            result = explore_inspiration(
                self.store,
                "为什么有人这么在乎虚名，这么喜欢对别人亦步亦趋",
                [{"role": "user", "content": "特征向量和矩阵之间的关系是什么？"}],
            )
        self.assertEqual(result["anchors"], [])
        self.assertNotIn("线性代数 · 特征向量", model.call_args.args[1])

    def test_inspiration_offline_fallback_is_a_substantial_natural_answer(self):
        with patch("core.inspiration.chat_json", return_value=None):
            result = explore_inspiration(self.store, "为什么有人这么在乎虚名")
        self.assertGreater(len(result["answer"]), 180)
        self.assertGreaterEqual(result["answer"].count("\n\n"), 2)
        self.assertNotIn("[推测]", result["answer"])

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
