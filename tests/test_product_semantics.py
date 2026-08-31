import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ProductSemanticsTests(unittest.TestCase):
    def test_continue_discussion_keeps_assistant_prompt_out_of_user_value(self):
        source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        match = re.search(
            r"function continueAgentDiscussion\(prompt\)\{(?P<body>.*?)\n\}",
            source,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn('input.value=""', body)
        self.assertNotRegex(body, r"input\.value\s*=\s*(?:prompt|assistantPrompt)")
        self.assertIn("input.placeholder", body)

    def test_reading_prompt_is_not_inserted_as_user_text(self):
        source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\.value\s*=\s*button\.dataset\.prompt")
        self.assertNotRegex(source, r"input\.value\s*=\s*prompt")

    def test_new_agent_conversation_clears_old_followup_prompt(self):
        source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        match = re.search(
            r"function startNewAgentConversation\(\)\{(?P<body>.*?)\}\n",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn('input.value=""', body)
        self.assertIn('input.placeholder="写下你想真正弄清的问题……"', body)
        self.assertNotRegex(source, r"input\.value\s*=\s*button\.dataset\.question")

    def test_legacy_meta_refusals_are_hidden_from_persisted_chat(self):
        source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("isLegacyMetaRefusal", source)
        self.assertGreaterEqual(len(re.findall(r"filter\([^\n]+isLegacyMetaRefusal", source)), 2)

    def test_reanswer_uses_revised_question_as_canonical_question(self):
        frontend = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        backend = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('canonicalQuestion=r.reanswer?.revised_question', frontend)
        self.assertIn('lastAgentExchange={question:canonicalQuestion', frontend)
        self.assertIn('question:canonicalQuestion', frontend)
        self.assertIn('.get("revised_question")', backend)
        reanswer_capture = backend[backend.index('surface="gardener_reanswer"'):]
        self.assertLess(
            reanswer_capture.index('.get("revised_question")'),
            reanswer_capture.index('self._json({"ok": True, "result": result})'),
        )

    def test_bilibili_guide_renders_structure_and_keeps_full_transcript(self):
        source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        start = source.index("function formatVideoAnalysis")
        end = source.index("window.readBilibili", start)
        formatter = source[start:end]
        for field in ["key_points", "chapter_outline", "concepts", "caveats", "questions", "timestamp", "evidence"]:
            self.assertIn(field, formatter)
        self.assertIn("r.transcript", formatter)
        self.assertNotRegex(formatter, r"r\.transcript\.(?:slice|substring)\(")
        self.assertIn("n.content=formatVideoAnalysis(a,r)", source)


if __name__ == "__main__":
    unittest.main()
