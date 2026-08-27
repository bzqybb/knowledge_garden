from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class AnswerFormattingTests(unittest.TestCase):
    def test_answer_paragraphs_use_two_em_first_line_indent(self) -> None:
        css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        self.assertIn(".assistant-turn>p", css)
        self.assertIn("text-indent:2em", css)

    def test_content_specific_bold_headings_render_as_mini_headings(self) -> None:
        javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('class="md-mini-heading"', javascript)
        self.assertIn("const strongHeading", javascript)
        self.assertIn("const strongLead", javascript)

    def test_frontend_no_longer_rewrites_fixed_answer_sections(self) -> None:
        javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        self.assertNotIn('"结论":"先说结论"', javascript)
        self.assertNotIn('"机制":"为什么"', javascript)

    def test_frontier_form_clears_only_after_successful_analysis(self) -> None:
        javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        handler = javascript.split('$("#analyze-form").onsubmit=', 1)[1].split(
            "function normalizeInspirationBranches", 1
        )[0]

        self.assertIn("const form=e.currentTarget", handler)
        self.assertIn("form.reset()", handler)
        self.assertLess(handler.index("form.reset()"), handler.index("}catch(err)"))
        self.assertNotIn("form.reset()", handler.split("}catch(err)", 1)[1])


if __name__ == "__main__":
    unittest.main()
