from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from core.pdf_ocr import (
    OCR_STATE_KEY, clean_pdf_text, discover_scanned_textbooks,
    ocr_textbook_into_store,
)
from core.storage import GardenStore


class PdfOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = GardenStore(self.root / "garden.db")
        self.pdf = self.root / "普通化学扫描版.pdf"
        writer = PdfWriter()
        for _ in range(5):
            writer.add_blank_page(width=180, height=260)
        with self.pdf.open("wb") as handle:
            writer.write(handle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clean_pdf_text_removes_invalid_unicode_and_joins_chinese_ocr(self) -> None:
        self.assertEqual(clean_pdf_text("化 学 反 应\ud800", from_ocr=True), "化学反应")
        self.assertEqual(clean_pdf_text("normal\udcff text"), "normal text")

    def test_discovers_scanned_books_without_indexed_text(self) -> None:
        result = discover_scanned_textbooks(self.store, self.root)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pages"], 5)
        self.assertEqual(result[0]["indexed_pages"], 0)

    def test_ocr_pages_are_indexed_and_resume_without_restarting(self) -> None:
        text = "化 学 反 应 与 平 衡 常 数 " * 8
        first_results = [
            {"page": 1, "page_count": 5, "language": "zh-Hans-CN", "text": text},
            {"page": 2, "page_count": 5, "language": "zh-Hans-CN", "text": text},
        ]
        with patch("core.pdf_ocr.iter_windows_pdf_ocr", return_value=iter(first_results)):
            first = ocr_textbook_into_store(self.store, self.pdf, max_pages=2)
        self.assertEqual(first["indexed"], 2)
        self.assertFalse(first["completed"])
        self.assertIn("化学反应", self.store.list_notes()[0]["content"])

        remaining = [
            {"page": page, "page_count": 5, "language": "zh-Hans-CN", "text": text}
            for page in (3, 4, 5)
        ]
        with patch("core.pdf_ocr.iter_windows_pdf_ocr", return_value=iter(remaining)) as run:
            second = ocr_textbook_into_store(self.store, self.pdf)
        self.assertEqual(second["start_page"], 3)
        self.assertTrue(second["completed"])
        self.assertEqual(len(self.store.list_notes()), 5)
        self.assertEqual(run.call_args.kwargs["start_page"], 3)
        self.assertTrue(self.store.setting(OCR_STATE_KEY)[str(self.pdf.resolve())]["completed"])

    def test_completed_scanned_book_is_not_queued_again(self) -> None:
        results = [{
            "page": 5, "page_count": 5, "language": "zh-Hans-CN",
            "text": "可识别教材文本" * 12,
        }]
        with patch("core.pdf_ocr.iter_windows_pdf_ocr", return_value=iter(results)):
            ocr_textbook_into_store(self.store, self.pdf)
        self.assertEqual(discover_scanned_textbooks(self.store, self.root), [])


if __name__ == "__main__":
    unittest.main()
