from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.feeds import (
    _extract_bilibili_videos,
    describe_feed,
    fetch_source,
    list_followed_sources,
    list_frontier_material,
    refresh_feeds,
)
from core.storage import GardenStore


BILIBILI_PAGE = """<html><script>window.__pinia=(function(a,b,c,d,e,f,g,h,i){
return{items:[
{type:a,author:b,mid:c,bvid:d,title:e,description:f,pubdate:g},
{type:a,author:h,mid:i,bvid:"BV1other0000",title:"其他博主",description:"不能混入",pubdate:g}
]}}("video","测试博主",42,"BV1owner0000","<em>线性代数</em>入门","矩阵与向量",1700000000,"其他博主",77));</script></html>"""

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><item>
<title>First article</title><link>https://example.com/one</link>
<description><![CDATA[<p>Useful &amp; grounded</p>]]></description>
<pubDate>2026-08-25</pubDate></item></channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>First video</title><link href="https://example.com/video"/>
<summary>Summary</summary><updated>2026-08-25</updated></entry></feed>"""


class FeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = GardenStore(Path(self.temp.name) / "garden.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_bilibili_profile_is_normalized_and_preserves_uid(self) -> None:
        source = describe_feed("https://space.bilibili.com/385670211/video")
        self.assertEqual(source["platform"], "bilibili")
        self.assertEqual(source["uid"], "385670211")
        self.assertEqual(source["url"], "https://space.bilibili.com/385670211")

    def test_bilibili_mobile_profile_is_supported(self) -> None:
        self.assertEqual(
            describe_feed("https://m.bilibili.com/space/42")["url"],
            "https://space.bilibili.com/42",
        )

    def test_youtube_channel_uses_its_public_atom_feed(self) -> None:
        source = describe_feed("https://www.youtube.com/channel/UC123456")
        self.assertEqual(source["platform"], "youtube")
        self.assertIn("channel_id=UC123456", source["feed_url"])

    def test_github_creator_and_repository_have_different_atom_feeds(self) -> None:
        creator = describe_feed("https://github.com/octocat")
        repository = describe_feed("https://github.com/octocat/Hello-World")
        self.assertEqual(creator["feed_url"], "https://github.com/octocat.atom")
        self.assertTrue(repository["feed_url"].endswith("/commits/HEAD.atom"))

    def test_substack_medium_and_standard_rss_are_supported(self) -> None:
        self.assertEqual(describe_feed("https://writer.substack.com")["feed_url"],
                         "https://writer.substack.com/feed")
        self.assertEqual(describe_feed("https://medium.com/@writer")["feed_url"],
                         "https://medium.com/feed/@writer")
        self.assertEqual(describe_feed("https://example.org/feed.xml")["platform"], "rss")

    def test_credentials_and_non_http_urls_are_rejected(self) -> None:
        for value in ("file:///tmp/private", "https://user:password@example.com/feed", "not-a-url"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    describe_feed(value)

    def test_bilibili_state_filters_videos_by_creator_uid_without_executing_javascript(self) -> None:
        videos = _extract_bilibili_videos(BILIBILI_PAGE, "42")
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["title"], "线性代数 入门")
        self.assertEqual(videos[0]["url"], "https://www.bilibili.com/video/BV1owner0000")
        self.assertEqual(videos[0]["summary"], "矩阵与向量")

    def test_bilibili_state_rejects_missing_creator(self) -> None:
        with self.assertRaisesRegex(ValueError, "UID 999"):
            _extract_bilibili_videos(BILIBILI_PAGE, "999")

    def test_bilibili_source_fetches_only_its_own_public_search_page(self) -> None:
        with patch("core.feeds._fetch_bytes", return_value=BILIBILI_PAGE.encode("utf-8")) as fetch:
            videos = fetch_source("https://space.bilibili.com/42", name="测试博主")
        self.assertEqual(len(videos), 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn("search.bilibili.com", fetch.call_args.args[0])

    def test_rss_and_atom_entries_are_still_supported(self) -> None:
        with patch("core.feeds._fetch_bytes", return_value=RSS):
            rss = fetch_source("https://example.com/feed.xml")
        with patch("core.feeds._fetch_bytes", return_value=ATOM):
            atom = fetch_source("https://example.com/feed.atom")
        self.assertEqual(rss[0]["title"], "First article")
        self.assertEqual(rss[0]["summary"], "Useful & grounded")
        self.assertEqual(atom[0]["url"], "https://example.com/video")

    def test_generic_blog_homepage_can_discover_an_advertised_feed(self) -> None:
        homepage = b'<html><head><link rel="alternate" type="application/rss+xml" href="/feed.xml"></head></html>'
        with patch("core.feeds._fetch_bytes", side_effect=[homepage, RSS]) as fetch:
            rows = fetch_source("https://example.com/blog")
        self.assertEqual(rows[0]["title"], "First article")
        self.assertEqual(fetch.call_args.args[0], "https://example.com/feed.xml")

    def test_refresh_imports_bilibili_seed_once_and_reports_platform(self) -> None:
        self.store.add_feed("测试博主", "https://space.bilibili.com/42")
        with (
            patch("core.feeds._fetch_bytes", return_value=BILIBILI_PAGE.encode("utf-8")),
            patch("core.bilibili_mcp.inspect_public_video", return_value={
                "status": "no_subtitle", "message": "没有公开字幕",
            }),
        ):
            first = refresh_feeds(self.store)
            second = refresh_feeds(self.store)
        self.assertEqual(first["fetched"], 1)
        self.assertEqual(first["added"], 1)
        self.assertEqual(first["sources"][0]["platform"], "bilibili")
        self.assertEqual(second["added"], 0)
        note = self.store.list_notes(kind="frontier")[0]
        self.assertEqual(note["source_url"], "https://www.bilibili.com/video/BV1owner0000")
        self.assertEqual(len(self.store.list_tasks()), 1)
        self.assertIsNotNone(list_followed_sources(self.store)[0]["last_checked_at"])

    def test_vault_raw_frontier_entries_are_visible_without_breaking_directory_taxonomy(self) -> None:
        vault = Path(self.temp.name) / "vault"
        vault.mkdir()
        self.store.set_setting("vault_path", str(vault))
        self.store.add_feed("测试博主", "https://space.bilibili.com/42")
        with (
            patch("core.feeds._fetch_bytes", return_value=BILIBILI_PAGE.encode("utf-8")),
            patch("core.bilibili_mcp.inspect_public_video", return_value={
                "status": "no_subtitle", "message": "没有公开字幕",
            }),
        ):
            first = refresh_feeds(self.store)
            second = refresh_feeds(self.store)

        self.assertEqual(first["added"], 1)
        self.assertEqual(second["added"], 0)
        self.assertEqual(len(self.store.list_notes(kind="raw")), 1)
        self.assertEqual(self.store.list_notes(kind="frontier"), [])
        visible = list_frontier_material(self.store)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["title"], "线性代数 入门")


    def test_refresh_returns_specific_error_instead_of_silent_success(self) -> None:
        self.store.add_feed("测试博主", "https://space.bilibili.com/42")
        with patch("core.feeds._fetch_bytes", side_effect=ValueError("B站暂时触发游客访问限制")):
            result = refresh_feeds(self.store)
        self.assertEqual(result["added"], 0)
        self.assertIn("游客访问限制", result["errors"][0])
        self.assertEqual(result["sources"][0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
