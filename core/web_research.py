from __future__ import annotations

import json
import ipaddress
import re
import socket
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


OPENALEX_API = "https://api.openalex.org/works"


class _WeChatArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.content_depth = 0
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self.content_depth == 0 and attributes.get("id") == "js_content":
            self.content_depth = 1
            return
        if self.content_depth:
            self.content_depth += 1
            if tag in {"script", "style", "noscript"}:
                self.skip_depth += 1
            if tag in {"p", "div", "section", "br", "li", "h1", "h2", "h3"}:
                self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self.content_depth:
            return
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        self.content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.content_depth and not self.skip_depth:
            text = unescape(data).strip()
            if text:
                self.parts.append(text)


def _abstract_from_index(index: dict[str, list[int]] | None, limit: int = 1400) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, offsets in index.items() for position in offsets]
    text = " ".join(word for _, word in sorted(positions))
    return text[:limit] + ("…" if len(text) > limit else "")


def _article_from_work(work: dict[str, Any]) -> dict[str, Any] | None:
    title = str(work.get("display_name") or work.get("title") or "").strip()
    if not title:
        return None
    authors = []
    for authorship in work.get("authorships") or []:
        name = str((authorship.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
        if len(authors) == 4:
            break
    location = work.get("primary_location") or work.get("best_oa_location") or {}
    oa_location = work.get("best_oa_location") or {}
    source = location.get("source") or {}
    doi = str(work.get("doi") or "").strip()
    url = doi or str(location.get("landing_page_url") or work.get("id") or "").strip()
    return {
        "title": title,
        "url": url,
        "year": work.get("publication_year"),
        "authors": authors,
        "venue": str(source.get("display_name") or "").strip(),
        "abstract": _abstract_from_index(work.get("abstract_inverted_index")),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "source": "OpenAlex / DOI",
        "publication_date": work.get("publication_date"),
        "open_access": bool((work.get("open_access") or {}).get("is_oa")),
        "pdf_url": str(oa_location.get("pdf_url") or "").strip(),
    }


def _safe_public_https(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
        return bool(addresses) and all(ipaddress.ip_address(value).is_global for value in addresses)
    except (OSError, ValueError):
        return False


def fetch_open_access_pdf_text(
    pdf_url: str, *, timeout: int = 25, max_bytes: int = 12_000_000, max_pages: int = 40,
) -> str:
    """Read text only from a public HTTPS PDF URL supplied by OpenAlex."""
    if not _safe_public_https(pdf_url):
        raise ValueError("开放全文链接不是可安全访问的公网 HTTPS 地址")
    request = Request(pdf_url, headers={"User-Agent": "KnowledgeGarden/1.0 (research reader)"})
    with urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("Content-Type") or "").lower()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("开放全文文件过大")
    if "pdf" not in content_type and not data.startswith(b"%PDF"):
        raise ValueError("开放全文链接没有返回 PDF")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取开放论文需要 pypdf") from exc
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages[:max_pages]]
    return "\n\n".join(page for page in pages if page)[:60_000]


def fetch_wechat_article_text(url: str, *, timeout: int = 20, max_bytes: int = 10_000_000) -> str:
    """Fetch the readable body of an explicitly selected WeChat article.

    The allow-list is intentionally narrow because article URLs originate in
    personal message data and must not become a general-purpose SSRF proxy.
    """
    parsed = urlparse(url.strip())
    if parsed.hostname != "mp.weixin.qq.com" or parsed.username or parsed.password:
        raise ValueError("只允许读取微信公众平台 mp.weixin.qq.com 的文章链接")
    safe_url = parsed._replace(scheme="https").geturl()
    if not _safe_public_https(safe_url):
        raise ValueError("公众号文章链接不是可安全访问的公网 HTTPS 地址")
    request = Request(safe_url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KnowledgeGarden/1.0",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    with urlopen(request, timeout=timeout) as response:
        final = urlparse(response.geturl())
        if final.hostname != "mp.weixin.qq.com":
            raise ValueError("公众号文章发生了不允许的跨站重定向")
        content_type = str(response.headers.get("Content-Type") or "").lower()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("公众号文章页面过大")
    if "html" not in content_type:
        raise ValueError("公众号链接没有返回 HTML 正文")
    html_text = data.decode("utf-8", errors="replace")
    parser = _WeChatArticleParser()
    parser.feed(html_text)
    text = "\n".join(
        part.strip() for part in re.split(r"\n+", "".join(parser.parts)) if part.strip()
    )
    if len(text) < 80:
        raise ValueError("微信没有返回可读正文，可能需要在浏览器中完成验证后再试")
    return text[:60_000]


def search_academic_articles(
    query: str, limit: int = 4, timeout: int = 20, *, from_publication_date: str = "",
) -> list[dict[str, Any]]:
    """Search scholarly metadata and abstracts without requiring another API key."""
    options = {"search": query.strip(), "per-page": max(1, min(limit + 4, 20)), "sort": "publication_date:desc"}
    if from_publication_date:
        options["filter"] = f"from_publication_date:{from_publication_date}"
    params = urlencode(options)
    request = Request(
        f"{OPENALEX_API}?{params}",
        headers={"User-Agent": "KnowledgeGarden/1.0 (local learning assistant)"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    articles = []
    for work in payload.get("results") or []:
        article = _article_from_work(work)
        if article and article["url"]:
            articles.append(article)
        if len(articles) == limit:
            break
    return articles
