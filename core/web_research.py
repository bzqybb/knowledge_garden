from __future__ import annotations

import json
import ipaddress
import re
import socket
import time
from xml.etree import ElementTree
from datetime import date
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
BING_RSS_SEARCH = "https://www.bing.com/search"


def search_public_web(query: str, limit: int = 5, timeout: int = 8) -> list[dict[str, Any]]:
    """Find public pages without an API key, preserving attributable snippets.

    Search results are evidence only for what their title and snippet actually
    say. Institutional domains are ranked first, but no arbitrary result URL is
    fetched, so this function cannot become an SSRF proxy.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []
    params = urlencode({
        "q": cleaned_query,
        "format": "rss",
        "count": max(1, min(limit, 8)),
    })
    request = Request(
        f"{BING_RSS_SEARCH}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0 KnowledgeGarden/1.0 (public-source lookup)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("公开网页检索结果过大")
    root = ElementTree.fromstring(payload)
    results: list[dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        title = unescape(str(item.findtext("title") or "")).strip()
        url = str(item.findtext("link") or "").strip()
        parsed = urlparse(url)
        if not title or parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        description = re.sub(r"<[^>]+>", " ", str(item.findtext("description") or ""))
        description = re.sub(r"\s+", " ", unescape(description)).strip()
        if not description:
            continue
        hostname = parsed.hostname.casefold()
        official = hostname.endswith((".edu.cn", ".ac.cn", ".gov.cn", ".edu", ".gov"))
        results.append({
            "title": title,
            "url": url,
            "abstract": description[:1800],
            "year": None,
            "authors": [],
            "venue": hostname,
            "source": "机构官网" if official else "公开网页",
            "source_type": "official_docs" if official else "public_web",
            "official": official,
        })
    results.sort(key=lambda result: result["official"], reverse=True)
    return results[:max(1, min(limit, 8))]


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


def _safe_public_https(url: str, *, attempts: int = 3) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    for attempt in range(max(1, attempts)):
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
            }
            return bool(addresses) and all(
                ipaddress.ip_address(value).is_global for value in addresses
            )
        except (OSError, ValueError):
            if attempt + 1 < max(1, attempts):
                time.sleep(0.25 * (attempt + 1))
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


def _request_json_with_retry(request: Request, timeout: int, attempts: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                raise
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
        time.sleep(0.35 * (attempt + 1))
    raise RuntimeError("在线学术检索失败") from last_error


def _crossref_date(item: dict[str, Any]) -> tuple[str, int | None]:
    parts = ((item.get("published-online") or item.get("published-print") or item.get("issued") or {}).get("date-parts") or [])
    values = parts[0] if parts else []
    if not values:
        return "", None
    padded = [*values[:3], 1, 1][:3]
    return "-".join(f"{int(value):02d}" if index else f"{int(value):04d}" for index, value in enumerate(padded)), int(values[0])


def _article_from_crossref(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(next(iter(item.get("title") or []), "")).strip()
    doi = str(item.get("DOI") or "").strip()
    if not title or not doi:
        return None
    authors = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [str(author.get("given") or "").strip(), str(author.get("family") or "").strip()] if part)
        if name:
            authors.append(name)
        if len(authors) == 4:
            break
    publication_date, year = _crossref_date(item)
    abstract = re.sub(r"<[^>]+>", " ", str(item.get("abstract") or ""))
    abstract = re.sub(r"\s+", " ", unescape(abstract)).strip()[:1400]
    return {
        "title": title,
        "url": f"https://doi.org/{doi}",
        "year": year,
        "authors": authors,
        "venue": str(next(iter(item.get("container-title") or []), "")).strip(),
        "abstract": abstract,
        "cited_by_count": int(item.get("is-referenced-by-count") or 0),
        "source": "Crossref / DOI",
        "publication_date": publication_date,
        "open_access": False,
        "pdf_url": "",
    }


def _query_term_coverage(query: str, article: dict[str, Any]) -> float:
    stop = {"recent", "research", "review", "learning", "applications", "application", "emerging", "technology"}
    query_terms = {
        token for token in re.findall(r"[a-z][a-z-]{3,}", query.lower()) if token not in stop
    }
    if not query_terms:
        return 1.0
    corpus = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    matched = {term for term in query_terms if term in corpus}
    return len(matched) / len(query_terms)


def search_academic_articles(
    query: str, limit: int = 4, timeout: int = 20, *, from_publication_date: str = "",
    diagnostics: dict[str, Any] | None = None, attempts_per_provider: int = 2,
) -> list[dict[str, Any]]:
    """Search OpenAlex with retry, then degrade to Crossref without another API key."""
    report = diagnostics if diagnostics is not None else {}
    report.update({"attempted": ["OpenAlex"], "provider": "", "errors": [], "degraded": False})
    # Search relevance must lead. Freshness is already scored later by the
    # garden ranker; sorting the provider by date first admits topical noise.
    options = {"search": query.strip(), "per-page": max(1, min(limit + 4, 20)), "sort": "relevance_score:desc"}
    if from_publication_date:
        options["filter"] = f"from_publication_date:{from_publication_date}"
    params = urlencode(options)
    request = Request(
        f"{OPENALEX_API}?{params}",
        headers={"User-Agent": "KnowledgeGarden/1.0 (local learning assistant)"},
    )
    articles = []
    try:
        payload = _request_json_with_retry(request, timeout, attempts=attempts_per_provider)
        for work in payload.get("results") or []:
            article = _article_from_work(work)
            if article and article["url"] and _query_term_coverage(query, article) >= 0.67:
                articles.append(article)
            if len(articles) == limit:
                break
        if articles:
            report["provider"] = "OpenAlex"
            return articles
        report["errors"].append("OpenAlex 没有返回符合条件的记录")
    except Exception as exc:
        code = f" HTTP {exc.code}" if isinstance(exc, HTTPError) else ""
        report["errors"].append(f"OpenAlex{code}：{exc.__class__.__name__}")

    report["attempted"].append("Crossref")
    report["degraded"] = True
    crossref_options: dict[str, Any] = {
        "query.bibliographic": query.strip(), "rows": max(1, min(limit + 4, 20)),
        "sort": "relevance", "order": "desc",
    }
    if from_publication_date:
        crossref_options["filter"] = f"from-pub-date:{from_publication_date},until-pub-date:{date.today().isoformat()}"
    crossref_request = Request(
        f"{CROSSREF_API}?{urlencode(crossref_options)}",
        headers={"User-Agent": "KnowledgeGarden/1.0 (mailto:knowledge-garden@localhost)"},
    )
    try:
        payload = _request_json_with_retry(crossref_request, timeout, attempts=attempts_per_provider)
        for item in (payload.get("message") or {}).get("items") or []:
            article = _article_from_crossref(item)
            if article and _query_term_coverage(query, article) >= 0.67:
                articles.append(article)
            if len(articles) == limit:
                break
        if articles:
            report["provider"] = "Crossref"
            return articles
        report["errors"].append("Crossref 没有返回符合条件的记录")
    except Exception as exc:
        code = f" HTTP {exc.code}" if isinstance(exc, HTTPError) else ""
        report["errors"].append(f"Crossref{code}：{exc.__class__.__name__}")
    raise RuntimeError("；".join(report["errors"]) or "在线学术检索没有返回结果")
