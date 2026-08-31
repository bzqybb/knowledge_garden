from __future__ import annotations

import re
from typing import Iterable


TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\[(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<text>.*)$"
)


def split_timestamped_text(text: str, max_chars: int = 9000) -> list[str]:
    """Split a transcript without dropping content or breaking timestamp lines."""
    source = str(text or "").strip()
    if not source:
        return []
    limit = max(1000, int(max_chars))
    lines = source.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in lines:
        addition = len(line) + (1 if current else 0)
        if current and current_size + addition > limit:
            chunks.append("\n".join(current).strip())
            current = []
            current_size = 0
        # An unusually long non-timestamp line still has to be covered in full.
        if not current and len(line) > limit and not TIMESTAMP_LINE_RE.match(line):
            for start in range(0, len(line), limit):
                part = line[start:start + limit].strip()
                if part:
                    chunks.append(part)
            continue
        current.append(line)
        current_size += addition
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def timestamp_evidence(text: str, concept: str = "", limit: int = 3) -> list[str]:
    """Return verbatim timestamped lines, preferring lines that name the concept."""
    lines = [line.strip() for line in str(text or "").splitlines() if TIMESTAMP_LINE_RE.match(line)]
    if not lines:
        return []
    needle = str(concept or "").strip().lower()
    preferred = [line for line in lines if needle and needle in line.lower()]
    ordered = preferred + [line for line in lines if line not in preferred]
    return ordered[:max(1, int(limit))]


def select_relevant_chunks(chunks: Iterable[str], concept: str, limit: int = 3) -> list[str]:
    """Select complete chunks relevant to one concept, wherever they occur."""
    items = [str(chunk or "").strip() for chunk in chunks if str(chunk or "").strip()]
    if not items:
        return []
    needle = str(concept or "").strip().lower()
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}|[\u4e00-\u9fff]{2,8}", concept)]
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(items):
        lower = chunk.lower()
        exact = lower.count(needle) if needle else 0
        overlap = sum(lower.count(term) for term in terms)
        # Exact concept matches dominate, while source order only breaks ties.
        scored.append((exact * 100 + overlap, -index, chunk))
    matched = [row for row in sorted(scored, reverse=True) if row[0] > 0]
    if not matched:
        matched = sorted(scored, reverse=True)[:1]
    selected = [row[2] for row in matched[:max(1, int(limit))]]
    # Restore chronological order for a readable evidence excerpt.
    positions = {chunk: index for index, chunk in enumerate(items)}
    return sorted(selected, key=lambda chunk: positions.get(chunk, 0))
