"""Serper search with trusted-source filtering.

All search results are post-filtered against the trusted-source whitelist.
The agent NEVER sees raw, unfiltered search results — the whitelist is
enforced at the tool boundary, not as a prompt instruction.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .trusted_sources import credibility_tier, extract_domain, filter_trusted, is_evidence_source

logger = logging.getLogger(__name__)

SERPER_URL = os.environ.get("SERPER_API_URL", "https://google.serper.dev/search")
SERPER_KEY = os.environ.get("SERPER_API_KEY", "")


def _search(query: str, num: int = 10, page: int = 1) -> dict[str, Any]:
    """Raw Serper search. Returns the parsed JSON or an error dict."""
    if not SERPER_KEY:
        return {"error": "SERPER_API_KEY not configured", "results": []}

    try:
        resp = httpx.post(
            SERPER_URL,
            json={"q": query, "num": num, "page": page},
            headers={"X-API-KEY": SERPER_KEY},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Serper HTTP %d: %s", exc.response.status_code, exc.response.text[:200])
        return {"error": f"Serper HTTP {exc.response.status_code}", "results": []}
    except Exception as exc:
        logger.error("Serper error: %s", exc)
        return {"error": str(exc), "results": []}


def search_trusted(
    query: str,
    claim_id: str = "",
    num_results: int = 8,
    evidence_only: bool = True,
) -> dict[str, Any]:
    """Search trusted sources for evidence about a claim.

    Args:
        query: Search query string.
        claim_id: Optional claim identifier for budget tracking.
        num_results: Max raw results to request from Serper.
        evidence_only: If True, filter to evidence-tier sources only (excludes Wikipedia).

    Returns:
        {
            "query": str,
            "organic": [{title, url, snippet, domain, tier, trusted, is_evidence}],
            "knowledge_graph": {...} or null,
            "answer_box": {...} or null,
            "total_trusted": int,
            "total_raw": int,
        }
    """
    t0 = time.monotonic()
    raw = _search(query, num=num_results)

    if "error" in raw:
        return {"query": query, "error": raw["error"], "organic": [], "total_trusted": 0, "total_raw": 0}

    organic = raw.get("organic", [])
    total_raw = len(organic)

    enriched: list[dict[str, Any]] = []
    for r in organic:
        url = r.get("link", "")
        domain = extract_domain(url)
        tier = credibility_tier(url)
        is_ev = is_evidence_source(url)
        if evidence_only and not is_ev:
            continue
        if tier == 0:
            continue
        enriched.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("snippet", ""),
            "domain": domain,
            "tier": tier,
            "tier_label": {1: "Wire Service", 2: "Fact-Checker", 3: "Institutional", 4: "Reference"}.get(tier, "Unknown"),
            "is_evidence_source": is_ev,
        })

    result = {
        "query": query,
        "organic": enriched,
        "knowledge_graph": raw.get("knowledgeGraph"),
        "answer_box": raw.get("answerBox"),
        "total_trusted": len(enriched),
        "total_raw": total_raw,
        "elapsed_ms": round((time.monotonic() - t0) * 1000),
    }
    logger.info("search_trusted(%s) → %d/%d trusted in %dms", query[:60], len(enriched), total_raw, result["elapsed_ms"])
    return result


def fetch_article_text(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch and extract readable text from a trusted article URL.

    Args:
        url: Article URL (must be in trusted whitelist).
        max_chars: Maximum characters to return (truncated with ellipsis).

    Returns:
        {"url": str, "domain": str, "title": str or None, "text": str, "truncated": bool}
    """
    from .trusted_sources import is_trusted

    domain = extract_domain(url)
    if not is_trusted(url):
        return {"url": url, "domain": domain, "error": f"Domain '{domain}' is not in the trusted whitelist.", "text": ""}

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "VeriLens-FactCheck-MCP/1.0 (trusted-source-verification)"},
            timeout=20.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return {"url": url, "domain": domain, "error": str(exc), "text": ""}

    # Simple text extraction: strip HTML tags
    import re
    html = resp.text
    # Remove scripts, styles
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else None

    truncated = len(text) > max_chars
    return {
        "url": url,
        "domain": domain,
        "title": title,
        "text": text[:max_chars] + ("…" if truncated else ""),
        "truncated": truncated,
    }
