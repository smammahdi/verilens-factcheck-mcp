#!/usr/bin/env python3
"""VeriLens FactCheck MCP Server.

Exposes a sandboxed, whitelist-restricted fact-checking toolset to AI agents.
The agent uses it to verify factual claims against trusted news outlets
(Reuters, AP, AFP, BBC, regional verified sources).

Why MCP instead of a generic web-search tool:
  If the agent can "verify" a claim against an arbitrary blog or AI-written
  aggregator, the entire product premise — flagging misinformation — collapses.
  Wrapping search, fetch, and verification inside MCP enforces the trusted-source
  whitelist, per-tool rate limits, and per-claim tool-call budgets at the
  protocol boundary — not as soft instructions inside a prompt.

Run:
  stdio (local dev):     python server.py
  Streamable HTTP:       python server.py --transport streamable-http --port 8080

Requirements:
  pip install fastmcp httpx
  + factcheck library (vendored in ../fact-verifier/factcheck/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

# Load .env if present (local dev)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from mcp.server.fastmcp import FastMCP

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.search import search_trusted as _do_search, fetch_article_text
from src.rate_limiter import get_limiter
from src.verifier import get_engine
from src.trusted_sources import (
    ALL_TRUSTED,
    EVIDENCE_SOURCES,
    credibility_tier,
    extract_domain,
    filter_evidence,
    filter_trusted,
    is_trusted,
    tier_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("factcheck-mcp")

mcp = FastMCP(
    "VeriLens FactCheck",
    description="VeriLens Fact-Check MCP — verify claims against trusted news sources with rate-limited, whitelist-enforced search and fetch tools.",
    dependencies=["fastmcp>=2.0", "httpx>=0.27", "python-dotenv>=1.0"],
)


# ── Tools ─────────────────────────────────────────────────────────────


@mcp.tool()
def search_trusted_sources(
    query: str,
    claim_id: str = "",
    num_results: int = 8,
    evidence_only: bool = True,
) -> str:
    """Search trusted news sources for evidence about a claim.

    ONLY searches domains on the trusted-source whitelist (Reuters, AP,
    AFP, BBC, NPR, PBS, fact-checking orgs, government/institutional sites).
    Results from blogs, social media, and AI aggregators are FILTERED OUT.

    Args:
        query: Search query (e.g. "Eiffel Tower construction date 1889").
        claim_id: Optional claim identifier for budget tracking.
        num_results: Number of raw results to request (1-10, default 8).
        evidence_only: If True, filter to evidence-tier sources (excludes Wikipedia).

    Returns:
        JSON with trusted search results including title, URL, domain, credibility tier.
    """
    limiter = get_limiter()
    if not limiter.allow("search_trusted_sources"):
        return json.dumps({"error": "Rate limit exceeded. Wait before retrying.", "results": []})

    if claim_id:
        if not limiter.claim_can_search(claim_id):
            return json.dumps({"error": f"Search budget exhausted for claim '{claim_id}' ({limiter.claim_status(claim_id)['searches']}).", "results": []})
        limiter.claim_record_search(claim_id)

    result = _do_search(query=query, claim_id=claim_id, num_results=num_results, evidence_only=evidence_only)

    # Add budget context
    if claim_id:
        result["claim_budget"] = limiter.claim_status(claim_id)
    result["rate_limit_remaining"] = limiter.remaining_global()

    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def fetch_article(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract text from a trusted-source article.

    Only URLs from the whitelist are accepted. Arbitrary blogs, social
    media, and user-generated content are REJECTED at the tool level.

    Args:
        url: Article URL (must be from a trusted domain).
        max_chars: Maximum characters to return (truncated with ellipsis).

    Returns:
        JSON with url, domain, title, text content, and truncation flag.
    """
    limiter = get_limiter()
    if not limiter.allow("fetch_article"):
        return json.dumps({"error": "Rate limit exceeded. Wait before retrying."})

    if not is_trusted(url):
        domain = extract_domain(url)
        return json.dumps({
            "error": f"URL rejected — '{domain}' is not in the trusted-source whitelist.",
            "trusted_domains_example": sorted(list(EVIDENCE_SOURCES))[:10],
        })

    result = fetch_article_text(url, max_chars=max_chars)
    result["rate_limit_remaining"] = limiter.remaining_global()
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def verify_claim(claim_text: str, claim_id: str = "") -> str:
    """Run full fact-check pipeline on a single claim.

    Searches trusted sources for evidence, retrieves articles, and uses
    an LLM to judge the claim against the evidence. Returns a verdict
    (corroborated / contradicted / developing / unverifiable) with
    confidence and source links.

    This is THE main tool. Use it to verify any factual claim.

    Args:
        claim_text: The complete claim text to verify.
        claim_id: Optional unique identifier for budget tracking.

    Returns:
        JSON with verdict, confidence (0.0-1.0), sources, and explanation.
    """
    limiter = get_limiter()
    if not limiter.allow("verify_claim"):
        return json.dumps({"error": "Rate limit exceeded. Wait before retrying.", "verdict": "unverifiable", "claims": []})

    engine = get_engine()
    if not engine.available:
        return json.dumps({
            "claims": [{"claim": claim_text, "verdict": "unverifiable", "confidence": 0.5, "sources": []}],
            "overall": "unverifiable",
            "explanation": f"Verification engine unavailable: {engine.error}",
            "elapsed_ms": 0,
        })

    claims_in = [{"claim_text": claim_text, "claim": claim_text}] if claim_id else [{"claim_text": claim_text}]
    result = engine.verify_claims(claims_in)

    # Add budget context
    if claim_id:
        result["claim_budget"] = limiter.claim_status(claim_id)
    result["rate_limit_remaining"] = limiter.remaining_global()

    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def batch_verify_claims(claims_json: str) -> str:
    """Verify multiple claims in a single batch.

    More efficient than calling verify_claim repeatedly. All claims
    share the same search and verify pipeline, reducing costs.

    Args:
        claims_json: JSON string of claim objects.
            Example: '[{"claim_text":"Eiffel Tower built in 1889."},{"claim_text":"Coffee cures cancer."}]'
            Each object may have 'claim' or 'claim_text' key.

    Returns:
        JSON with all claims (each with verdict/confidence/sources),
        overall verdict, and explanation.
    """
    limiter = get_limiter()
    if not limiter.allow("batch_verify_claims"):
        return json.dumps({"error": "Rate limit exceeded. Wait before retrying."})

    try:
        claims = json.loads(claims_json)
        if not isinstance(claims, list):
            return json.dumps({"error": "claims_json must be a JSON array of claim objects."})
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    # Check per-claim budgets
    for i, c in enumerate(claims):
        cid = c.get("claim_id", "") or c.get("id", "")
        if cid and not limiter.claim_can_search(cid) and not limiter.claim_can_fetch(cid):
            claims[i] = {**c, "verdict": "unverifiable", "confidence": 0.5,
                         "sources": [], "error": f"Budget exhausted for claim '{cid}'"}

    engine = get_engine()
    result = engine.verify_claims(claims)
    result["rate_limit_remaining"] = limiter.remaining_global()
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def resolve_entity(name: str) -> str:
    """Resolve a named entity (person, org, event, place) against trusted knowledge bases.

    Searches Wikipedia + Britannica (reference tier only). These results
    are for entity understanding, NOT for verdict evidence.

    Args:
        name: Entity name to resolve (e.g. "Eiffel Tower", "WHO", "Barack Obama").

    Returns:
        JSON with entity name, description, and reference URLs.
    """
    limiter = get_limiter()
    if not limiter.allow("resolve_entity"):
        return json.dumps({"error": "Rate limit exceeded. Wait before retrying."})

    # Use Wikipedia + Britannica for entity resolution
    result = _do_search(query=f"{name} site:wikipedia.org OR site:britannica.com", num_results=3, evidence_only=False)

    entity = {
        "name": name,
        "references": [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
            for r in result.get("organic", [])[:3]
        ],
        "rate_limit_remaining": limiter.remaining_global(),
    }
    return json.dumps(entity, indent=2, ensure_ascii=False)


@mcp.tool()
def get_cached_verdict(claim_text: str) -> str:
    """Check if a claim has been previously verified and cached.

    Not yet implemented — the MCP server is stateless. Caching is handled
    by the Chrome extension's chrome.storage.local mirror.
    Returns a note explaining this.

    Args:
        claim_text: The claim text to look up.

    Returns:
        JSON indicating cache status.
    """
    return json.dumps({
        "cached": False,
        "note": "MCP server is stateless. Cached verdicts are stored in the Chrome extension's local storage (chrome.storage.local). Use verify_claim for fresh verification.",
    })


@mcp.tool()
def list_trusted_sources(tier: int = 0) -> str:
    """List trusted source domains by credibility tier.

    Args:
        tier: Filter by tier (1=Wire Services, 2=Fact-Checkers, 3=Institutional, 4=Reference, 0=All).

    Returns:
        JSON mapping tier → list of domains.
    """
    from src.trusted_sources import TIER_1_NEWS, TIER_2_FACTCHECK, TIER_3_INSTITUTIONAL, TIER_4_REFERENCE
    tiers = {
        "tier_1_wire_services": sorted(TIER_1_NEWS),
        "tier_2_fact_checkers": sorted(TIER_2_FACTCHECK),
        "tier_3_institutional": sorted(TIER_3_INSTITUTIONAL),
        "tier_4_reference": sorted(TIER_4_REFERENCE),
    }
    if tier and 1 <= tier <= 4:
        key = f"tier_{tier}_" + {1: "wire_services", 2: "fact_checkers", 3: "institutional", 4: "reference"}[tier]
        return json.dumps({key: tiers[key]}, indent=2)
    return json.dumps({"total_trusted_domains": sum(len(v) for v in tiers.values()), **tiers}, indent=2)


# ── Resources ──────────────────────────────────────────────────────────


@mcp.resource("trusted://sources/evidence")
def trusted_evidence_sources() -> str:
    """List of evidence-tier trusted domains."""
    return json.dumps(sorted(EVIDENCE_SOURCES), indent=2)


@mcp.resource("trusted://sources/all")
def all_trusted_sources() -> str:
    """List of all trusted domains (including reference tier)."""
    return json.dumps(sorted(ALL_TRUSTED), indent=2)


@mcp.resource("config://status")
def server_status() -> str:
    """Server configuration and health status."""
    engine = get_engine()
    return json.dumps({
        "server": "VeriLens FactCheck MCP",
        "version": "1.0.0",
        "transports": ["stdio", "streamable-http"],
        "engine": {
            "available": engine.available,
            "error": engine.error,
            "model": os.environ.get("VERIFIER_MODEL", "qwen/qwen3.5-flash-02-23"),
        },
        "serper": {
            "configured": bool(os.environ.get("SERPER_API_KEY")),
            "rate_limit_remaining": get_limiter().remaining_global(),
        },
        "openrouter": {
            "configured": bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        },
        "trusted_domains": len(ALL_TRUSTED),
        "evidence_domains": len(EVIDENCE_SOURCES),
    }, indent=2)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VeriLens FactCheck MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio for local, streamable-http for production)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for streamable-http transport")
    parser.add_argument("--host", default="0.0.0.0", help="Host for streamable-http transport")
    args = parser.parse_args()

    logger.info("Starting VeriLens FactCheck MCP on %s transport", args.transport)
    engine = get_engine()
    if not engine.available:
        logger.warning("Verification engine import failed: %s — verify_claim will return unverifiable", engine.error)

    if args.transport == "streamable-http":
        logger.info("Listening on %s:%d", args.host, args.port)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
