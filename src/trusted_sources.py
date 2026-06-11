"""Trusted source whitelist for fact-checking.

The entire product premise collapses if the agent can "verify" a claim
against an arbitrary blog or AI-written aggregator. This module enforces
a strict whitelist of trusted outlets, with tiered credibility scoring.
"""

from __future__ import annotations

from urllib.parse import urlparse

# ── Tier 1: Traditional wire services & public broadcasters ──────────
TIER_1_NEWS = {
    "reuters.com",
    "apnews.com",
    "ap.org",
    "afp.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "pbs.org",
    "c-span.org",
    "aljazeera.com",
    "dw.com",
    "france24.com",
}

# ── Tier 2: Established fact-checking organisations ──────────────────
TIER_2_FACTCHECK = {
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "fullfact.org",
    "leadstories.com",
}

# ── Tier 3: Government, IGO, academic, & institutional ───────────────
TIER_3_INSTITUTIONAL = {
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "cdc.gov",
    "nih.gov",
    "europa.eu",
    "gov.uk",
    "canada.ca",
    "nasa.gov",
    "noaa.gov",
    "epa.gov",
    "nature.com",
    "science.org",
    "pnas.org",
    "arxiv.org",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "sciencedirect.com",
}

# ── Tier 4: Wikipedia (entity resolution only, never verdict evidence) ─
TIER_4_REFERENCE = {
    "wikipedia.org",
    "wikidata.org",
    "britannica.com",
}

# ── Aggregated whitelists ─────────────────────────────────────────────
ALL_TRUSTED: set[str] = (
    TIER_1_NEWS | TIER_2_FACTCHECK | TIER_3_INSTITUTIONAL | TIER_4_REFERENCE
)

# Subset that is authoritative enough to COUNT as evidence for a verdict.
# Wikipedia / Britannica are excluded — they are references for entity
# resolution, not evidence sources for fact-checking.
EVIDENCE_SOURCES: set[str] = (
    TIER_1_NEWS | TIER_2_FACTCHECK | TIER_3_INSTITUTIONAL
)

# Domains that are explicitly banned even if they appear in search results.
BLOCKED_DOMAINS = {
    "medium.com",          # self-published
    "substack.com",        # self-published
    "blogspot.com",        # self-published
    "wordpress.com",       # self-published
    "quora.com",           # user-generated
    "reddit.com",          # user-generated
    "twitter.com",         # user-generated
    "x.com",               # user-generated
    "facebook.com",        # user-generated
    "instagram.com",       # user-generated
    "tiktok.com",          # user-generated
    "youtube.com",         # user-generated
    "change.org",          # petition site
    "gofundme.com",        # fundraising
    "linkedin.com",        # self-published
}


def extract_domain(url: str) -> str:
    """Return the effective second-level domain (e.g. 'bbc.co.uk' not 'www.bbc.co.uk')."""
    try:
        host = urlparse(url).hostname or ""
        parts = host.lower().split(".")
        # Handle ccTLDs like .co.uk, .com.au
        if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac"):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return ""


def is_blocked(url: str) -> bool:
    """True if the domain is explicitly banned."""
    domain = extract_domain(url)
    return domain in BLOCKED_DOMAINS


def is_trusted(url: str) -> bool:
    """True if the domain is in the trusted whitelist and not blocked."""
    domain = extract_domain(url)
    return domain in ALL_TRUSTED and domain not in BLOCKED_DOMAINS


def is_evidence_source(url: str) -> bool:
    """True if the domain can count as verdict evidence (excludes Wikipedia et al)."""
    domain = extract_domain(url)
    return domain in EVIDENCE_SOURCES and domain not in BLOCKED_DOMAINS


def credibility_tier(url: str) -> int:
    """Return 1–4 for the credibility tier, or 0 if untrusted."""
    domain = extract_domain(url)
    if domain in BLOCKED_DOMAINS:
        return 0
    if domain in TIER_1_NEWS:
        return 1
    if domain in TIER_2_FACTCHECK:
        return 2
    if domain in TIER_3_INSTITUTIONAL:
        return 3
    if domain in TIER_4_REFERENCE:
        return 4
    return 0


def tier_label(tier: int) -> str:
    return {1: "Wire Service / Public Broadcaster", 2: "Fact-Check Organisation",
            3: "Institutional / Academic", 4: "Reference"}.get(tier, "Untrusted")


def filter_trusted(urls: list[str]) -> list[str]:
    """Return only trusted, non-blocked URLs."""
    return [u for u in urls if is_trusted(u)]


def filter_evidence(urls: list[str]) -> list[str]:
    """Return only evidence-quality trusted URLs."""
    return [u for u in urls if is_evidence_source(u)]
