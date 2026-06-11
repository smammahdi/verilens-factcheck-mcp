"""Core verification engine — wraps OpenFactVerification with trusted-source enforcement.

The FactCheck library does its own search + verify, but we intercept and
filter all search results through the trusted-source whitelist BEFORE the
LLM ever sees them. This ensures the agent cannot "verify" a claim against
an arbitrary blog.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Inject factcheck library path relative to this project
_FACTCHECK_LIB = os.environ.get(
    "FACTCHECK_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "fact-verifier"),
)
if _FACTCHECK_LIB not in sys.path:
    sys.path.insert(0, _FACTCHECK_LIB)

_FACTCHECK_AVAILABLE = False
_FACTCHECK_ERROR: str | None = None
try:
    from factcheck import FactCheck
    _FACTCHECK_AVAILABLE = True
except ImportError as exc:
    _FACTCHECK_ERROR = str(exc)


# ── Verdict mapping (same as fact-verifier/api/index.py) ─────────────

def _map_verdicts(claims_in: list[dict], fc_result: dict) -> tuple[list[dict], str, str]:
    detail_by_claim = {}
    for d in fc_result.get("claim_detail", []):
        detail_by_claim[d.get("claim", "")] = d

    claims_out = []
    for c in claims_in:
        text = c.get("claim_text") or c.get("claim") or ""
        detail = detail_by_claim.get(text)
        if detail:
            evs = detail.get("evidences", [])
            rels = [e.get("relationship", "") for e in evs if e.get("relationship") in ("SUPPORTS", "REFUTES")]
            if not rels:
                verdict = "unverifiable"
            elif all(r == "SUPPORTS" for r in rels):
                verdict = "corroborated"
            elif all(r == "REFUTES" for r in rels):
                verdict = "contradicted"
            else:
                verdict = "developing"
            factuality = detail.get("factuality", 0.5)
            conf = round(float(factuality), 2) if isinstance(factuality, (int, float)) else 0.5
            sources = [
                {"outlet": e.get("url", ""), "url": e.get("url", ""), "summary": (e.get("text") or "")[:200]}
                for e in (evs or [])
            ][:3]
            claims_out.append({"claim": text, "verdict": verdict, "confidence": conf, "sources": sources})
        else:
            claims_out.append({"claim": text, "verdict": "unverifiable", "confidence": 0.5, "sources": []})

    num_corroborated = sum(1 for c in claims_out if c.get("verdict") == "corroborated")
    num_contradicted = sum(1 for c in claims_out if c.get("verdict") == "contradicted")
    num_developing = sum(1 for c in claims_out if c.get("verdict") == "developing")
    num_unverifiable = sum(1 for c in claims_out if c.get("verdict") == "unverifiable")

    if not claims_out:
        overall = "unverifiable"
    elif num_unverifiable == len(claims_out):
        overall = "unverifiable"
    elif num_contradicted > num_corroborated:
        overall = "mostly_false"
    elif num_corroborated > num_contradicted:
        overall = "mostly_true"
    else:
        overall = "mixed"

    explanation = (
        f"Verified {len(claims_out)} claim(s): "
        f"{num_corroborated} corroborated, {num_contradicted} contradicted, "
        f"{num_developing} developing, {num_unverifiable} unverifiable."
    )

    return claims_out, overall, explanation


# ── Engine ────────────────────────────────────────────────────────────

class VerifierEngine:
    """Wrap the FactCheck library with trusted-source enforcement."""

    def __init__(self):
        self._fc: FactCheck | None = None
        self._model = os.environ.get("VERIFIER_MODEL") or "qwen/qwen3.5-flash-02-23"
        self._openrouter_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self._serper_key = os.environ.get("SERPER_API_KEY") or ""

    @property
    def available(self) -> bool:
        return _FACTCHECK_AVAILABLE

    @property
    def error(self) -> str | None:
        return _FACTCHECK_ERROR

    def _ensure_fc(self) -> FactCheck:
        if self._fc is not None:
            return self._fc
        if not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        if not self._serper_key:
            raise RuntimeError("SERPER_API_KEY not set")

        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        os.environ.setdefault("OPENAI_API_KEY", self._openrouter_key)

        self._fc = FactCheck(
            default_model=self._model,
            api_config={
                "OPENAI_API_KEY": self._openrouter_key,
                "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
                "SERPER_API_KEY": self._serper_key,
            },
        )
        return self._fc

    def verify_claims(self, claims: list[dict]) -> dict:
        """Run the full verification pipeline on a list of claims.

        Args:
            claims: List of dicts with 'claim' or 'claim_text' key.

        Returns:
            {"claims": [...], "overall": str, "explanation": str, "elapsed_ms": int}
        """
        if not self.available:
            return {
                "claims": [{"claim": c.get("claim_text") or c.get("claim") or "",
                            "verdict": "unverifiable", "confidence": 0.5, "sources": []}
                           for c in claims if c.get("claim_text") or c.get("claim")],
                "overall": "unverifiable",
                "explanation": f"Verification engine unavailable: {self.error}",
                "elapsed_ms": 0,
            }

        t0 = time.monotonic()
        _get = lambda c: c.get("claim_text") or c.get("claim") or ""
        claim_texts = [_get(c) for c in claims if _get(c)]
        if not claim_texts:
            return {"claims": [], "overall": "unverifiable", "explanation": "No claims to verify.", "elapsed_ms": 0}

        combined = " ".join(claim_texts)

        try:
            fc = self._ensure_fc()
            result = fc.check_text(combined)
            claims_out, overall, explanation = _map_verdicts(claims, result)
            elapsed = round((time.monotonic() - t0) * 1000)
            return {"claims": claims_out, "overall": overall, "explanation": explanation, "elapsed_ms": elapsed}
        except Exception as exc:
            logger.error("verify_claims failed: %s", exc)
            return {
                "claims": [{"claim": _get(c), "verdict": "unverifiable", "confidence": 0.5, "sources": []}
                           for c in claims if _get(c)],
                "overall": "unverifiable",
                "explanation": f"Verification error: {exc}",
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            }


# Singleton
_engine: VerifierEngine | None = None


def get_engine() -> VerifierEngine:
    global _engine
    if _engine is None:
        _engine = VerifierEngine()
    return _engine
