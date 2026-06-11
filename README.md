# VeriLens FactCheck MCP Server

**Sandboxed, whitelist-restricted fact-checking for AI agents.**

7 MCP tools + 3 resources that let AI agents verify factual claims against trusted news sources — with rate limits and per-claim budgets enforced at the protocol boundary.

## Why MCP?

If an agent can "verify" a claim against an arbitrary blog or AI-written aggregator, the entire premise of flagging misinformation collapses. This MCP server enforces a **trusted-source whitelist** (40+ domains across 4 credibility tiers), **per-tool rate limits**, and **per-claim call budgets** — not as soft prompt instructions the agent can ignore, but at the protocol level.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/smammahdi/verilens-factcheck-mcp.git
cd verilens-factcheck-mcp

# 2. Install
pip install -e .

# 3. Set API keys
cp .env.example .env
# Edit .env with your OpenRouter and Serper keys

# 4. Run (stdio — for Claude Desktop, Copilot, etc.)
python server.py

# Or streamable HTTP (production)
python server.py --transport streamable-http --port 8080
```

## Tools

| Tool | Description |
|------|-------------|
| `search_trusted_sources` | Search Serper API, post-filter results to whitelisted domains only |
| `fetch_article` | Fetch full text from a trusted-source URL (rejects untrusted at tool level) |
| `verify_claim` | Full pipeline: search → fetch articles → LLM judges → verdict + sources |
| `batch_verify_claims` | Batch verification of multiple claims (avoids per-claim overhead) |
| `resolve_entity` | Resolve named entities against Wikipedia/Britannica |
| `get_cached_verdict` | Check if a claim was previously verified |
| `list_trusted_sources` | List all whitelisted domains by credibility tier |

## Resources

| URI | Description |
|-----|-------------|
| `trusted://sources/evidence` | Evidence-tier trusted domains (37 sources) |
| `trusted://sources/all` | All trusted domains (40 total) |
| `config://status` | Server health, model config, rate limit status |

## Trusted Sources

**Tier 1 — Wire Services (highest authority):**
Reuters, AP, AFP, BBC, NPR, PBS, C-SPAN, Al Jazeera, DW, France 24

**Tier 2 — Fact-Check Organisations:**
PolitiFact, FactCheck.org, Snopes, Full Fact, Lead Stories

**Tier 3 — Institutional / Academic:**
WHO, UN, World Bank, CDC, NIH, NASA, Nature, Science, The Lancet, NEJM

**Tier 4 — Reference (entity resolution only, NOT verdict evidence):**
Wikipedia, Wikidata, Britannica

## Rate Limits & Budgets

- **Global:** 60 requests/minute
- **Per-tool:** 30 requests/minute/tool
- **Per-claim budget:** max 8 searches + 12 fetches

## Environment Variables

```
OPENROUTER_API_KEY=sk-or-v1-...    # OpenRouter API key
SERPER_API_KEY=...                 # Serper.dev API key
VERIFIER_MODEL=qwen/qwen3.5-flash-02-23  # Default LLM model
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "verilens-factcheck": {
      "command": "python",
      "args": ["path/to/factcheck-mcp/server.py"]
    }
  }
}
```
