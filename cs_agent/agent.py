"""Rho-Bank customer service agent: policy + env tools + KB search (RAG)."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent

from env_toolset import EnvApiToolset
from rag_tools import kb_search_bm25, kb_search_vector

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
POLICY_PATH = Path(os.environ.get("KB_POLICY_PATH", "/app/kb/policy.md"))

# Retrieval mode switch (env-overridable; read at container start).
#   bm25_primary (default): keyword search (kb_search_bm25, local Redis,
#     ~instant) is the workhorse; semantic search (kb_search_vector) is a last
#     resort, because each vector search embeds the query via Vertex (~11s/call)
#     and was timing out the personal agent's synchronous CS call.
#   legacy: original behaviour, BM25 and vector presented as co-equal options.
RETRIEVAL_MODE = os.environ.get("CS_RETRIEVAL_MODE", "bm25_primary").lower()

RAG_GUIDANCE_BM25_PRIMARY = """

## Knowledge Base Access

You do NOT have the knowledge base inlined. Procedures, eligibility rules,
internal tool names, and scenario-specific guidance all live in the knowledge
base — search it before you act.

- Use kb_search_bm25(query) as your primary search. It is fast keyword search,
  so query with the specific terms that matter: the action (e.g. "open",
  "close", "dispute", "transfer"), the object (e.g. "savings account",
  "debit card"), and any distinguishing words ("personal" vs "business").
- If the results don't clearly contain what you need, refine the keywords and
  search again (one or two more focused tries).
- Only if kb_search_bm25 still returns nothing useful after rephrasing should
  you fall back to kb_search_vector(query) as a last resort — it is much
  slower, so do not use it routinely.
"""

RAG_GUIDANCE_LEGACY = """

## Knowledge Base Access

You do NOT have the knowledge base inlined. Before answering policy questions
or performing scenario-specific procedures, search the knowledge base:
- kb_search_bm25(query): keyword search.
- kb_search_vector(query): semantic search for natural-language questions.

Search before you act; procedures, eligibility rules, internal tool names,
and scenario-specific guidance all live in the knowledge base. If a search
comes up empty, rephrase and try again before telling the customer you can't
find the information.
"""

RAG_GUIDANCE = (
    RAG_GUIDANCE_LEGACY if RETRIEVAL_MODE == "legacy" else RAG_GUIDANCE_BM25_PRIMARY
)

root_agent = LlmAgent(
    name="cs_agent",
    model=MODEL,
    instruction=POLICY_PATH.read_text() + RAG_GUIDANCE,
    tools=[EnvApiToolset(), kb_search_bm25, kb_search_vector],
)
