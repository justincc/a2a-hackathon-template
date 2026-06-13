"""Rho-Bank customer service agent: policy + env tools + KB search (RAG)."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent

from env_toolset import EnvApiToolset
from rag_tools import kb_search_bm25, kb_search_vector
from verify_tool import verify_helper_enabled, verify_user

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

# Deterministic verification helper (default on; CS_VERIFY_HELPER=0 disables).
# When enabled, verification runs through verify_user() in code so the
# log_verification write gets correct on-file values; the guidance routes the
# agent to it and away from calling log_verification by hand.
VERIFY_GUIDANCE = """

## Identity verification

To verify a user's identity, call verify_user(user_id, ...) with whatever
identity details the user has given you (date_of_birth, email, phone_number,
address). It confirms at least 2 of those 4 match the on-file record and, on
success, logs the verification for you with the correct on-file values.

- If you do not yet know the user_id, look it up first with
  get_user_information_by_name or get_user_information_by_email.
- Do NOT call log_verification yourself when verify_user is available, and do
  not verify more than once per conversation.
- If verify_user returns verified=false, ask the user for additional identity
  details and call verify_user again.
"""

_tools = [EnvApiToolset(), kb_search_bm25, kb_search_vector]
_instruction = POLICY_PATH.read_text() + RAG_GUIDANCE
if verify_helper_enabled():
    _tools.append(verify_user)
    _instruction += VERIFY_GUIDANCE

root_agent = LlmAgent(
    name="cs_agent",
    model=MODEL,
    instruction=_instruction,
    tools=_tools,
)
