#!/usr/bin/env python3
"""Append an additive `Keywords:` line to each KB document to improve BM25 recall.

In the default retrieval mode the knowledge base is searched with BM25 (keyword)
search, which misses a document when the user/agent phrases a request with words
that don't lexically appear in the doc's title (e.g. "cancel my account" vs a doc
titled "Closing ... Accounts"). This appends a deterministic line of SYNONYMS /
alternative phrasings derived from each doc's title so those queries still hit.

Purely ADDITIVE: never edits existing text, only appends one `Keywords:`
paragraph; idempotent (skips docs that already have one); changes NO facts (tool
names, arguments, thresholds, enum values are untouched).

Usage:
    python kb/add_keywords.py            # dry run: preview + format round-trip check
    python kb/add_keywords.py --apply    # write changes

After --apply, rebuild the index with KB_FORCE_REINDEX=1 (BM25-only mode makes
this a fast, text-only rebuild) so the new keywords are searchable.
"""
import json
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "documents"
MARKER = "\n\nKeywords: "

# stem (matched against title words by prefix) -> alternative phrasings to add
SYNONYMS = {
    "open":      ["open", "create", "start", "apply for", "set up", "sign up"],
    "clos":      ["close", "cancel", "terminate", "shut down", "deactivate", "delete"],
    "transfer":  ["transfer", "move", "send"],
    "deposit":   ["deposit", "pay in", "add money", "add funds"],
    "withdraw":  ["withdraw", "take out", "cash out"],
    "disput":    ["dispute", "contest", "challenge", "chargeback", "charge back", "wrong charge", "incorrect charge"],
    "filing":    ["file", "submit", "raise", "lodge", "open a case"],
    "activat":   ["activate", "enable", "turn on"],
    "freez":     ["freeze", "lock", "block", "suspend"],
    "order":     ["order", "request", "get"],
    "replac":    ["replace", "reissue", "replacement"],
    "updat":     ["update", "change", "edit", "modify", "amend", "correct"],
    "chang":     ["change", "update", "edit", "modify"],
    "appl":      ["apply", "application", "request"],
    "refer":     ["referral", "refer a friend", "invite"],
    "check":     ["checking", "checking account", "current account"],
    "saving":    ["savings", "savings account"],
    "debit":     ["debit card", "debit", "card"],
    "credit":    ["credit card", "credit", "card"],
    "account":   ["account", "accounts"],
    "transact":  ["transaction", "payment", "charge", "purchase"],
    "fund":      ["funds", "money", "balance", "cash"],
    "fraud":     ["fraud", "unauthorized", "scam", "stolen"],
    "interest":  ["interest", "apr", "apy", "rate"],
    "fee":       ["fee", "charge", "cost"],
    "provision": ["provisional credit", "temporary credit", "interim refund"],
    "eligib":    ["eligibility", "eligible", "qualify", "requirements"],
    "personal":  ["personal"],
    "business":  ["business"],
    "address":   ["address", "mailing address"],
    "email":     ["email", "email address"],
    "phone":     ["phone", "phone number", "mobile"],
    "verif":     ["verify", "verification", "identity"],
    "limit":     ["limit", "credit limit"],
    "statement": ["statement"],
    "loan":      ["loan"],
    "balanc":    ["balance", "funds", "money"],
    "status":    ["status", "check status", "track"],
}


def keywords_for(title: str) -> list[str]:
    title_l = title.lower()
    words = re.findall(r"[a-z]+", title_l)
    syns: set[str] = set()
    for stem, alts in SYNONYMS.items():
        if any(w.startswith(stem) for w in words):
            syns.update(alts)
    # drop phrases already present verbatim in the title (no added value)
    return sorted(s for s in syns if s not in title_l)


def main() -> None:
    apply = "--apply" in sys.argv
    files = sorted(DOCS_DIR.glob("*.json"))

    # Format round-trip safety check: dumping an unchanged doc must reproduce the
    # original bytes, so the only diff applied is the appended Keywords line.
    sample_raw = files[0].read_text()
    d0 = json.loads(sample_raw)
    rt = json.dumps(d0, indent=2, ensure_ascii=False) + ("\n" if sample_raw.endswith("\n") else "")
    print(f"format round-trip byte-identical: {rt == sample_raw}")

    changed = skipped = 0
    samples = []
    for f in files:
        raw = f.read_text()
        d = json.loads(raw)
        content = d.get("content", "")
        if "\nKeywords:" in content:
            skipped += 1
            continue
        kws = keywords_for(d.get("title", ""))
        if not kws:
            skipped += 1
            continue
        d["content"] = content + MARKER + ", ".join(kws)
        changed += 1
        if len(samples) < 6:
            samples.append((d["title"], ", ".join(kws)))
        if apply:
            out = json.dumps(d, indent=2, ensure_ascii=False)
            if raw.endswith("\n"):
                out += "\n"
            f.write_text(out)

    print(f"docs: {len(files)}  changed: {changed}  skipped: {skipped}  (apply={apply})")
    for t, k in samples:
        print(f"\n  {t}\n    Keywords: {k}")


if __name__ == "__main__":
    main()
