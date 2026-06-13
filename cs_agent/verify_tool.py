"""Deterministic identity-verification helper for the CS agent.

Encapsulates the policy's verification flow in code so the high-frequency,
argument-sensitive log_verification write is correct by construction:
  1. read the user's on-file record (get_user_information_by_id),
  2. check that >=2 of {date_of_birth, email, phone_number, address} the user
     PROVIDED match the on-file values,
  3. on success, call log_verification with the exact ON-FILE values plus the
     timestamp parsed from get_current_time().

The LLM still decides WHEN to verify and supplies the user's claimed details;
this removes the hand-copying / formatting errors from the actual write. Enabled
by default; set CS_VERIFY_HELPER=0 to fall back to manual verification.
"""

import os
import re

from google.adk.tools import ToolContext

from env_toolset import _post_tool_call, session_id

CHECK_FIELDS = ("date_of_birth", "email", "phone_number", "address")


def verify_helper_enabled() -> bool:
    return os.environ.get("CS_VERIFY_HELPER", "1").lower() not in ("0", "false", "no")


def _parse_record(text: str) -> dict:
    """Parse the first record from query_database_tool's formatted output.

    That output looks like:
        Found 1 record(s) in 'users':

        1. Record ID: 123
           name: Mia Smith
           user_id: 123
           ...
    """
    record: dict[str, str] = {}
    started = False
    for line in (text or "").splitlines():
        m = re.match(r"\s+([a-z_]+):\s*(.*)$", line)
        if m:
            started = True
            key, val = m.group(1), m.group(2).strip()
            record.setdefault(key, val)  # first record only
        elif started and not line.strip():
            break  # blank line ends the first record block
    return record


def _parse_time(text: str) -> str:
    """Pull the timestamp out of get_current_time()'s sentence reply."""
    m = re.search(r"current time is\s+(.*?)\.?\s*$", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _matches(field: str, provided: str, on_file: str) -> bool:
    p, o = (provided or "").strip().lower(), (on_file or "").strip().lower()
    if not p or not o:
        return False
    if field in ("phone_number", "date_of_birth"):
        return re.sub(r"\D", "", p) == re.sub(r"\D", "", o)
    if field == "address":
        p, o = re.sub(r"\s+", " ", p), re.sub(r"\s+", " ", o)
        return p == o or p in o or o in p  # tolerate partial address
    return re.sub(r"\s+", " ", p) == re.sub(r"\s+", " ", o)  # email etc.


async def verify_user(
    user_id: str,
    tool_context: ToolContext,
    date_of_birth: str = "",
    email: str = "",
    phone_number: str = "",
    address: str = "",
) -> dict:
    """Verify a user's identity and, on success, log the verification record.

    Provide the user_id plus whichever identity details the user gave you
    (date_of_birth, email, phone_number, address). The user must correctly
    provide at least 2 of those 4 (name and user_id do NOT count). On success
    this logs the verification automatically using the on-file values -- do NOT
    also call log_verification yourself.

    Returns a dict with 'verified' (bool), 'matched_fields', and a 'message'.
    """
    sid = session_id(tool_context)
    res = await _post_tool_call(sid, "get_user_information_by_id", {"user_id": user_id})
    if res.get("error"):
        return {"verified": False, "message": f"Could not look up user: {res.get('content')}"}
    record = _parse_record(res.get("content", ""))
    if not record.get("user_id"):
        return {
            "verified": False,
            "message": f"No user found for user_id '{user_id}'. Confirm the user_id first.",
        }

    provided = {
        "date_of_birth": date_of_birth,
        "email": email,
        "phone_number": phone_number,
        "address": address,
    }
    matched = [f for f in CHECK_FIELDS if _matches(f, provided[f], record.get(f, ""))]
    if len(matched) < 2:
        return {
            "verified": False,
            "matched_fields": matched,
            "message": (
                f"Identity NOT verified: only {len(matched)} of the required 2 "
                "identity fields matched. Ask the user for more details (date of "
                "birth, email, phone number, or address) and call verify_user again."
            ),
        }

    tres = await _post_tool_call(sid, "get_current_time", {})
    time_verified = _parse_time(tres.get("content", ""))
    log = await _post_tool_call(
        sid,
        "log_verification",
        {
            "name": record.get("name", ""),
            "user_id": record.get("user_id", user_id),
            "address": record.get("address", ""),
            "email": record.get("email", ""),
            "phone_number": record.get("phone_number", ""),
            "date_of_birth": record.get("date_of_birth", ""),
            "time_verified": time_verified,
        },
    )
    if log.get("error"):
        return {"verified": False, "message": f"Verification logging failed: {log.get('content')}"}
    return {
        "verified": True,
        "matched_fields": matched,
        "message": f"Identity verified ({len(matched)} fields matched) and logged.",
    }
