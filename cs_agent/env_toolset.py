"""Dynamic ADK toolset over the harness env API.

Tools are fetched live for the current session (keyed by the incoming A2A
contextId), so tools granted mid-conversation appear automatically. A generic
call_env_tool fallback covers anything not yet in the fetched list."""

import json
import os
from typing import Any, Optional

import httpx
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types

ENV_API_URL = os.environ["ENV_API_URL"].rstrip("/")
ENV_API_TOKEN = os.environ["ENV_API_TOKEN"]

_HEADERS = {"Authorization": f"Bearer {ENV_API_TOKEN}"}


def session_id(context: ReadonlyContext) -> str:
    """The env session id == the incoming A2A contextId (ADK keys its session
    on the contextId, so this is free)."""
    return context.session.id


async def _post_tool_call(sid: str, name: str, arguments: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ENV_API_URL}/sessions/{sid}/tools/{name}",
                json={"arguments": arguments},
                headers=_HEADERS,
            )
            if resp.status_code != 200:
                return {"error": True, "content": f"HTTP {resp.status_code}: {resp.text}"}
            return resp.json()
    except Exception as e:
        return {"error": True, "content": f"env tool call failed: {e!r}"}


async def call_env_tool(
    tool_name: str, arguments_json: str, tool_context: ToolContext
) -> dict:
    """Call any environment tool by name.

    Use this for tools that are not in your tool list yet (for example a tool
    you were just granted). `arguments_json` is a JSON object string, e.g.
    '{"user_id": "abc123"}'.
    """
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return {"error": True, "content": f"Invalid arguments JSON: {e}"}
    return await _post_tool_call(session_id(tool_context), tool_name, arguments)


class EnvApiTool(BaseTool):
    """One env tool, declared from its OpenAI-style schema."""

    def __init__(self, schema: dict):
        function = schema["function"]
        super().__init__(
            name=function["name"], description=function.get("description", "")
        )
        self._parameters = function.get("parameters")

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict:
        return await _post_tool_call(session_id(tool_context), self.name, args)


class EnvApiToolset(BaseToolset):
    """Serves the env tools for the current session plus the generic fallback."""

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        fallback: list[BaseTool] = [FunctionTool(call_env_tool)]
        if readonly_context is None:
            # No session yet (e.g. agent card construction).
            return fallback
        sid = session_id(readonly_context)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ENV_API_URL}/sessions/{sid}/tools", headers=_HEADERS
            )
        resp.raise_for_status()
        return [EnvApiTool(schema) for schema in resp.json()["tools"]] + fallback

    async def close(self) -> None:
        pass
