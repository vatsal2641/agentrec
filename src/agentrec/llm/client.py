"""LLM clients behind one tiny interface: chat(messages, tools) -> LLMResponse.

Messages / tools use the OpenAI "chat completions + function calling" format,
which Ollama, vLLM, Groq, Together, OpenAI and others all accept. So the
same agent code is designed to run against a free local model (NOTE: this client
was written without network access and has not been tested against a live
endpoint; test it first with scripts/demo.py --llm ...):

    ollama pull qwen2.5:7b-instruct
    ollama serve
    OpenAICompatClient("http://localhost:11434/v1", "qwen2.5:7b-instruct")

WHAT TOOL CALLING IS (mechanically)
  1. We send the conversation + a JSON-schema description of each tool.
  2. The model replies either with text, or with {"tool_calls": [{name, arguments(JSON)}]}.
  3. OUR code executes the function and appends its result as a "tool" message.
  4. Repeat. The model never executes anything; it only proposes calls.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None     # set when the model emitted invalid JSON


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0


class LLMClient(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenAICompatClient:
    """Minimal HTTP client for any /v1/chat/completions endpoint with tool calling."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0,
                 temperature: float = 0.0) -> None:
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.temperature = timeout, temperature
        self.name = f"openai_compat:{model}"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        import httpx  # optional dependency

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "tools": tools,
                   "tool_choice": "auto", "temperature": self.temperature}
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments", "{}")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                err = None
            except json.JSONDecodeError as e:
                args, err = {}, f"invalid JSON arguments: {e}"
            calls.append(ToolCall(tc.get("id", f"call_{i}"), fn.get("name", ""), args, err))
        return LLMResponse(msg.get("content"), calls, data.get("usage", {}), time.perf_counter() - t0)


class ScriptedLLM:
    """Replays a fixed list of responses. Used in tests to force specific behaviour
    (e.g. a hallucinated item id) and check that the agent's guardrails catch it."""

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls += 1
        if not self.responses:
            return LLMResponse("I have nothing more to add.")
        return self.responses.pop(0)
