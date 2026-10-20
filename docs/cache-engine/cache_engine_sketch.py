"""cache-engine (DRAFT) — Context Economy Engine for Hermes.

Register as an llm_request middleware: it clips oversized tool results and leaves a
pointer line so the model can re-fetch selectively. Nothing here touches messages
before ``frozen_upto`` — rewriting history invalidates the provider prompt cache.

Enable with plugins.enabled: [cache-engine] in config.yaml.
"""

from __future__ import annotations

CAPS = {"web_extract": 900, "skill_view": 900, "session_search": 900, "memory_recall": 900}
DEFAULT_CAP = 1200
HARD_CAP = 6000


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def clip(text: str, cap: int) -> tuple[str, bool]:
    """Keep head 60% / tail 40% — errors and results live at the edges."""
    if _approx_tokens(text) <= cap:
        return text, False
    chars = cap * 3
    head = int(chars * 0.6)
    tail = chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n… [{omitted} chars omitted] …\n{text[-tail:]}", True


def middleware(payload, **kwargs):
    """llm_request middleware: payload['messages'] is the wire message list."""
    messages = payload.get("messages") or []
    frozen = int(kwargs.get("frozen_upto") or 0)
    changed = False
    for m in messages[frozen:]:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        body = m.get("content")
        if not isinstance(body, str):
            continue
        cap = min(CAPS.get(m.get("name", ""), DEFAULT_CAP), HARD_CAP)
        clipped, did = clip(body, cap)
        if did:
            m["content"] = clipped
            changed = True
    return payload if not changed else payload


def register(ctx) -> None:
    ctx.register_middleware("llm_request", middleware)
