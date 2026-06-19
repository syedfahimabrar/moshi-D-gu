"""
Minimal tool registry — no agent framework, no LLM-in-the-loop.

Add a new tool by decorating an async function with @tool.
The function signature defines its call interface; args are passed as kwargs.
"""
import asyncio
import datetime
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TOOLS: dict[str, Callable] = {}
_TIMEOUT = 5.0  # seconds per tool call


def tool(fn: Callable) -> Callable:
    _TOOLS[fn.__name__] = fn
    return fn


async def call(name: str, args: dict) -> str:
    fn = _TOOLS.get(name)
    if fn is None:
        return f"(unknown tool: {name})"
    try:
        return str(await asyncio.wait_for(fn(**args), timeout=_TIMEOUT))
    except asyncio.TimeoutError:
        logger.warning(f"[tool] {name} timed out after {_TIMEOUT}s")
        return f"(tool {name} timed out)"
    except Exception as exc:
        logger.error(f"[tool] {name} raised {exc}")
        return f"(tool {name} error)"


def get_tool_spec_prompt() -> str:
    """One-line description of available tools, for injection into the system prompt."""
    names = list(_TOOLS.keys())
    return (
        "You have access to real-time tools: " + ", ".join(names) + ". "
        "When answering questions about the current time or weather, "
        "say the word 'time' or 'weather' naturally in your response."
    )


# ── Built-in tools ──────────────────────────────────────────────────────────

@tool
async def get_time() -> str:
    now = datetime.datetime.now()
    # e.g. "3:45 PM on Thursday, June 19, 2026"
    return now.strftime("%-I:%M %p on %A, %B %-d, %Y")


@tool
async def get_weather(city: str = "local") -> str:
    """Fetch current weather using wttr.in (no API key required)."""
    try:
        import aiohttp  # already a dep via server.py
        url = f"https://wttr.in/{city}?format=3"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=_TIMEOUT)
            ) as resp:
                if resp.status == 200:
                    return (await resp.text()).strip()
        return f"Weather unavailable for {city}"
    except Exception as exc:
        logger.error(f"[tool] get_weather({city!r}): {exc}")
        return f"Weather unavailable for {city}"
