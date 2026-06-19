"""
Minimal tool registry — no agent framework, no LLM-in-the-loop.

Add a new tool by decorating an async function with @tool.
The function signature defines its call interface; args are passed as kwargs.
"""
import asyncio
import datetime
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_TOOLS: dict[str, Callable] = {}
_TIMEOUT = 5.0  # seconds per tool call

# Cities to fetch when no specific city is requested
_CONTEXT_CITIES = ["Stockholm", "London", "New York", "Tokyo", "Sydney", "Dubai"]


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
    names = list(_TOOLS.keys())
    return (
        "You have access to real-time tools: " + ", ".join(names) + ". "
        "When someone asks about the current time or weather, "
        "give them the accurate answer from those tools."
    )


# ── Built-in tools ──────────────────────────────────────────────────────────

@tool
async def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%-I:%M %p on %A, %B %-d, %Y")


async def _fetch_one(session, city: str) -> str:
    """Fetch weather for a single city; returns 'City: condition, temp' or error."""
    import aiohttp
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=%C,+%t"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=_TIMEOUT)) as resp:
            if resp.status == 200:
                raw = (await resp.text()).strip()
                return f"{city}: {raw}"
    except Exception:
        pass
    return f"{city}: unavailable"


@tool
async def get_weather(city: str = "") -> str:
    """Fetch current weather. If no city given, returns context for all major cities."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        if city:
            return await _fetch_one(session, city)
        # No specific city — fetch all context cities in parallel
        results = await asyncio.gather(*[_fetch_one(session, c) for c in _CONTEXT_CITIES])
    return " | ".join(results)
