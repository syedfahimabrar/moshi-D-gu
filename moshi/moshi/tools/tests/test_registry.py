"""Unit tests for registry.py — no GPU, no model weights required."""
import asyncio
import datetime
import pytest

from ..registry import call, get_time, get_weather, get_tool_spec_prompt, _TOOLS


def test_tools_registered():
    assert "get_time" in _TOOLS
    assert "get_weather" in _TOOLS


def test_get_tool_spec_prompt_mentions_tools():
    spec = get_tool_spec_prompt()
    assert "get_time" in spec
    assert "get_weather" in spec


def test_get_time_returns_string():
    result = asyncio.run(get_time())
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_time_contains_am_pm():
    result = asyncio.run(get_time())
    assert "AM" in result or "PM" in result


def test_get_time_contains_current_year():
    result = asyncio.run(get_time())
    year = str(datetime.datetime.now().year)
    assert year in result


def test_call_unknown_tool():
    result = asyncio.run(call("nonexistent_tool", {}))
    assert "unknown" in result.lower()


def test_call_get_time():
    result = asyncio.run(call("get_time", {}))
    assert isinstance(result, str)
    assert "AM" in result or "PM" in result


def test_call_timeout():
    """Registering a slow tool and calling it should hit timeout gracefully."""
    async def _slow():
        await asyncio.sleep(999)
        return "never"

    import moshi.moshi.tools.registry as reg
    original_timeout = reg._TIMEOUT
    reg._TOOLS["_slow_test"] = _slow
    reg._TIMEOUT = 0.05  # 50 ms

    result = asyncio.run(call("_slow_test", {}))
    assert "timed out" in result.lower()

    # Cleanup
    del reg._TOOLS["_slow_test"]
    reg._TIMEOUT = original_timeout
