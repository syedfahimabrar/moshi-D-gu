"""Unit tests for the special-token tool protocol — no GPU or weights needed."""
from ..protocol import (
    OrchestratorState, ToolIntent,
    TOOL_CALL_ID, TOOL_END_ID, TOOL_RESULT_ID, TOOL_RESULT_END_ID,
)
from ..orchestrator import _parse_call


def test_special_token_ids():
    assert (TOOL_CALL_ID, TOOL_END_ID, TOOL_RESULT_ID, TOOL_RESULT_END_ID) == (
        32000, 32001, 32002, 32003
    )


def test_parse_call_no_args():
    intent = _parse_call("get_time")
    assert intent == ToolIntent(name="get_time", args={})


def test_parse_call_with_json_args():
    intent = _parse_call('get_weather {"city": "Dhaka"}')
    assert intent is not None
    assert intent.name == "get_weather"
    assert intent.args == {"city": "Dhaka"}


def test_parse_call_multiword_city():
    intent = _parse_call('get_weather {"city": "New York"}')
    assert intent.args == {"city": "New York"}


def test_parse_call_malformed_json_falls_back_to_empty_args():
    intent = _parse_call("get_weather {city: Dhaka")  # invalid JSON
    assert intent is not None
    assert intent.name == "get_weather"
    assert intent.args == {}


def test_parse_call_empty():
    assert _parse_call("") is None
    assert _parse_call("   ") is None


def test_orchestrator_state_enum():
    states = {
        OrchestratorState.NORMAL,
        OrchestratorState.IN_CALL,
        OrchestratorState.EXEC,
        OrchestratorState.INJECT,
    }
    assert len(states) == 4
