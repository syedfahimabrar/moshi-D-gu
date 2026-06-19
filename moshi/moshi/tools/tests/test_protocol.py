"""Unit tests for protocol.py — no GPU or model weights required."""
import pytest
from ..protocol import detect_intent, OrchestratorState, ToolIntent


def test_detect_time():
    intent = detect_intent("The current time is")
    assert intent is not None
    assert intent.name == "get_time"
    assert intent.args == {}


def test_detect_time_question():
    intent = detect_intent("What time is it right now?")
    assert intent is not None
    assert intent.name == "get_time"


def test_detect_weather_city():
    intent = detect_intent("Let me check the weather in London for you.")
    assert intent is not None
    assert intent.name == "get_weather"
    assert intent.args == {"city": "London"}


def test_detect_weather_city_multiword():
    intent = detect_intent("The weather in New York is")
    assert intent is not None
    assert intent.name == "get_weather"
    assert intent.args == {"city": "New York"}


def test_detect_weather_no_city():
    intent = detect_intent("What's the weather like today?")
    assert intent is not None
    assert intent.name == "get_weather"
    assert intent.args == {"city": "local"}


def test_no_match():
    assert detect_intent("How are you doing today?") is None
    assert detect_intent("Tell me a joke.") is None
    assert detect_intent("") is None


def test_weather_beats_time():
    # "weather" should not trigger get_time even if both words present
    intent = detect_intent("The weather right now and the time are both interesting.")
    assert intent is not None
    assert intent.name == "get_weather"


def test_partial_buffer_no_false_trigger():
    # Partial words should not fire; only complete keyword matches
    assert detect_intent("weath") is None
    assert detect_intent("ti") is None


def test_orchestrator_state_enum():
    assert OrchestratorState.NORMAL != OrchestratorState.EXEC
    assert OrchestratorState.EXEC != OrchestratorState.INJECT
