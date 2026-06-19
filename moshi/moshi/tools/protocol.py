"""
Trigger-pattern detection for the tool-calling inner-monologue interceptor.

Phase-A approach (training-free): watch Moshi's own text stream (its inner
monologue / speech output) for keywords that indicate the model is about to
talk about time or weather.  When a keyword fires, the orchestrator dispatches
a local tool call and injects the real result as forced text tokens so the
model *speaks* the correct answer.
"""
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional


# Special token IDs added by the fine-tuning patch
TOOL_CALL_ID       = 32000  # <|tool_call|>
TOOL_END_ID        = 32001  # <|tool_end|>
TOOL_RESULT_ID     = 32002  # <|tool_result|>
TOOL_RESULT_END_ID = 32003  # <|tool_result_end|>


class OrchestratorState(Enum):
    NORMAL  = auto()  # monitoring; no active tool call
    IN_CALL = auto()  # buffering tokens between <|tool_call|> and <|tool_end|>
    EXEC    = auto()  # tool dispatched; model speaks freely while we wait
    INJECT  = auto()  # result ready; forcing text tokens one per frame


@dataclass
class ToolIntent:
    name: str
    args: dict


# Common non-city words that terminate a city name in "weather in <CITY> today".
_STOP = r'(?:today|now|like|is|are|will|this|please|how|what|currently|there|here|outside)'

# Each entry: (compiled regex, tool_name, args_factory(match) -> dict)
# Listed most-specific first so "weather in London" beats bare "weather".
_TRIGGERS: list[tuple[re.Pattern, str, Callable]] = [
    (
        re.compile(
            r'\bweather\b.{0,40}?\bin\s+'
            r'((?:(?!' + _STOP + r'\b)[A-Za-z]+(?:\s+(?!' + _STOP + r'\b)|$))*[A-Za-z]+)',
            re.IGNORECASE,
        ),
        "get_weather",
        lambda m: {"city": m.group(1).strip()},
    ),
    (
        re.compile(r'\bweather\b', re.IGNORECASE),
        "get_weather",
        lambda _: {"city": "local"},
    ),
    (
        re.compile(r"\b(time|o'clock|what time)\b", re.IGNORECASE),
        "get_time",
        lambda _: {},
    ),
]


def detect_intent(text: str) -> Optional[ToolIntent]:
    """Return the first matching ToolIntent in *text*, or None."""
    for pattern, tool_name, args_fn in _TRIGGERS:
        m = pattern.search(text)
        if m:
            return ToolIntent(name=tool_name, args=args_fn(m))
    return None
