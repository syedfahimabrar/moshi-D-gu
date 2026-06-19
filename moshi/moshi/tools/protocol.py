"""
Tool-calling protocol for Moshi's inner monologue.

Moshi is fine-tuned to emit reserved special tokens in its text stream to call
a tool and to consume the injected result:

    <|tool_call|> get_weather {"city": "Dhaka"} <|tool_end|>
    <|tool_result|> ...result text... <|tool_result_end|>

This module defines the special-token IDs, the orchestrator state machine, and
the parsed-intent container. The tool call is decided by the model itself —
there is no keyword matching on the model's natural speech.
"""
from dataclasses import dataclass, field
from enum import Enum, auto


# Special token IDs added by the fine-tuning patch (text vocab 32000–32003).
TOOL_CALL_ID       = 32000  # <|tool_call|>
TOOL_END_ID        = 32001  # <|tool_end|>
TOOL_RESULT_ID     = 32002  # <|tool_result|>
TOOL_RESULT_END_ID = 32003  # <|tool_result_end|>


class OrchestratorState(Enum):
    NORMAL  = auto()  # monitoring the text stream
    IN_CALL = auto()  # buffering tokens between <|tool_call|> and <|tool_end|>
    EXEC    = auto()  # tool dispatched; model speaks freely while we wait
    INJECT  = auto()  # result ready; forcing text tokens one per frame


@dataclass
class ToolIntent:
    name: str
    args: dict = field(default_factory=dict)
