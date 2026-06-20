"""
ToolOrchestrator — wraps LMGen.step() to intercept Moshi's inner-monologue
text stream, dispatch local tool calls, and inject results as forced text
tokens so the model speaks the answer.

The tool call is driven entirely by the model itself. Moshi was fine-tuned to
emit the reserved special tokens; the orchestrator only reacts to them — there
is no keyword matching or any other heuristic on the model's natural speech.

State machine (per session, reset on reset_streaming()):

    NORMAL  → 32000 <|tool_call|>        → IN_CALL   (buffer call tokens)
    IN_CALL → 32001 <|tool_end|>         → EXEC      (dispatch tool async)
    EXEC    → tool result ready          → INJECT    (force 32002 + result + 32003)
    INJECT  → inject queue drained       → NORMAL    (+ cooldown)

The model speaks freely during EXEC; during INJECT we force one result token
per frame so the audio is conditioned on the real tool output.
"""
import asyncio
import json
import re
import logging
from collections import deque
from typing import Optional

import torch
import sentencepiece

from .protocol import (
    OrchestratorState, ToolIntent,
    TOOL_CALL_ID, TOOL_END_ID, TOOL_RESULT_ID, TOOL_RESULT_END_ID,
)
from . import registry as reg

logger = logging.getLogger(__name__)

import os
# Frames (12.5 Hz) before another call can fire. The model keeps "wanting" to
# call for a while after a request, so this guards against repeats. Tunable via
# the TOOL_COOLDOWN env var without code changes.
_COOLDOWN_FRAMES = int(os.environ.get("TOOL_COOLDOWN", "75"))


class ToolOrchestrator:
    """
    Drop-in wrapper around LMGen for the per-frame inference loop.

    Usage::

        orchestrator = ToolOrchestrator(lm_gen, text_tokenizer)
        tokens = orchestrator.step(codes[:, :, c:c+1])
    """

    def __init__(
        self,
        lm_gen,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
    ) -> None:
        self.lm_gen    = lm_gen
        self.tokenizer = text_tokenizer
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        self.state             = OrchestratorState.NORMAL
        self._call_buf: list[int] = []     # tokens between <|tool_call|> … <|tool_end|>
        self._inject_queue: deque[int] = deque()
        self._cooldown         = 0
        self._pending_task: Optional[asyncio.Task] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        input_tokens: torch.Tensor,
        moshi_tokens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self._cooldown > 0:
            self._cooldown -= 1

        forced = self._next_inject_token()

        tokens = self.lm_gen.step(
            input_tokens=input_tokens,
            moshi_tokens=moshi_tokens,
            text_token=forced,
        )

        if tokens is not None:
            self._observe(int(tokens[0, 0, 0].item()))

        return tokens

    # ── Internal ──────────────────────────────────────────────────────────────

    def _next_inject_token(self) -> Optional[int]:
        if self.state is not OrchestratorState.INJECT:
            return None
        if not self._inject_queue:
            self.state     = OrchestratorState.NORMAL
            self._cooldown = _COOLDOWN_FRAMES
            return None
        return self._inject_queue.popleft()

    def _observe(self, token_id: int) -> None:
        # Model opens a tool call.
        if (
            token_id == TOOL_CALL_ID
            and self.state is OrchestratorState.NORMAL
            and self._cooldown == 0
        ):
            logger.info("[orchestrator] <|tool_call|> detected — buffering call")
            self._call_buf = []
            self.state     = OrchestratorState.IN_CALL
            return

        # Inside a call: buffer until the model closes it.
        if self.state is OrchestratorState.IN_CALL:
            if token_id == TOOL_END_ID:
                raw = self.tokenizer.decode(self._call_buf).strip()
                logger.info(f"[orchestrator] call decoded: {raw!r}")
                intent = _parse_call(raw)
                # Only dispatch a recognised tool; ignore garbled calls silently
                # (don't speak "unknown tool: ...").
                if intent and intent.name in reg._TOOLS:
                    self._call_buf = []
                    self.state     = OrchestratorState.EXEC
                    self._pending_task = asyncio.ensure_future(self._run_tool(intent))
                else:
                    logger.warning(f"[orchestrator] ignoring unrecognised call: {raw!r}")
                    self._call_buf = []
                    self.state     = OrchestratorState.NORMAL
                    self._cooldown = _COOLDOWN_FRAMES
            else:
                self._call_buf.append(token_id)

    async def _run_tool(self, intent: ToolIntent) -> None:
        try:
            result = await reg.call(intent.name, intent.args)
            logger.info(f"[orchestrator] result: {result!r}")
            result_tokens = self.tokenizer.encode(result)
            # Wrap with <|tool_result|> … <|tool_result_end|> — the exact format
            # the model was fine-tuned to consume.
            self._inject_queue.extend(
                [TOOL_RESULT_ID] + result_tokens + [TOOL_RESULT_END_ID]
            )
            self.state = OrchestratorState.INJECT
        except Exception as exc:
            logger.error(f"[orchestrator] tool task failed: {exc}")
            self.state     = OrchestratorState.NORMAL
            self._cooldown = _COOLDOWN_FRAMES
        finally:
            self._pending_task = None


def _parse_call(raw: str) -> Optional[ToolIntent]:
    """Parse the content the model emitted inside <|tool_call|>…<|tool_end|>.

    The model often phrases the call naturally ("get the time", "weather in
    London") rather than as the exact function name, so map by intent:
      - mentions "weather" → get_weather (+ city after "in", or a capitalised name)
      - mentions "time"/"clock" → get_time
    Falls back to strict "name args" / JSON parsing.
    """
    s = raw.strip()
    if not s:
        return None
    low = s.lower()

    if "weather" in low:
        city = None
        m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]*)", s)
        if m:
            city = m.group(1).strip()
        else:
            caps = re.findall(r"\b[A-Z][a-z]+\b", s)
            if caps:
                city = " ".join(caps)
        return ToolIntent("get_weather", {"city": city} if city else {})

    if "time" in low or "clock" in low:
        return ToolIntent("get_time", {})

    # Strict fallback: "get_weather London" or 'get_weather {"city": "X"}'
    parts = s.split(None, 1)
    name = parts[0]
    args: dict = {}
    if len(parts) > 1:
        rest = parts[1].strip()
        try:
            args = json.loads(rest)
        except json.JSONDecodeError:
            if name == "get_weather" and rest:
                args = {"city": rest}
    return ToolIntent(name=name, args=args)
