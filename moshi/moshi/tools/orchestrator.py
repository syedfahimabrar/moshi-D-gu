"""
ToolOrchestrator — wraps LMGen.step() to intercept the inner-monologue text
stream, dispatch local tool calls, and inject results as forced text tokens.

Two detection paths (both active):

  Phase-B (fine-tuned model): watch for special token IDs 32000–32003.
    NORMAL → 32000 (<|tool_call|>) → IN_CALL (buffer tokens)
    IN_CALL → 32001 (<|tool_end|>)  → EXEC (dispatch tool)
    EXEC → result ready → INJECT (force 32002 + result + 32003)
    INJECT → queue drained → NORMAL + cooldown

  Phase-A fallback (keyword, training-free): watch rolling text buffer for
    keywords ("time", "weather") in NORMAL state. Injects raw result text.
"""
import asyncio
import logging
from collections import deque
from typing import Optional

import torch
import sentencepiece

from .protocol import (
    OrchestratorState, ToolIntent, detect_intent,
    TOOL_CALL_ID, TOOL_END_ID, TOOL_RESULT_ID, TOOL_RESULT_END_ID,
)
from . import registry as reg

logger = logging.getLogger(__name__)

_BUFFER_MAX      = 160  # rolling text buffer chars (keyword fallback)
_COOLDOWN_FRAMES = 75   # ~6 s at 12.5 Hz before same trigger can fire again


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
        self._text_buffer      = ""        # keyword fallback
        self._call_buf: list[int] = []     # token IDs between <|tool_call|> … <|tool_end|>
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
        # ── Phase-B: structured special-token protocol ────────────────────────
        if token_id == TOOL_CALL_ID and self.state is OrchestratorState.NORMAL and self._cooldown == 0:
            logger.info("[orchestrator] <|tool_call|> detected — buffering call")
            self._call_buf = []
            self.state     = OrchestratorState.IN_CALL
            return

        if self.state is OrchestratorState.IN_CALL:
            if token_id == TOOL_END_ID:
                # Decode buffered tokens → "get_time" or "get_weather {\"city\":\"X\"}"
                raw = self.tokenizer.decode(self._call_buf).strip()
                logger.info(f"[orchestrator] call buffer decoded: {raw!r}")
                intent = _parse_call(raw)
                if intent:
                    self._call_buf = []
                    self.state     = OrchestratorState.EXEC
                    self._pending_task = asyncio.ensure_future(self._run_tool(intent))
                else:
                    logger.warning(f"[orchestrator] unrecognised call: {raw!r}")
                    self.state     = OrchestratorState.NORMAL
                    self._cooldown = _COOLDOWN_FRAMES
            else:
                self._call_buf.append(token_id)
            return

        # ── Phase-A fallback: keyword detection in rolling text buffer ─────────
        if token_id in (0, 3) or token_id >= self.tokenizer.get_piece_size():
            return
        piece = self.tokenizer.id_to_piece(token_id).replace("▁", " ")
        self._text_buffer += piece
        if len(self._text_buffer) > _BUFFER_MAX:
            self._text_buffer = self._text_buffer[-_BUFFER_MAX:]

        if (
            self.state is OrchestratorState.NORMAL
            and self._cooldown == 0
            and self._pending_task is None
        ):
            intent = detect_intent(self._text_buffer)
            if intent:
                logger.info(f"[orchestrator] keyword intent: {intent}")
                self._text_buffer = ""
                self.state        = OrchestratorState.EXEC
                self._pending_task = asyncio.ensure_future(self._run_tool(intent, structured=False))

    async def _run_tool(self, intent: ToolIntent, structured: bool = True) -> None:
        try:
            result = await reg.call(intent.name, intent.args)
            logger.info(f"[orchestrator] result: {result!r}")
            result_tokens = self.tokenizer.encode(result)
            if structured:
                # Wrap with <|tool_result|> … <|tool_result_end|> so the model
                # sees the same format it was fine-tuned on.
                self._inject_queue.extend([TOOL_RESULT_ID] + result_tokens + [TOOL_RESULT_END_ID])
            else:
                self._inject_queue.extend(result_tokens)
            self.state = OrchestratorState.INJECT
        except Exception as exc:
            logger.error(f"[orchestrator] tool task failed: {exc}")
            self.state     = OrchestratorState.NORMAL
            self._cooldown = _COOLDOWN_FRAMES
        finally:
            self._pending_task = None


def _parse_call(raw: str) -> Optional[ToolIntent]:
    """Parse 'get_time' or 'get_weather {"city": "Dhaka"}' into a ToolIntent."""
    import json
    parts = raw.split(None, 1)
    if not parts:
        return None
    name = parts[0]
    args: dict = {}
    if len(parts) > 1:
        try:
            args = json.loads(parts[1])
        except json.JSONDecodeError:
            pass
    return ToolIntent(name=name, args=args)
