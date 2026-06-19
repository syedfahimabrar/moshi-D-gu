"""
ToolOrchestrator — wraps LMGen.step() to intercept the inner-monologue text
stream, dispatch local tool calls, and inject results as forced text tokens.

Architecture note
-----------------
LMGen.step() already accepts an optional ``text_token`` int argument that
teacher-forces the text channel for the current frame *before* the depth
transformer runs (so audio is conditioned on the forced text — i.e. the model
speaks it). _step_text_prompt_core() uses exactly this mechanism to inject the
system persona prompt.  The orchestrator reuses it at conversation time to
inject real tool results.

State machine (per session, reset when reset_streaming() is called):

  NORMAL → (keyword in text buffer) → EXEC → (tool result ready) → INJECT → NORMAL
                                        ↑                                        │
                                        └──────────────── cooldown ──────────────┘

The model speaks freely during EXEC; during INJECT we force one result token
per frame.  A cooldown prevents re-triggering on the echoed result text.
"""
import asyncio
import logging
from collections import deque
from typing import Optional

import torch
import sentencepiece

from .protocol import OrchestratorState, ToolIntent, detect_intent
from . import registry as reg

logger = logging.getLogger(__name__)

# Rolling text buffer size (chars) for trigger detection
_BUFFER_MAX = 160
# Frames before the same trigger can fire again (~2 seconds at 12.5 Hz)
_COOLDOWN_FRAMES = 25


class ToolOrchestrator:
    """
    Drop-in wrapper around LMGen for the per-frame inference loop.

    Usage (server / async context)::

        orchestrator = ToolOrchestrator(lm_gen, text_tokenizer)
        # replace:  tokens = lm_gen.step(codes[:, :, c:c+1])
        # with:     tokens = orchestrator.step(codes[:, :, c:c+1])

    Usage (offline / sync context — must be inside asyncio.run())::

        # Same as above; ensure the outer coroutine is run with asyncio.run()
        # and sprinkle `await asyncio.sleep(0)` in the frame loop so pending
        # tool tasks get a chance to execute.
    """

    def __init__(
        self,
        lm_gen,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
    ) -> None:
        self.lm_gen = lm_gen
        self.tokenizer = text_tokenizer
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        self.state = OrchestratorState.NORMAL
        self._text_buffer = ""
        self._inject_queue: deque[int] = deque()
        self._cooldown = 0
        self._pending_task: Optional[asyncio.Task] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def step(
        self,
        input_tokens: torch.Tensor,
        moshi_tokens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Call once per 80 ms frame; mirrors LMGen.step() signature."""
        # Advance cooldown counter
        if self._cooldown > 0:
            self._cooldown -= 1

        # Pop next injection token (None → model samples freely this frame)
        forced = self._next_inject_token()

        tokens = self.lm_gen.step(
            input_tokens=input_tokens,
            moshi_tokens=moshi_tokens,
            text_token=forced,
        )

        if tokens is not None:
            text_token_id = int(tokens[0, 0, 0].item())
            self._observe(text_token_id)

        return tokens

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _next_inject_token(self) -> Optional[int]:
        if self.state is not OrchestratorState.INJECT:
            return None
        if not self._inject_queue:
            # Queue drained → return to NORMAL and start cooldown
            self.state = OrchestratorState.NORMAL
            self._cooldown = _COOLDOWN_FRAMES
            return None
        token = self._inject_queue.popleft()
        return token

    def _observe(self, token_id: int) -> None:
        """Update rolling text buffer and fire tool calls when a trigger matches."""
        # Skip PAD (3) and EPAD (0) and out-of-range ids
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
                logger.info(f"[orchestrator] detected intent: {intent}")
                self._text_buffer = ""
                self.state = OrchestratorState.EXEC
                self._pending_task = asyncio.ensure_future(self._run_tool(intent))

    async def _run_tool(self, intent: ToolIntent) -> None:
        try:
            result = await reg.call(intent.name, intent.args)
            logger.info(f"[orchestrator] tool result: {result!r}")
            result_tokens = self.tokenizer.encode(result)
            self._inject_queue.extend(result_tokens)
            self.state = OrchestratorState.INJECT
        except Exception as exc:
            logger.error(f"[orchestrator] tool task failed: {exc}")
            self.state = OrchestratorState.NORMAL
            self._cooldown = _COOLDOWN_FRAMES
        finally:
            self._pending_task = None
