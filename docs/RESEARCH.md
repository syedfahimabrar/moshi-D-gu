# Native Tool-Calling for Full-Duplex Speech LLMs (Moshi / PersonaPlex)

**A research log + system architecture.**

This document records the problem, the architecture we built, the experimental
path we followed (including the dead-ends and *why* they failed), the current
state, and a detailed plan for the remaining work. It is written to be
self-contained for a research write-up.

---

## 1. Problem statement

We want a **full-duplex speech-to-speech** model (NVIDIA **PersonaPlex**, a
fine-tune of Kyutai **Moshi**) to **call external tools mid-conversation** —
e.g. *"what time is it?"*, *"what's the weather in London?"* — and **speak the
real answer**, with the tool-calling decision made **inside the model itself**.

Hard constraints (the research framing):

- **No second LLM / agent framework.** No LangChain, no MCP orchestrator, no
  text-LLM doing the reasoning. The only external pieces are thin local Python
  functions that fetch data (`get_time`, `get_weather`).
- **Model-driven, not keyword-hacked.** The decision to call a tool must come
  from the model emitting a structured signal in its own output stream — not
  from pattern-matching its speech.
- **Live / streaming.** It must work in the real-time, full-duplex server, not
  just offline on canned transcripts.

---

## 2. Background: why Moshi makes this possible

### 2.1 The Inner Monologue

Per the Moshi paper (arXiv:2410.00037), at every **80 ms frame (12.5 Hz)** the
model predicts, in order:

1. **one text token** — the "inner monologue"
2. then the **audio tokens** for that frame (8 RVQ codebooks via the depth
   transformer / RQ-Transformer).

The text stream is a time-aligned, *slightly-leading* transcript of **Moshi's
own speech**. Two properties make it the ideal side-channel for tool calls:

- **It is text** — we can define structured control tokens in it.
- **It leads the audio** — forcing text tokens steers the speech that follows
  (this is exactly how Moshi is turned into a streaming TTS). We reuse this to
  *inject* a tool result and have the model speak it.

A subtlety that dominated the whole project: **the text stream is Moshi's
output, not the user's input.** The user's voice enters as a *separate* audio
stream (encoded by Mimi). The monologue does not contain the user's words.

### 2.2 The code tensor (what the LM actually consumes)

`LMModel.forward_train(codes)` and the streaming `LMGen.step` operate on a
tensor of **17 rows** (`num_codebooks = n_q + 1 = 17`, with `n_q=16`,
`dep_q=8`, `audio_offset=1`, `AUDIO_TOKENS_PER_STREAM=8`):

```
codes[B, 17, T]      T = number of 12.5 Hz frames
┌───────────────────────────────────────────────────────────────┐
│ row 0      : TEXT monologue        (Moshi)   ← we drive this    │
│ rows 1..8  : Moshi AUDIO codebooks (Moshi)   ← predicted        │
│ rows 9..16 : USER  AUDIO codebooks (human)   ← conditioning     │
└───────────────────────────────────────────────────────────────┘
delays = [0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]   (per-row temporal offset)
```

**Critical for training design:** `forward_train` computes loss only on the
**text row (0)** and **Moshi audio rows (1..8)**. The **user audio rows (9..16)
are pure conditioning** — never predicted. This is what lets us place the
user's spoken question in rows 9..16 and train the text row to react to it.

Per-frame mechanics in `LMGen.prepare_step_input`:
- user mic audio → Mimi.encode → written to rows `9..16` (`input_tokens`)
- Moshi's own audio prediction → rows `1..8`
- text token sampled from `text_logits` (temp 0.7, top-k 25 by default)

### 2.3 Special tokens

We extended the text vocabulary from 32000 → **32004** (patched checkpoint
`abrarfahim/moshi-tool-patched`, `text_card` auto-detected from the
`text_linear.weight` shape):

| id | token | meaning |
|----|-------|---------|
| 32000 | `<|tool_call|>` | model opens a tool call |
| 32001 | `<|tool_end|>` | model closes the call |
| 32002 | `<|tool_result|>` | orchestrator opens the injected result |
| 32003 | `<|tool_result_end|>` | orchestrator closes the result |

---

## 3. System architecture

Three subsystems: **(A) offline data generation**, **(B) LoRA fine-tuning**,
**(C) live serving with the orchestrator.**

### 3.1 Data generation (notebook `01_generate_data.ipynb` + `gen_tool_data.py`)

```
 gen_tool_data.examples()            edge-tts (15 voices)          Mimi.encode
 abstract Turn/Example   ──TTS──▶  user-question waveform  ──▶   8 user-audio codebooks
 (query, tool, result, reply)         (24 kHz mono, disk-cached)        │
        │                                                               ▼
        │  render_emit() → text tokens (call + result + reply)   build codes[17, T]:
        │                                                          row0  = text monologue
        └────────────────────────────────────────────────────▶   rows1-8 = silence (Mimi)
                                                                   rows9-16 = user audio
                                                                          │
                                                                          ▼
                                                        HF dataset  abrarfahim/moshi-tool-audio
                                                        (parquet: type, query, reply, voice,
                                                         audio[playable], codes[17×T], mask[T])
```

**Frame-exact alignment.** Each segment's waveform is padded to a whole number
of frames (`FRAME = 1920 samples = 24000/12.5`), so audio frames and text
tokens line up 1:1. Per example the timeline is:

```
[lead idle] [user speaks the question] [gap] [<|tool_call|> name <|tool_end|>] [<|tool_result|> result <|tool_result_end|>] [spoken reply] [trail idle]
   PAD              PAD                  PAD          EMIT (mask 1)                    INJECTED (mask 0)                        EMIT (mask 1)     PAD
```

**Masking scheme (the core of calibration — see §4.5):**

| segment | text target | mask | rationale |
|---------|-------------|------|-----------|
| lead/idle/listening/gap | `PAD` | **1** | train the model to stay silent / *not* call |
| `<|tool_call|> name <|tool_end|>` | the tokens | **1** | the model must EMIT these |
| `<|tool_result|> … <|tool_result_end|>` | the tokens | **0** | force-fed at runtime, not predicted |
| spoken reply | the tokens | **1** | the model must EMIT these |

**Example categories** (~2,500 examples): `time`, `time_multiturn`, `weather`
(per-city), `weather_local`, `weather_multiturn`, `mixed` (call↔normal-chat for
post-call continuation), and negatives `silence`, `chitchat`, `distractor`
(sentences containing "time"/"weather" that must **not** trigger).

### 3.2 Fine-tuning (notebook `02_finetune_lora.ipynb`)

```
abrarfahim/moshi-tool-patched (7B)         abrarfahim/moshi-tool-audio (dataset)
            │                                          │
            ▼                                          ▼
   get_moshi_lm() to GPU                     codes[17,T] + mask[T]
            │                                          │
   PEFT LoRA (r=16, α=32,                              │
   target out_proj/in_proj) ──────────────────────────┤
            │                                          ▼
            │                       compute_loss: forward_train(codes) →
            │                       cross_entropy on TEXT row where mask=1,
            │                       conditioned on the real user audio (rows 9-16),
            │                       PAD/EPAD downweighted (0.3)
            ▼
   merge_and_unload() → save → abrarfahim/moshi-tool-finetuned
```

- Batch size 1 (variable-length audio), grad-accum 16, LR 2e-4, ~6 epochs.
- Loss is **text-only**: we never train Moshi's audio output (the spoken reply
  audio is generated by the base model at inference; we only need the *text*
  decision + the result consumption).
- **Sanity check** feeds real audio codes from held-out examples and verifies
  the text row argmax = `<|tool_call|>` at the request frame (positives) and
  never elsewhere (negatives).

### 3.3 Live serving (`moshi/server.py`, `tools/orchestrator.py`)

```
 mic ─▶ Mimi.enc ─▶ user audio codes ─┐
                                       ▼
                      ┌──────────  per 80 ms frame  ──────────┐
                      │  VAD gate: user RMS > thr?            │  (tool calls only allowed
                      │     → lm_gen._tool_enabled            │   within ~3 s of user speech)
                      │                                       │
   forced result ───▶ │  LMGen.step(user_audio, text_token)  │
   tokens (INJECT)    │     • temporal transformer           │
                      │     • TEXT decode + tool-threshold:   │  emit 32000 when it is the top
                      │       if 32000 is top non-PAD token   │  NON-padding token & logit≥thr
                      │       and logit≥thr → force 32000     │
                      │     • depformer → 8 Moshi audio toks  │
                      └───────────────┬───────────────────────┘
                                      ▼
        text token ─▶ ToolOrchestrator.observe()        Moshi audio ─▶ Mimi.dec ─▶ speaker
                          state machine (below)
                                      │
                          ToolRegistry.call(name, args)   get_time(), get_weather(city)
```

**Orchestrator state machine** (per session):

```
 NORMAL ──32000──▶ IN_CALL ──32001──▶ EXEC ──result ready──▶ INJECT ──queue drained──▶ NORMAL
   ▲   (cooldown)   (buffer        (dispatch tool      (force 32002 + result          (+cooldown)
   │                 call tokens)   async, validate     + 32003 into text stream,
   └─────────────────────────────── name vs registry)  one token/frame) ──────────────┘
```

- On `<|tool_end|>` the buffered call span is decoded and parsed **by intent**
  (the model phrases it naturally, e.g. "get the time" → `get_time`,
  "weather in London" → `get_weather{city:London}`). Unknown tools are ignored
  silently (no spoken error).
- The result is wrapped in `<|tool_result|> … <|tool_result_end|>` and forced
  into the text stream so Moshi *speaks* the real value.

**Runtime control knobs** (all added during the investigation; CLI/env):
`--tool-threshold`, `--tool-prompt`, `--tool-vad-window`, `--tool-vad-rms`,
`TOOL_COOLDOWN`, `DEBUG_TOOL`.

---

## 4. The research path (what we tried, what we learned)

This is the heart of the log. Each step was driven by an observed failure.

### 4.1 Phase A — training-free keyword interception (baseline)

Watch Moshi's own text stream for the words "time"/"weather"; when they appear,
fire the tool and inject the result.

- **Worked live** — because when Moshi answers a time question it naturally
  *says* "time", so the keyword appears in its monologue.
- **Rejected by design** — it's not model-driven; it piggybacks on words the
  model already speaks. The research goal requires the model to emit an explicit
  call signal. (Also brittle: false triggers like "I had a great **time**",
  city mis-extraction "in **many** places".)

### 4.2 Text-only LoRA fine-tuning of the special tokens

Teach the model to emit `<|tool_call|>` by fine-tuning on **text-only**
sequences (audio rows = zeros): `[user query text][PAD gap][<|tool_call|>…]`.

- **Offline: worked perfectly** (loss 0.50→0.056; sanity 5/5).
- **Live: total failure.** The model never emitted the token; it hallucinated
  times/weather instead.

**The key finding #1 — the text/audio modality gap.** Offline we fed the
question as **text** into the monologue. Live, the question arrives as **audio**
and the monologue does *not* contain it. We had trained `P(call | question
text)` but at inference needed `P(call | question audio)`. Text-only fine-tuning
**cannot** teach a speech model to react to speech. (We also confirmed the
training fed silence in *all* audio rows, so the model learned a pure text
pattern.)

### 4.3 Phase B — audio-grounded fine-tuning

Put the **Mimi-encoded spoken question** into the user-audio rows (9..16) and
train the text row to emit the call — conditioned on *heard* audio.

- Pipeline: edge-tts (15 voices for speaker robustness) → Mimi → `codes[17,T]`.
- Verified the tensor layout from `lm.py`: user audio at rows 9..16 is pure
  conditioning (not in the loss), exactly what we need.
- **Offline: 5/5 positives, 3/3 negatives**, multi-turn re-calls worked.
- **Live: the mechanism worked** — first confirmed real injected time
  ("3:26 PM" matched the server clock, spoken by the model from *our* voice).

This validated the core thesis: **a speech model can be taught to emit a
tool-call token from heard audio.**

### 4.4 Decoding interventions (and why the model still misbehaved)

Live, the model under-fired/over-fired. Instrumenting the text logits
(`DEBUG_TOOL`) revealed **key finding #2 — the padding-dominance problem**:

```
[dbg] <|tool_call|> rank=2 logit=1.85 top5=[3, 0, 32000, 758, 553]
```

`<|tool_call|>` (32000) was consistently **rank 2 — beaten only by `PAD`(3) and
`EPAD`(0)**, the silence tokens. The model *wanted* to call (it was the top
*content* token) but greedy/sampling always picks PAD because the monologue is
silent most frames.

Interventions added:
- **`--tool-threshold`**: emit 32000 when it is the top non-PAD/EPAD token and
  its logit ≥ threshold — i.e. surface the model's own signal past the silence
  tokens. (Not a keyword hack: it reads the model's learned logit, not its
  words.)
- **`--tool-prompt`**: a system prompt telling the model it *has* live tools —
  fixed the contradiction where it emitted a call *and* said "I can't access
  live data" (its language head was unaware of the tools).
- **`--tool-vad-*`**: only allow a call within ~3 s of detected user speech.

### 4.5 Phase 1b — calibration (the deeper fix)

Even with the gates, debug showed 32000 sitting at **rank-2 for the *entire*
conversation** (not just at requests) → persistent over-firing, and the model
even **invented tools** (`get_day`, `get_story`). 

**Key finding #3 — we never trained suppression.** Our loss only covered the
call+reply frames (`mask=1`); every other frame was `mask=0`, so nothing ever
penalized the model for "wanting to call" during normal speech. The signal was
*flat-high*, which no decoding rule can cleanly threshold.

Phase 1b data changes:
1. **Train the in-between frames** (listening / idle / gap) to predict `PAD`
   (`mask=1`) → actively pushes 32000 *down* except at the request.
2. **Mixed examples** (call→chat, chat→call) for post-call continuation.
3. **Heavier negatives** (silence/chitchat/distractor) → ~2,546 examples.
4. **Downweight PAD/EPAD (0.3)** in the loss so the now-dominant suppression
   frames don't collapse the model into permanent silence.

- **Offline after 1b: negatives 3/3 clean, positives 4/5** (only the first turn
  of `weather_multiturn` missed).
- **Live after 1b: weather now fires with real data** ("London: Cloudy, +20°C")
  and negatives are much quieter — but some over-eagerness and tool-invention
  remain (see §6).

---

## 5. Repository map

| Path | Role |
|------|------|
| `notebooks/gen_tool_data.py` | abstract Turn/Example generators + `render_emit` (single source of truth) |
| `notebooks/01_generate_data.ipynb` | TTS → Mimi → `codes[17,T]` → push HF dataset + card |
| `notebooks/02_finetune_lora.ipynb` | LoRA fine-tune (audio-grounded) → push HF model |
| `moshi/moshi/models/loaders.py` | `get_moshi_lm` + `_peek_text_card` (auto-detect 32004 vocab) |
| `moshi/moshi/models/lm.py` | `forward_train`, `LMGen.step`, tool-threshold + `_tool_enabled` gate, `DEBUG_TOOL` |
| `moshi/moshi/tools/protocol.py` | special-token ids, `OrchestratorState`, `ToolIntent` |
| `moshi/moshi/tools/orchestrator.py` | observe/inject state machine, intent parser, cooldown |
| `moshi/moshi/tools/registry.py` | `get_time`, `get_weather` (wttr.in), `@tool` |
| `moshi/moshi/server.py` | per-frame loop, VAD gate, tool prompt, CLI knobs |

**Artifacts (HuggingFace):** `abrarfahim/moshi-tool-patched` (weights+tokenizer),
`abrarfahim/moshi-tool-audio` (dataset), `abrarfahim/moshi-tool-finetuned` (output).

---

## 6. Current state & known limitations

**Working:** model-driven, audio-grounded tool calling on live speech; real
`get_time` and `get_weather` results injected and spoken; no keyword hacks;
offline negatives clean.

**Limitations (all over-eagerness / calibration):**
1. **Tool invention** — emits calls for non-existent tools (`get_day`,
   `getyear`). Harmless to function (orchestrator ignores unknown tools) but the
   model *speaks* a hallucinated answer.
2. **Pre-call hallucination** — occasionally voices a made-up value before the
   real result lands.
3. **Stale values** — Moshi has no live clock; it repeats the last injected
   value until it re-calls (only on a new request after the cooldown).
4. **Residual reliance on runtime gates** (threshold + VAD) rather than a fully
   self-calibrated signal.

---

## 7. Evaluation methodology

- **Offline (per checkpoint):** feed held-out `codes[17,T]`; report
  positives-fire rate (argmax = 32000 at the request frame) and negatives-clean
  rate (32000 never appears) per category.
- **Logit instrumentation (`DEBUG_TOOL`):** rank/logit of 32000 per frame —
  distinguishes "model didn't learn" (low rank) from "padding dominates"
  (rank-2) from "well-calibrated" (low at idle, rank-1 at request).
- **Live:** scripted spoken prompts (time / weather / distractors / silence);
  measure trigger precision & recall, argument correctness, and whether the
  spoken value matches the injected (real) value.

Proposed quantitative metrics for the write-up: trigger **precision/recall**,
**arg-format validity**, **false-call rate during normal speech**, **call
latency**, and **idle-frame 32000 rank distribution** (calibration proxy).

---

## 8. Further work

### Phase 1c — finish calibration (next)
- **Anti-invention negatives:** users ask about day/date/year/jokes → Moshi
  answers normally, **no call** → teaches the tool set is exactly {time, weather}.
- **Re-call examples:** repeated "what time is it now?" → fresh call (kills
  parroting/staleness).
- **System prompt in training:** prepend the persona/tool prompt so the training
  distribution matches live serving.
- Re-sweep `--tool-threshold` / VAD; goal: drop the gates entirely once idle-rank
  is low.

### Phase 2 — real datasets (planned)
Mine a public function-calling corpus (e.g. `glaiveai/glaive-function-calling-v2`)
for (a) realistic time/weather **phrasings** → positives, and (b) a large pool of
non-time/weather utterances → **realistic negatives**; run all through the same
TTS→Mimi pipeline; keep the 2-tool scope and special-token format.

### Phase 3 — capability & rigor
- **Spoken-response audio training:** currently Moshi's audio rows are silence
  in training (text-only loss). Train on rendered Moshi-response audio so the
  answer is conditioned correctly (reduces pre-call hallucination).
- **More tools** with real arguments (timers, units, search) — requires arg-rich
  training data.
- **Barge-in / cancellation** during EXEC (full-duplex interruption).
- **Latency budget** measurement and async-dispatch hardening.
- **Ablations** for the paper: text-only vs audio-grounded; with/without
  suppression frames; PAD-downweight sweep; voice-count vs speaker
  generalization; threshold/VAD necessity vs calibration.

---

## 9. Key contributions (for the write-up)

1. A **training-free → fine-tuned** progression showing that text-stream tool
   calling in a full-duplex speech LLM **requires audio-grounded supervision**
   (text-only transfers offline but fails live — the modality gap).
2. A concrete **audio-grounded data recipe**: place TTS+Mimi-encoded user speech
   in the conditioning rows; train the text row with a masking scheme that
   includes **explicit suppression frames** to calibrate the call signal.
3. A diagnosis of the **padding-dominance** failure mode (the learned call token
   is rank-2 behind silence tokens) and two complementary fixes: a decode-time
   **top-non-padding threshold** and training-time **suppression-frame loss**.
4. A minimal, agent-framework-free **orchestrator** that reads the model's own
   emitted call and injects real results back into the inner monologue.
