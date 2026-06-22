# Step-by-Step Walkthrough: How We Built Audio-Grounded Tool Calling

A pedagogical companion to `RESEARCH.md`. This walks through the whole system in
the order it actually runs, with diagrams and a concrete frame-by-frame trace of
a single tool call. Read top to bottom to understand *how it works*; use §8 to
*reproduce it*.

---

## 0. The mental model (read this first)

Moshi is a full-duplex speech model. Every **80 ms** it emits, for one "frame":

```
            ┌── 1 TEXT token  (the "inner monologue" — leads the audio)
 1 frame ──▶├── 8 Moshi audio tokens (what Moshi says)
            └──   (it also consumes 8 user-audio tokens = what it hears)
```

Two facts are the entire basis of the project:

1. The **text token comes first and steers the audio** → if we can get a special
   `<|tool_call|>` token to appear in that text stream, we have a tool trigger;
   and if we *force* result text into that stream, Moshi will *speak* it.
2. The **text stream is Moshi's output, not the user's input.** The user's words
   arrive as a separate **audio** stream. ⟹ to make Moshi react to a *spoken*
   request, it must learn `request-audio → tool-token`, which can only be taught
   with **audio-grounded** training (this is the single most important lesson).

So the plan is: (1) add tool tokens to the vocab, (2) teach the model to emit
them *from heard audio* via fine-tuning, (3) at serving time catch the emitted
token, run a local function, and inject the result back into the monologue.

---

## 1. The data structure everything hinges on: `codes[17, T]`

The LM consumes a tensor of **17 rows × T frames**:

```
                    frame →   0    1    2   ...                    T-1
       ┌─────────────────────────────────────────────────────────────┐
row 0  │ TEXT monologue      PAD  PAD  ...  <|tool_call|> get_time ... │  ← we drive / read this
       ├─────────────────────────────────────────────────────────────┤
rows   │ Moshi AUDIO (8)     ...silence/own speech...                  │  ← Moshi speaks
1..8   │                                                               │
       ├─────────────────────────────────────────────────────────────┤
rows   │ USER AUDIO (8)      ── the spoken question (Mimi codes) ──    │  ← what Moshi hears
9..16  │                                                               │  ← CONDITIONING ONLY
       └─────────────────────────────────────────────────────────────┘
```

- 1 frame = `1920 samples` at 24 kHz (`24000 / 12.5 Hz`).
- `forward_train` computes loss on **row 0 (text)** and **rows 1–8 (Moshi audio)**
  only. **Rows 9–16 (user audio) are pure conditioning** — never predicted. That
  property is *why* we can drop the user's spoken question into rows 9–16 and
  train row 0 to react to it.

**Special tokens** (we grew the text vocab 32000 → 32004):

| id | token | who emits it |
|----|-------|--------------|
| 32000 | `<|tool_call|>` | the model |
| 32001 | `<|tool_end|>` | the model |
| 32002 | `<|tool_result|>` | the orchestrator (injected) |
| 32003 | `<|tool_result_end|>` | the orchestrator (injected) |

---

## 2. STEP 1 — Patch the model (add the tool tokens)

The base PersonaPlex text head outputs 32000 classes. We added 4 rows to the
text embedding and the output projection (`text_linear`) and 4 entries to the
SentencePiece tokenizer, producing **`text_card = 32004`**.

- Loader auto-detects this: `_peek_text_card()` reads `text_linear.weight`'s
  shape from the checkpoint so the model is built with the right vocab size.
- Artifact: **`abrarfahim/moshi-tool-patched`** (weights + tokenizer).

```
PersonaPlex-7B  ──add 4 text rows (embed + text_linear) + 4 tokenizer pieces──▶  moshi-tool-patched
```

---

## 3. STEP 2 — Generate audio-grounded data (`01_generate_data.ipynb`)

The pipeline turns an abstract conversation spec into `codes[17,T]`.

```
 gen_tool_data.examples()                  edge-tts (15 voices)         Mimi.encode (GPU)
 ──────────────────────────               ────────────────────         ─────────────────
 Example(turns=[Turn(                      "what time is it"  ──wav──▶  8 user-audio codebooks
   query="what time is it",                (24 kHz, disk-cached)              │
   tool=("get_time", None),                                                   │
   result="11:55 PM ...",        render_emit() → text tokens:                 ▼
   reply="It's 11:55 pm")])      <|tool_call|> get_time <|tool_end|>   assemble codes[17,T]:
        │                        <|tool_result|> 11:55 PM <|tool_result_end|>   row 0  = text
        │                        It's 11:55 pm                                  rows 1-8 = silence
        └──────────────────────────────────────────────────────────────────▶  rows 9-16 = user audio
                                                                                      │
                                                                                      ▼
                                                              HF dataset abrarfahim/moshi-tool-audio
```

### 3a. The per-example timeline (frame alignment)

Each segment's waveform is padded to a whole number of frames so audio and text
line up 1:1:

```
 frames:  [ lead idle ][  user speaks the question  ][gap][ CALL ][  RESULT (injected ctx) ][ REPLY ][ trail idle ]
 text:      PAD ...         PAD ... (listening)       PAD  32000 get_time 32001  32002 "11:55 PM" 32003  "It's 11:55 pm"  PAD ...
 user aud:  silence         <the spoken question>     sil   silence                silence                 silence        silence
 mask:       1                    1                     1    1   1      1    0     0  0        0       1  1  1            1
            └── trained to PAD (suppress) ──┘         └─ EMIT ─┘     └── injected, mask 0 ──┘ └─ EMIT ─┘  └ suppress ┘
```

### 3b. The masking scheme (the calibration core)

| segment | text target | `mask` | why |
|---------|-------------|--------|-----|
| lead / listening / gap / trail | `PAD` | **1** | **teach it to stay silent & NOT call** when not asked |
| `<|tool_call|> name <|tool_end|>` | the tokens | **1** | the model must **emit** these |
| `<|tool_result|> … <|tool_result_end|>` | the tokens | **0** | force-fed at runtime, not predicted |
| spoken reply | the tokens | **1** | the model must **emit** these |

The PAD-everywhere-except-the-call contrast is what makes the call token a sharp
spike instead of a constant background signal (see RESEARCH.md §4.5).

Output dataset columns: `type, query, reply, voice, audio (playable), codes[17×T], mask[T]`.

---

## 4. STEP 3 — Fine-tune with LoRA (`02_finetune_lora.ipynb`)

```
 moshi-tool-patched (7B)                 moshi-tool-audio (dataset)
        │                                        │
   load to GPU                          codes[17,T] + mask[T]   (batch 1)
        │                                        │
   PEFT LoRA  r=16 α=32                          ▼
   target: out_proj, in_proj   ──▶  forward_train(codes) → text_logits[B,T,32004]
   (temporal-transformer attn)        loss = cross_entropy( text_logits[mask==1],
        │                                                   codes[0][mask==1],
        │                                                   weight: PAD/EPAD ×0.3 )
        ▼
   merge_and_unload → save → abrarfahim/moshi-tool-finetuned
```

Key choices and *why*:
- **LoRA, rank 16** on the temporal transformer's attention projections — cheap,
  enough capacity to add the trigger behavior without disturbing the base.
- **Text-only loss.** We never train Moshi's audio output; we only need the text
  *decision* + the ability to *consume* an injected result. The spoken reply
  audio is produced by the base model's depformer at inference.
- **PAD/EPAD down-weight (0.3).** After Phase-1b most frames are "predict PAD"
  (suppression); down-weighting prevents the model from collapsing into silence.
- **Sanity gate.** Feed held-out `codes[17,T]`; assert `argmax(text_logits)`
  = 32000 at the request frame (positives fire) and never elsewhere (negatives
  clean).

---

## 5. STEP 4 — Serve & orchestrate (`server.py`, `tools/`)

At inference there is no teacher forcing — the model *generates*. Three problems
appear that don't exist offline, each with a fix:

| problem (live only) | fix | where |
|---------------------|-----|-------|
| text is **sampled** (temp 0.7, top-k 25), and the call token is rank-2 behind PAD | **tool-threshold**: emit 32000 when it's the top *non-PAD/EPAD* token & logit ≥ θ | `lm.py` |
| model **over-fires** when idle/eager | **VAD gate**: only allow a call within ~3 s of user speech | `server.py` |
| raw result gets **read aloud** | **audio mute**: force Moshi audio to Mimi-silence during the injected result | `orchestrator.py` |
| language head **refuses** ("can't access live data") | **tool system-prompt** declaring it has live tools | `server.py` |

### 5a. The per-frame serving loop

```
 mic PCM ─▶ RMS VAD ─▶ set lm_gen._tool_enabled (user spoke recently?)
        ─▶ Mimi.encode ─▶ user-audio codes [1,8,1]
                              │
        ┌──── orchestrator.step(user_codes) ────────────────────────────┐
        │  forced = next injected token (or None)                        │
        │  if injecting: also force Moshi audio = silence (mute)         │
        │  tokens = lm_gen.step(user_codes, moshi_tokens, text_token)    │
        │     └─ temporal transformer → text_logits                      │
        │        └─ tool-threshold rule may FORCE 32000                  │
        │     └─ depformer → 8 Moshi audio tokens                        │
        │  observe(text_token)  → state machine                         │
        └────────────────────────────────────────────────────────────────┘
                              │
        text<32000 ─▶ websocket (shown)     Moshi audio ─▶ Mimi.decode ─▶ speaker
```

### 5b. The orchestrator state machine

```
 NORMAL ──sees 32000──▶ IN_CALL ──sees 32001──▶ EXEC ──result ready──▶ INJECT ──drained──▶ NORMAL
   ▲  (cooldown gate)   (buffer the      (decode call span by intent,   (force 32002 + result    (+cooldown,
   │                     call tokens)     validate vs registry,          + 32003, one/frame,       reset)
   └───────────────────────────────────── dispatch tool async) ───────── audio muted) ─────────────┘
```

- **Intent parsing**: the model phrases the call naturally ("get the time",
  "weather in London"), so we map by intent (`time→get_time`,
  `weather→get_weather` + city) rather than requiring the exact function name.
- **Registry**: `get_time()` and `get_weather(city)` (wttr.in). Unknown/garbled
  calls are ignored silently (no spoken error).

---

## 6. A concrete frame-by-frame trace ("what time is it?")

This is the whole loop for one successful call. Frame numbers are illustrative.

```
frame  user audio          text row (sampled/forced)     orchestrator     spoken?
─────  ─────────────────   ──────────────────────────    ─────────────    ───────
0-4    silence             PAD                           NORMAL            (silent)
5-30   "what time is it"   PAD  (Moshi listens)          NORMAL            (silent)
                           VAD: user speaking → _tool_enabled = True
33     silence (gap)       32000  ← threshold surfaces   NORMAL→IN_CALL    (not voiced)
                           it (top non-PAD token, ≥θ)
34-35  silence             "get" "_time" (sampled)       IN_CALL: buffer   (not voiced)
36     silence             32001                         IN_CALL→EXEC      (not voiced)
                           → decode buffer "get_time" → registry.get_time()
                           → "11:55 PM on Sunday, June 22, 2026"
37     silence             32002  (FORCED)               EXEC→INJECT       muted
38-46  silence             "11:55 PM ..." (FORCED)       INJECT            MUTED (audio=silence)
47     silence             32003  (FORCED)               INJECT            muted
48+    silence             "It's 11:55 pm" (sampled)     INJECT→NORMAL     SPOKEN ✓
                                                         (+cooldown)
```

Net effect: the user hears **only** "It's 11:55 pm" — the model decided to call
from your *voice*, the real clock value was injected, and the raw machine string
was muted.

---

## 7. The experimental path (how we got here, condensed)

| attempt | result | lesson |
|---------|--------|--------|
| Phase A: keyword interception | works live, but a hack | not model-driven; rejected |
| text-only fine-tune | 5/5 offline, **fails live** | **modality gap**: text-trained ≠ audio-driven |
| audio-grounded fine-tune | real time injected live ✓ | audio grounding is the unlock |
| add decode threshold + VAD | fires, but over/under | **padding dominance**: call token stuck rank-2 |
| Phase 1b: train suppression frames | negatives clean, weather fires | must train "don't call" frames too |
| audio-mute on inject | raw result no longer spoken | matches training (silent audio rows) |
| **open**: barge-in, invention, calibration | partial | **data-coverage** problems → Phase 1c |

---

## 8. Reproduce from scratch (commands)

```bash
# 0. (one-time) patch model → abrarfahim/moshi-tool-patched   [done]

# 1. data — Modal GPU notebook
#    run notebooks/01_generate_data.ipynb top-to-bottom
#    → pushes abrarfahim/moshi-tool-audio   (inspect + play audio in the cell)

# 2. train — Modal GPU notebook (24 GB+)
#    run notebooks/02_finetune_lora.ipynb top-to-bottom
#    → sanity check (positives fire / negatives clean)
#    → pushes abrarfahim/moshi-tool-finetuned

# 3. serve — GPU box
rm -rf ~/.cache/huggingface/hub/models--abrarfahim--moshi-tool-finetuned
python -m moshi.server \
  --moshi-hf-repo abrarfahim/moshi-tool-finetuned \
  --port 5001 --ssl ~/ssl \
  --tool-threshold 1.3            # + optional: --tool-vad-window/--tool-vad-rms, TOOL_COOLDOWN, DEBUG_TOOL
```

Debugging aid: `DEBUG_TOOL=1` logs the rank/logit of `<|tool_call|>` per frame —
the single best signal for "did the model learn / is it calibrated."

---

## 9. Where to read more

- `RESEARCH.md` — the research log, key findings, evaluation methodology, and the
  phased future-work plan (Phase 1c barge-in, Phase 2 real datasets, Phase 3
  spoken-response training + ablations).
- Code: `notebooks/gen_tool_data.py`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`,
  `moshi/moshi/tools/{protocol,orchestrator,registry}.py`, `moshi/moshi/models/lm.py`,
  `moshi/moshi/server.py`.
