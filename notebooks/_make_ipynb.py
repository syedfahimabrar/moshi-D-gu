"""Run once to generate the three .ipynb files. Delete afterwards."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def lines(src: str) -> list[str]:
    parts = src.strip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts else [])


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines(src)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


def nb(cells: list) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": cells,
    }


def save(notebook: dict, name: str):
    p = HERE / name
    p.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 00 — Push patched weights to HuggingFace (run on server)
# ═══════════════════════════════════════════════════════════════════════════════
save(nb([

md("""\
# 00 — Push Patched Weights to HuggingFace

**Run this on your server once** to upload patched weights to a private HF repo.
Modal notebooks will pull from there.

Prerequisites
- `pip install huggingface_hub`
- `huggingface-cli login`  OR paste your token into `HF_TOKEN` below\
"""),

code("!pip install huggingface_hub -q"),

code("""\
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── Fill these in ─────────────────────────────────────────────────────────────
HF_REPO     = "YOUR_HF_USERNAME/moshi-tool-patched"  # created as private
HF_TOKEN    = None   # paste token string, or None to use huggingface-cli login
PATCHED_DIR = Path("../weights/patched")
# ──────────────────────────────────────────────────────────────────────────────

assert PATCHED_DIR.exists(), f"Not found: {PATCHED_DIR} — run run_server.sh first"\
"""),

code("""\
api = HfApi(token=HF_TOKEN)
create_repo(HF_REPO, private=True, exist_ok=True, token=HF_TOKEN)
print(f"Repo ready: https://huggingface.co/{HF_REPO}")

for f in sorted(PATCHED_DIR.iterdir()):
    if not f.is_file():
        continue
    print(f"Uploading {f.name}  ({f.stat().st_size/1e9:.2f} GB) ...")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=HF_REPO, token=HF_TOKEN)
    print(f"  ✓ {f.name}")

print(f"\\nDone. Paste this into notebook 02:\\nHF_PATCHED_REPO = '{HF_REPO}'")\
"""),

]), "00_push_patched_to_hf.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 01 — Generate training data  (CPU, run in Modal or locally)
# ═══════════════════════════════════════════════════════════════════════════════
save(nb([

md("""\
# 01 — Generate Tool-Calling Training Data

**CPU only. Run in Modal notebook before notebook 02.**

Creates ~400 synthetic training sequences that teach Moshi:
1. Emit `<|tool_call|>get_time<|tool_end|>` when context implies a time query
2. Emit `<|tool_call|>get_weather CITY<|tool_end|>` for weather
3. Respond naturally after a silent `<|tool_result|>…<|tool_result_end|>` injection

Output saved to `data/tool_calls.jsonl`.\
"""),

code("""\
# Install / upgrade if needed (comment out after first run)
!pip install sentencepiece huggingface_hub -q\
"""),

code("""\
import json, random, sys
from pathlib import Path

# Clone repo if not present (Modal environment)
REPO = Path("/repo")
if not REPO.exists():
    import subprocess
    subprocess.run(["git", "clone",
        "https://github.com/syedfahimabrar/moshi-D-gu.git",
        str(REPO)], check=True)

sys.path.insert(0, str(REPO / "moshi"))

import sentencepiece as spm

TOKENIZER_PATH = REPO / "weights/patched/tokenizer_spm_32k_3.model"
if not TOKENIZER_PATH.exists():
    # Pull from HF if local copy missing
    from huggingface_hub import hf_hub_download
    HF_PATCHED_REPO = "YOUR_HF_USERNAME/moshi-tool-patched"
    HF_TOKEN = None  # set if needed
    TOKENIZER_PATH = Path(hf_hub_download(
        repo_id=HF_PATCHED_REPO, filename="tokenizer_spm_32k_3.model",
        local_dir="/tmp/patched", token=HF_TOKEN))

tok = spm.SentencePieceProcessor()
tok.Load(str(TOKENIZER_PATH))
print(f"Tokenizer: {tok.get_piece_size()} tokens")\
"""),

code("""\
PAD_ID             = 3
TOOL_CALL_ID       = 32000
TOOL_END_ID        = 32001
TOOL_RESULT_ID     = 32002
TOOL_RESULT_END_ID = 32003

assert tok.id_to_piece(TOOL_CALL_ID) == "<|tool_call|>", "Tokenizer not patched!"
print("Special tokens:", [tok.id_to_piece(i) for i in [32000,32001,32002,32003]])\
"""),

code("""\
def encode(text):
    return [i for i in tok.encode(text) if i < 32000]

_DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
_MONTHS = ["January","February","March","April","May","June",
           "July","August","September","October","November","December"]

_TIME_TRIGGERS = [
    "what time is it", "tell me the time", "what's the current time",
    "do you know what time it is", "can you check the time",
    "what time do you have", "I need to know the time",
    "checking the time", "let me check the time",
]

_TIME_RESPONSES = [
    "It's {t} right now.", "The time is {t}.", "Right now it's {t}.",
    "It's currently {t}.", "The current time is {t}.",
]

def _random_time():
    h, m = random.randint(1,12), random.randint(0,59)
    ap = random.choice(["AM","PM"])
    result = f"Live time: {h}:{m:02d} {ap} on {random.choice(_DAYS)}, {random.choice(_MONTHS)} {random.randint(1,28)}, 2026"
    spoken = (f"{h} o'clock {ap.lower()}" if m==0 else
              f"quarter past {h} {ap.lower()}" if m==15 else
              f"half past {h} {ap.lower()}" if m==30 else
              f"{h}:{m:02d} {ap.lower()}")
    return result, spoken

def make_time_example():
    result_str, spoken = _random_time()
    trigger_ids = encode(random.choice(_TIME_TRIGGERS))
    gap = [PAD_ID] * random.randint(5, 20)
    tool_seq = ([TOOL_CALL_ID] + encode("get_time") + [TOOL_END_ID] +
                [TOOL_RESULT_ID] + encode(result_str) + [TOOL_RESULT_END_ID] +
                encode(random.choice(_TIME_RESPONSES).format(t=spoken)))
    tokens = trigger_ids + gap + tool_seq
    mask   = [0]*(len(trigger_ids)+len(gap)) + [1]*len(tool_seq)
    return {"tokens": tokens, "mask": mask, "type": "time"}

print("Time example:", tok.decode([t for t in make_time_example()["tokens"] if t < 32000]))\
"""),

code("""\
_CITIES = [
    "London","New York","Tokyo","Paris","Sydney","Dubai","Mumbai","Toronto",
    "Berlin","Singapore","Los Angeles","Seoul","Bangkok","Istanbul","Cairo",
    "Amsterdam","Madrid","Rome","Hong Kong","Kuala Lumpur","Dhaka","Karachi",
]

_CONDITIONS = [
    ("sunny",         lambda c: f"Live weather: {c}: ☀️ {random.randint(18,35)}°C"),
    ("cloudy",        lambda c: f"Live weather: {c}: ☁️ {random.randint(10,22)}°C"),
    ("partly cloudy", lambda c: f"Live weather: {c}: ⛅ {random.randint(12,28)}°C"),
    ("rainy",         lambda c: f"Live weather: {c}: 🌧️ {random.randint(8,18)}°C"),
    ("snowy",         lambda c: f"Live weather: {c}: ❄️ {random.randint(-8,2)}°C"),
    ("windy",         lambda c: f"Live weather: {c}: 💨 {random.randint(10,20)}°C"),
]

_WEATHER_RESPONSES = {
    "sunny":         "It's sunny and warm in {city}, {temp} right now.",
    "cloudy":        "Overcast in {city} today, about {temp}.",
    "partly cloudy": "Partly cloudy in {city}, {temp}.",
    "rainy":         "It's raining in {city}, {temp} out there.",
    "snowy":         "Snowing in {city} right now, {temp}.",
    "windy":         "Pretty windy in {city}, {temp}.",
}

def make_weather_example(city=None):
    city = city or random.choice(_CITIES)
    cond, result_fn = random.choice(_CONDITIONS)
    result_str = result_fn(city)
    temp = result_str.split("°C")[0].split()[-1] + "°C"
    triggers = [f"what's the weather in {city}", f"weather in {city}",
                f"how's the weather in {city}", f"check weather in {city}"]
    trigger_ids = encode(random.choice(triggers))
    gap = [PAD_ID] * random.randint(5, 20)
    tool_seq = ([TOOL_CALL_ID] + encode(f"get_weather {city}") + [TOOL_END_ID] +
                [TOOL_RESULT_ID] + encode(result_str) + [TOOL_RESULT_END_ID] +
                encode(_WEATHER_RESPONSES[cond].format(city=city, temp=temp)))
    tokens = trigger_ids + gap + tool_seq
    mask   = [0]*(len(trigger_ids)+len(gap)) + [1]*len(tool_seq)
    return {"tokens": tokens, "mask": mask, "type": "weather"}

print("Weather example:", tok.decode([t for t in make_weather_example("London")["tokens"] if t < 32000]))\
"""),

code("""\
_NULL_PHRASES = [
    "hey how are you doing", "tell me a fun fact", "what do you think about music",
    "how does the internet work", "what's your favorite book",
    "what are some good movies", "tell me about space exploration",
    "what is machine learning", "describe a beautiful sunset",
]

def make_null_example():
    phrase_ids   = encode(random.choice(_NULL_PHRASES))
    gap          = [PAD_ID] * random.randint(5, 20)
    response_ids = encode(random.choice(["I'd be happy to help.", "Great question.",
                                         "That's interesting.", "Let me think about that."]))
    tokens = phrase_ids + gap + response_ids
    mask   = [0]*(len(phrase_ids)+len(gap)) + [1]*len(response_ids)
    return {"tokens": tokens, "mask": mask, "type": "null"}\
"""),

code("""\
random.seed(42)
dataset = []

for _ in range(150):
    dataset.append(make_time_example())

for city in _CITIES:
    dataset.append(make_weather_example(city))
    dataset.append(make_weather_example(city))
for _ in range(140):
    dataset.append(make_weather_example())

for _ in range(50):
    dataset.append(make_null_example())

random.shuffle(dataset)

counts = {}
for ex in dataset:
    counts[ex["type"]] = counts.get(ex["type"], 0) + 1

print(f"Total: {len(dataset)} examples")
print("Breakdown:", counts)
print(f"Avg length: {sum(len(e['tokens']) for e in dataset)/len(dataset):.0f} tokens")\
"""),

code("""\
OUT = Path("data/tool_calls.jsonl")
OUT.parent.mkdir(exist_ok=True)

with OUT.open("w") as f:
    for ex in dataset:
        f.write(json.dumps(ex) + "\\n")

print(f"Saved {len(dataset)} examples → {OUT}")\
"""),

code("""\
# Preview one of each type
for t in ["time", "weather", "null"]:
    ex = next(e for e in dataset if e["type"] == t)
    readable = []
    for tid in ex["tokens"]:
        if   tid == TOOL_CALL_ID:       readable.append("<CALL>")
        elif tid == TOOL_END_ID:        readable.append("<END>")
        elif tid == TOOL_RESULT_ID:     readable.append("<RESULT>")
        elif tid == TOOL_RESULT_END_ID: readable.append("<RESULT_END>")
        elif tid == PAD_ID:             readable.append("[P]")
        else: readable.append(tok.id_to_piece(tid).replace("▁", " "))
    print(f"[{t}] {''.join(readable[:80])}...")
    print()\
"""),

]), "01_generate_data.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 02 — LoRA fine-tune + push to HF  (GPU A100 on Modal)
# ═══════════════════════════════════════════════════════════════════════════════
save(nb([

md("""\
# 02 — LoRA Fine-tuning + Push to HuggingFace

**Run on Modal GPU notebook (A100 recommended).**

Steps:
1. Pull patched Moshi weights from your private HF repo
2. Apply LoRA to temporal transformer attention layers (PEFT)
3. Fine-tune on `data/tool_calls.jsonl` — text stream only
4. Merge LoRA and push fine-tuned model to HuggingFace

VRAM guide:
| GPU | VRAM | LoRA rank | Batch |
|-----|------|-----------|-------|
| A100 40/80 GB | 40-80 GB | 16 | 4-8 |
| RTX 3090/4090 | 24 GB | 8 | 2 |
| V100 16 GB | 16 GB | 4 | 1 |\
"""),

code("""\
import subprocess
r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader"], capture_output=True, text=True)
print(r.stdout.strip())

import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")\
"""),

code("""\
!pip install peft==0.11.0 safetensors sentencepiece huggingface_hub -q\
"""),

code("""\
import json, os, sys, shutil
from pathlib import Path

# Clone repo
REPO = Path("/repo")
if not REPO.exists():
    import subprocess
    subprocess.run(["git","clone",
        "https://github.com/syedfahimabrar/moshi-D-gu.git",
        str(REPO)], check=True)

sys.path.insert(0, str(REPO / "moshi"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from moshi.models.loaders import get_moshi_lm

# ── Config — fill these in ────────────────────────────────────────────────────
HF_PATCHED_REPO = "YOUR_HF_USERNAME/moshi-tool-patched"    # from notebook 00
HF_OUTPUT_REPO  = "YOUR_HF_USERNAME/moshi-tool-finetuned"  # where result goes
HF_TOKEN        = os.environ.get("HF_TOKEN", None)          # Modal secret or paste here

LORA_RANK    = 8      # reduce to 4 if < 20 GB VRAM
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
BATCH_SIZE   = 2      # reduce to 1 if OOM
GRAD_ACCUM   = 4      # effective batch = BATCH_SIZE × GRAD_ACCUM
LR           = 2e-4
EPOCHS       = 3
MAX_SEQ_LEN  = 512
GRAD_CKPT    = True

WEIGHTS_DIR = Path("/tmp/moshi_weights")
OUT_DIR     = Path("/tmp/moshi_finetuned")
DATA_PATH   = Path("data/tool_calls.jsonl")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16

assert DATA_PATH.exists(), f"Run notebook 01 first to generate {DATA_PATH}"
print(f"Device: {DEVICE} | rank: {LORA_RANK} | batch: {BATCH_SIZE} | epochs: {EPOCHS}")\
"""),

code("""\
from huggingface_hub import hf_hub_download

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

for filename in ["model.safetensors", "tokenizer_spm_32k_3.model"]:
    dest = WEIGHTS_DIR / filename
    if dest.exists():
        print(f"Already present: {filename}")
        continue
    print(f"Downloading {filename} ...")
    hf_hub_download(repo_id=HF_PATCHED_REPO, filename=filename,
                    local_dir=str(WEIGHTS_DIR), token=HF_TOKEN)
    print(f"  ✓ {dest}")

print("Weights ready.")\
"""),

code("""\
tok = spm.SentencePieceProcessor()
tok.Load(str(WEIGHTS_DIR / "tokenizer_spm_32k_3.model"))
print(f"Tokenizer: {tok.get_piece_size()} tokens")

PAD_ID             = 3
TOOL_CALL_ID       = 32000
TOOL_END_ID        = 32001
TOOL_RESULT_ID     = 32002
TOOL_RESULT_END_ID = 32003\
"""),

code("""\
print("Loading model (1-2 min) ...")
lm_model = get_moshi_lm(WEIGHTS_DIR / "model.safetensors", device=DEVICE, dtype=DTYPE)
lm_model.train()
total = sum(p.numel() for p in lm_model.parameters())
print(f"Loaded: {total/1e9:.2f}B params  text_card={lm_model.text_card}")\
"""),

code("""\
# Inspect Linear layers to confirm TARGET_MODULES names
linear_layers = [(n, m) for n, m in lm_model.named_modules() if isinstance(m, nn.Linear)]
seen = set()
for name, mod in linear_layers:
    pattern = ".".join(name.split(".")[-2:])
    if pattern not in seen:
        seen.add(pattern)
        print(f"  {name:<60} {list(mod.weight.shape)}")

print(f"\\nTotal Linear layers: {len(linear_layers)}")
print("\\nUpdate TARGET_MODULES below to match attention projection names (out_proj / in_proj etc)")\
"""),

code("""\
from peft import LoraConfig, get_peft_model

# Adjust TARGET_MODULES based on the layer names printed above
TARGET_MODULES = ["out_proj", "in_proj"]

for p in lm_model.parameters():
    p.requires_grad = False

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias="none",
)
lm_model = get_peft_model(lm_model, lora_config)
lm_model.print_trainable_parameters()

if GRAD_CKPT:
    lm_model.enable_input_require_grads()
    lm_model.gradient_checkpointing_enable()
    print("Gradient checkpointing ON")\
"""),

code("""\
class ToolCallDataset(Dataset):
    def __init__(self, path, max_len=MAX_SEQ_LEN):
        self.examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                t = torch.tensor(ex["tokens"][:max_len], dtype=torch.long)
                m = torch.tensor(ex["mask"][:max_len],   dtype=torch.long)
                self.examples.append({"tokens": t, "mask": m})
        print(f"Dataset: {len(self.examples)} examples")

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

def collate(batch):
    L = max(b["tokens"].shape[0] for b in batch)
    tokens = torch.zeros(len(batch), L, dtype=torch.long)
    masks  = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        tokens[i,:n] = b["tokens"]
        masks[i,:n]  = b["mask"]
    return {"tokens": tokens, "masks": masks}

dataset    = ToolCallDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate, num_workers=0)
print(f"Batches/epoch: {len(dataloader)}")\
"""),

code("""\
def compute_loss(model, tokens, masks):
    B, T = tokens.shape
    inp  = tokens[:, :-1].unsqueeze(1)          # [B,1,T-1]
    tgt  = tokens[:, 1:]                         # [B,T-1]
    msk  = masks[:, 1:].bool()                   # [B,T-1]
    audio_in = torch.zeros(B, model.n_q, T-1, dtype=torch.long, device=tokens.device)
    out = model(audio_in, inp)
    logits = out[0]                              # [B, vocab, T-1]
    B2, C, Tm = logits.shape
    flat_l = logits.permute(0,2,1).reshape(-1, C)
    flat_t = tgt.reshape(-1)
    flat_m = msk.reshape(-1)
    if flat_m.sum() == 0:
        return torch.tensor(0., requires_grad=True, device=tokens.device)
    return nn.functional.cross_entropy(flat_l[flat_m], flat_t[flat_m])\
"""),

code("""\
optimizer = torch.optim.AdamW(
    [p for p in lm_model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(dataloader))

lm_model.train()
global_step = 0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    epoch_loss, n = 0., 0
    for step, batch in enumerate(dataloader):
        tokens = batch["tokens"].to(DEVICE)
        masks  = batch["masks"].to(DEVICE)
        loss   = compute_loss(lm_model, tokens, masks) / GRAD_ACCUM
        loss.backward()
        epoch_loss += loss.item() * GRAD_ACCUM
        n += 1
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in lm_model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}  step {global_step}"
                      f"  loss {epoch_loss/n:.4f}"
                      f"  lr {scheduler.get_last_lr()[0]:.2e}")
    print(f"=== Epoch {epoch+1} done  loss={epoch_loss/n:.4f} ===")

print("Training complete.")\
"""),

code("""\
from safetensors.torch import save_file

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Merging LoRA weights ...")
merged = lm_model.merge_and_unload()
merged.eval()

out_weight = OUT_DIR / "model.safetensors"
save_file({k: v.contiguous() for k, v in merged.state_dict().items()}, str(out_weight))
shutil.copy2(WEIGHTS_DIR / "tokenizer_spm_32k_3.model",
             OUT_DIR / "tokenizer_spm_32k_3.model")

print(f"Saved → {OUT_DIR}")\
"""),

code("""\
# Sanity check: does the model emit <|tool_call|> on a time trigger?
from moshi.models.loaders import get_moshi_lm as load_lm

check = load_lm(OUT_DIR / "model.safetensors", device=DEVICE, dtype=DTYPE)
check.eval()

ids = torch.tensor([[PAD_ID]*10 + tok.encode("what time is it")],
                   dtype=torch.long, device=DEVICE)
generated = []
with torch.no_grad():
    for _ in range(20):
        audio_in = torch.zeros(1, check.n_q, ids.shape[1], dtype=torch.long, device=DEVICE)
        out      = check(audio_in, ids.unsqueeze(1))
        nxt      = out[0][:,:,-1].argmax(-1).item()
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if nxt == TOOL_END_ID:
            break

readable = []
for t in generated:
    if   t == TOOL_CALL_ID:       readable.append("<|tool_call|>")
    elif t == TOOL_END_ID:        readable.append("<|tool_end|>")
    elif t == TOOL_RESULT_ID:     readable.append("<|tool_result|>")
    elif t == TOOL_RESULT_END_ID: readable.append("<|tool_result_end|>")
    elif t < tok.get_piece_size(): readable.append(tok.id_to_piece(t))

print("Generated:", "".join(readable))
print()
if TOOL_CALL_ID in generated and TOOL_END_ID in generated:
    print("✓ Fine-tuning worked — model emits tool call tokens!")
else:
    print("⚠ Tool tokens not emitted. Try more epochs or higher LoRA rank.")\
"""),

code("""\
from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)
create_repo(HF_OUTPUT_REPO, private=True, exist_ok=True, token=HF_TOKEN)
print(f"Repo: https://huggingface.co/{HF_OUTPUT_REPO}")

for f in OUT_DIR.iterdir():
    if not f.is_file():
        continue
    print(f"Uploading {f.name}  ({f.stat().st_size/1e9:.2f} GB) ...")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=HF_OUTPUT_REPO, token=HF_TOKEN)
    print(f"  ✓ {f.name}")

print(f'''
Upload complete!

On your server:
  bash run_server.sh --hf-model {HF_OUTPUT_REPO}
''')\
"""),

]), "02_finetune_lora.ipynb")

print("Done.")
