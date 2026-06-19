"""
Tool-calling training-data generator (shared by notebooks 01 and 02).

Call `generate(tok)` to get a list of {"tokens", "mask", "type"} rows, or
`save(path, tok)` to write the JSONL directly. Keeping this in one module means
notebook 02 can regenerate fresh data right after cloning — no stale committed
file, no GitHub round-trip.

Design:
- Call / result / time formats match the runtime tools exactly:
    call    "get_weather London"            (orchestrator._parse_call)
    result  "London: Clear, +27°C"          (registry._fetch_one, wttr.in %C,%t)
    time    "8:31 PM on Friday, June 19, 2026"  (registry.get_time)
- Principled masking: trainable=True (mask=1) ONLY on tokens the model must
  EMIT — the tool call and the spoken reply. The injected <|tool_result|> block
  is context (mask=0) because at runtime it is force-fed, not predicted.
- Heavy negatives (silence / chit-chat / distractors) so the model does NOT
  fire on silence or on sentences that merely contain "time"/"weather".
"""
import json
import random
from pathlib import Path

PAD_ID             = 3
TOOL_CALL_ID       = 32000
TOOL_END_ID        = 32001
TOOL_RESULT_ID     = 32002
TOOL_RESULT_END_ID = 32003

_DAYS   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

_CITIES = [
    "London", "New York", "Tokyo", "Paris", "Sydney", "Dubai", "Mumbai", "Toronto",
    "Berlin", "Singapore", "Los Angeles", "Seoul", "Bangkok", "Istanbul", "Cairo",
    "Amsterdam", "Madrid", "Rome", "Hong Kong", "Kuala Lumpur", "Dhaka", "Karachi",
    "Stockholm", "Oslo", "Helsinki", "Moscow", "Beijing", "Delhi", "Lagos", "Nairobi",
    "Chicago", "Boston", "Seattle", "Miami", "Vancouver", "Melbourne", "Auckland",
]
_CONDITIONS = [
    "Clear", "Sunny", "Partly cloudy", "Cloudy", "Overcast", "Mist", "Fog",
    "Patchy rain nearby", "Light rain", "Moderate rain", "Heavy rain",
    "Light drizzle", "Thundery outbreaks", "Light snow", "Haze",
]

_TIME_TRIGGERS = [
    "what time is it", "what time is it now", "tell me the time",
    "what's the current time", "do you know what time it is",
    "can you check the time", "what time do you have",
    "I need to know the time", "got the time", "hey what's the time",
    "could you tell me the time", "what's the time right now",
    "do you have the time", "what time is it over there",
]
_TIME_FOLLOWUPS = [
    "what about now", "and now", "what time is it now",
    "can you check again", "and the time now", "how about now",
]
_TIME_RESPONSES = [
    "It's {s} right now.", "The time is {s}.", "Right now it's {s}.",
    "It's currently {s}.", "It's {s} at the moment.", "Looks like it's {s}.",
]
_WEATHER_RESPONSES = [
    "It's {cond} in {city} right now, about {temp}.",
    "Looks like {cond_l} in {city}, around {temp}.",
    "In {city} it's {cond_l}, {temp} out there.",
    "{city} is {cond_l} at the moment, {temp}.",
    "Right now {city} has {cond_l} skies, {temp}.",
]
_WEATHER_LOCAL_RESP = [
    "It's {cond_l} where you are, about {temp}.",
    "Locally it's {cond_l}, around {temp}.",
    "Right now it's {cond_l} outside, {temp}.",
]
_CHITCHAT = [
    ("hey how are you doing", "I'm doing great, thanks for asking."),
    ("tell me a fun fact", "Sure — octopuses have three hearts."),
    ("what do you think about music", "I love it, music can really change a mood."),
    ("what's your favorite book", "I'm fond of a good science fiction novel."),
    ("can you recommend a movie", "If you like thrillers, Inception is a classic."),
    ("tell me about space", "Space is vast — billions of galaxies out there."),
    ("what is machine learning", "It's teaching computers to learn from data."),
    ("how do I cook pasta", "Boil salted water, add the pasta, simmer till tender."),
    ("what's a good hobby", "Photography is fun and gets you outdoors."),
    ("tell me a joke", "Why did the scarecrow win an award? He was outstanding in his field."),
    ("how was your day", "Pretty good, just enjoying the conversation."),
    ("what should I eat", "Maybe something light, like a fresh salad."),
]
_DISTRACTORS = [
    ("I had a great time at the party", "Sounds like a lot of fun!"),
    ("long time no see, how have you been", "I've been well, good to catch up."),
    ("it's about time we got started", "Agreed, let's dive in."),
    ("the weather was lovely on our trip", "That makes for a perfect getaway."),
    ("I've been feeling under the weather", "Sorry to hear that, hope you feel better soon."),
    ("time flies when you're having fun", "It really does!"),
    ("we had a wonderful time in Italy", "Italy is beautiful this time of year."),
    ("once upon a time there was a castle", "Ooh, I love a good story."),
    ("that movie was a waste of time", "Shame when a film doesn't deliver."),
    ("let's weather the storm together", "Absolutely, we'll get through it."),
    ("do you have time for a quick chat", "Of course, I'm happy to talk."),
    ("the timing of that was perfect", "Couldn't have gone better."),
    ("she gave me a hard time about it", "That can be frustrating."),
    ("spring is my favorite time of year", "The blossoms make it special."),
]


def _make(tok):
    """Return a dict of generator functions bound to a tokenizer."""
    def encode(text):
        return [i for i in tok.encode(text) if i < 32000]

    def build(segments, ex_type):
        tokens, mask = [], []
        for toks, train in segments:
            tokens.extend(toks)
            mask.extend([1 if train else 0] * len(toks))
        return {"tokens": tokens, "mask": mask, "type": ex_type}

    def gap(lo=4, hi=18):
        return [PAD_ID] * random.randint(lo, hi)

    # ── time ──────────────────────────────────────────────────────────────────
    def real_time():
        h, m = random.randint(1, 12), random.randint(0, 59)
        ap = random.choice(["AM", "PM"])
        day, mon, d = random.choice(_DAYS), random.choice(_MONTHS), random.randint(1, 28)
        result = f"{h}:{m:02d} {ap} on {day}, {mon} {d}, 2026"
        spoken = (f"{h} {ap.lower()}" if m == 0 else f"{h}:{m:02d} {ap.lower()}")
        return result, spoken

    def time_turn(trigger):
        result, spoken = real_time()
        return [
            (encode(trigger), False), (gap(), False),
            ([TOOL_CALL_ID], True), (encode("get_time"), True), ([TOOL_END_ID], True),
            ([TOOL_RESULT_ID], False), (encode(result), False), ([TOOL_RESULT_END_ID], False),
            (encode(random.choice(_TIME_RESPONSES).format(s=spoken)), True),
        ]

    def time_example():
        return build(time_turn(random.choice(_TIME_TRIGGERS)), "time")

    def time_multiturn():
        segs = time_turn(random.choice(_TIME_TRIGGERS))
        segs.append((gap(8, 25), False))
        segs.extend(time_turn(random.choice(_TIME_FOLLOWUPS)))
        return build(segs, "time_multiturn")

    # ── weather ────────────────────────────────────────────────────────────────
    def real_weather(city):
        cond = random.choice(_CONDITIONS)
        t = random.randint(-8, 40)
        temp = f"+{t}°C" if t >= 0 else f"{t}°C"
        return f"{city}: {cond}, {temp}", cond, temp

    def weather_turn(trigger, city):
        result, cond, temp = real_weather(city)
        resp = random.choice(_WEATHER_RESPONSES).format(
            city=city, cond=cond, cond_l=cond.lower(), temp=temp)
        return [
            (encode(trigger), False), (gap(), False),
            ([TOOL_CALL_ID], True), (encode(f"get_weather {city}"), True), ([TOOL_END_ID], True),
            ([TOOL_RESULT_ID], False), (encode(result), False), ([TOOL_RESULT_END_ID], False),
            (encode(resp), True),
        ]

    def weather_example(city=None):
        city = city or random.choice(_CITIES)
        triggers = [f"what's the weather in {city}", f"weather in {city}",
                    f"how's the weather in {city}", f"what's it like in {city}",
                    f"is it raining in {city}", f"how hot is it in {city}",
                    f"tell me the weather in {city}", f"check the weather in {city}"]
        return build(weather_turn(random.choice(triggers), city), "weather")

    def weather_local():
        cond = random.choice(_CONDITIONS)
        t = random.randint(-8, 40)
        temp = f"+{t}°C" if t >= 0 else f"{t}°C"
        result = f"Local: {cond}, {temp}"
        trigger = random.choice([
            "what's the weather like", "how's the weather", "what's it like outside",
            "is it cold out", "do I need a jacket", "what's the weather today",
            "how's it looking outside", "is it raining",
        ])
        resp = random.choice(_WEATHER_LOCAL_RESP).format(cond_l=cond.lower(), temp=temp)
        segs = [
            (encode(trigger), False), (gap(), False),
            ([TOOL_CALL_ID], True), (encode("get_weather"), True), ([TOOL_END_ID], True),
            ([TOOL_RESULT_ID], False), (encode(result), False), ([TOOL_RESULT_END_ID], False),
            (encode(resp), True),
        ]
        return build(segs, "weather_local")

    def weather_multiturn():
        c1, c2 = random.sample(_CITIES, 2)
        segs = weather_turn(f"what's the weather in {c1}", c1)
        segs.append((gap(8, 25), False))
        segs.extend(weather_turn(random.choice([f"and {c2}", f"what about {c2}",
                                                f"how about {c2}", f"and in {c2}"]), c2))
        return build(segs, "weather_multiturn")

    # ── negatives ────────────────────────────────────────────────────────────────
    def silence_example():
        n = random.randint(20, 60)
        k = random.randint(n // 3, n // 2)
        return {"tokens": [PAD_ID] * n, "mask": [0] * k + [1] * (n - k), "type": "silence"}

    def chitchat_example():
        q, a = random.choice(_CHITCHAT)
        return build([(encode(q), False), (gap(), False), (encode(a), True)], "chitchat")

    def distractor_example():
        q, a = random.choice(_DISTRACTORS)
        return build([(encode(q), False), (gap(), False), (encode(a), True)], "distractor")

    return locals()


def generate(tok, seed=42):
    """Build and return the full dataset (list of rows)."""
    random.seed(seed)
    g = _make(tok)
    data = []
    for _ in range(500):
        data.append(g["time_example"]())
    for _ in range(150):
        data.append(g["time_multiturn"]())
    for city in _CITIES:
        for _ in range(12):
            data.append(g["weather_example"](city))
    for _ in range(150):
        data.append(g["weather_example"]())
    for _ in range(150):
        data.append(g["weather_local"]())
    for _ in range(150):
        data.append(g["weather_multiturn"]())
    for _ in range(200):
        data.append(g["silence_example"]())
    for _ in range(250):
        data.append(g["chitchat_example"]())
    for _ in range(300):
        data.append(g["distractor_example"]())
    random.shuffle(data)
    return data


def save(path, tok, seed=42):
    """Generate and write the dataset to a JSONL file; returns the dataset."""
    data = generate(tok, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    return data
