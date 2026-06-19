"""
Tool-calling training-data generator (audio-grounded).

Produces abstract, turn-structured examples that are independent of the
tokenizer and of audio. Each example is a list of Turn objects:

    Turn.query   : str   -> the user's SPOKEN input (TTS'd into the user audio
                           stream).  None means a pure-silence turn.
    Turn.tool    : ("get_time", None) | ("get_weather", "London") | None
    Turn.result  : str   -> the tool result text injected into Moshi's text
                           stream (mask=0, it is force-fed at runtime).
    Turn.reply   : str   -> what Moshi SPEAKS after (its text monologue, mask=1).

Notebook 01 turns each example into a [17, T] code tensor:
    row 0     = text monologue   (PAD while listening, then call + reply)
    rows 1:9  = Moshi audio      (silence)
    rows 9:17 = user audio       (Mimi-encoded TTS of Turn.query)

This is the key difference from the old text-only data: the question now lives
in the *audio* rows, so the model learns to emit <|tool_call|> when it HEARS a
request, not when it reads one.

render_emit(turn, tok) returns the (tokens, mask) the model must produce in the
text stream after hearing that turn's audio.
"""
import random
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class Turn:
    query: Optional[str]                 # spoken user input (None = silence)
    reply: str = ""                      # Moshi's spoken reply
    tool: Optional[tuple] = None         # (name, arg) e.g. ("get_weather","London")
    result: Optional[str] = None         # injected tool result text


@dataclass
class Example:
    type: str
    turns: list = field(default_factory=list)


# ── builders ──────────────────────────────────────────────────────────────────
def _time():
    h, m = random.randint(1, 12), random.randint(0, 59)
    ap = random.choice(["AM", "PM"])
    day, mon, d = random.choice(_DAYS), random.choice(_MONTHS), random.randint(1, 28)
    result = f"{h}:{m:02d} {ap} on {day}, {mon} {d}, 2026"
    spoken = (f"{h} {ap.lower()}" if m == 0 else f"{h}:{m:02d} {ap.lower()}")
    return result, spoken


def _weather(city):
    cond = random.choice(_CONDITIONS)
    t = random.randint(-8, 40)
    temp = f"+{t}°C" if t >= 0 else f"{t}°C"
    return f"{city}: {cond}, {temp}", cond, temp


def _time_turn(query):
    result, spoken = _time()
    return Turn(query=query, tool=("get_time", None), result=result,
                reply=random.choice(_TIME_RESPONSES).format(s=spoken))


def _weather_turn(query, city):
    result, cond, temp = _weather(city)
    reply = random.choice(_WEATHER_RESPONSES).format(
        city=city, cond=cond, cond_l=cond.lower(), temp=temp)
    return Turn(query=query, tool=("get_weather", city), result=result, reply=reply)


def _make(kind, city=None):
    if kind == "time":
        return Example("time", [_time_turn(random.choice(_TIME_TRIGGERS))])
    if kind == "time_multiturn":
        return Example("time_multiturn", [
            _time_turn(random.choice(_TIME_TRIGGERS)),
            _time_turn(random.choice(_TIME_FOLLOWUPS)),
        ])
    if kind == "weather":
        city = city or random.choice(_CITIES)
        q = random.choice([f"what's the weather in {city}", f"weather in {city}",
                           f"how's the weather in {city}", f"what's it like in {city}",
                           f"is it raining in {city}", f"how hot is it in {city}",
                           f"tell me the weather in {city}", f"check the weather in {city}"])
        return Example("weather", [_weather_turn(q, city)])
    if kind == "weather_local":
        cond = random.choice(_CONDITIONS)
        t = random.randint(-8, 40)
        temp = f"+{t}°C" if t >= 0 else f"{t}°C"
        q = random.choice(["what's the weather like", "how's the weather",
                           "what's it like outside", "is it cold out", "do I need a jacket",
                           "what's the weather today", "how's it looking outside", "is it raining"])
        reply = random.choice(_WEATHER_LOCAL_RESP).format(cond_l=cond.lower(), temp=temp)
        return Example("weather_local",
                       [Turn(query=q, tool=("get_weather", None),
                             result=f"Local: {cond}, {temp}", reply=reply)])
    if kind == "weather_multiturn":
        c1, c2 = random.sample(_CITIES, 2)
        q2 = random.choice([f"and {c2}", f"what about {c2}", f"how about {c2}", f"and in {c2}"])
        return Example("weather_multiturn", [
            _weather_turn(f"what's the weather in {c1}", c1),
            _weather_turn(q2, c2),
        ])
    if kind == "chitchat":
        q, a = random.choice(_CHITCHAT)
        return Example("chitchat", [Turn(query=q, reply=a)])
    if kind == "distractor":
        q, a = random.choice(_DISTRACTORS)
        return Example("distractor", [Turn(query=q, reply=a)])
    if kind == "silence":
        return Example("silence", [Turn(query=None, reply="")])
    raise ValueError(kind)


def examples(seed=42):
    """Return the full list of abstract Examples."""
    random.seed(seed)
    out = []
    for _ in range(400):
        out.append(_make("time"))
    for _ in range(120):
        out.append(_make("time_multiturn"))
    for city in _CITIES:
        for _ in range(8):
            out.append(_make("weather", city))
    for _ in range(120):
        out.append(_make("weather"))
    for _ in range(120):
        out.append(_make("weather_local"))
    for _ in range(120):
        out.append(_make("weather_multiturn"))
    for _ in range(180):
        out.append(_make("silence"))
    for _ in range(220):
        out.append(_make("chitchat"))
    for _ in range(260):
        out.append(_make("distractor"))
    random.shuffle(out)
    return out


def render_emit(turn, tok):
    """Tokens + mask the model must EMIT in its text stream for this turn.

    mask=1 → trained (tool call + spoken reply); mask=0 → injected context
    (the <|tool_result|> block, which is force-fed at runtime).
    """
    def enc(s):
        return [i for i in tok.encode(s) if i < 32000]

    toks, mask = [], []
    if turn.tool is not None:
        name, arg = turn.tool
        call = name if arg is None else f"{name} {arg}"
        toks.append(TOOL_CALL_ID);                 mask.append(1)
        for t in enc(call):     toks.append(t);    mask.append(1)
        toks.append(TOOL_END_ID);                  mask.append(1)
        toks.append(TOOL_RESULT_ID);               mask.append(0)
        for t in enc(turn.result or ""): toks.append(t); mask.append(0)
        toks.append(TOOL_RESULT_END_ID);           mask.append(0)
    for t in enc(turn.reply):   toks.append(t);    mask.append(1)
    return toks, mask
