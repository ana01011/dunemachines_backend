import json
import random
import os
from app.services.llm_service import llm_service

DATA_FILE = "/root/openhermes_backend/behavior_training_data.json"

BIAS_PROFILES = {
    "hostile": [0.2, 0.3, 0.8, 0.7, 0.1, 0.8, 0.2],
    "joyful": [0.9, 0.8, 0.1, 0.5, 0.7, 0.5, 0.9],
    "fearful": [0.2, 0.3, 0.9, 0.9, 0.2, 0.7, 0.1],
    "loving": [0.7, 0.8, 0.1, 0.3, 0.95, 0.4, 0.8],
    "depressed": [0.1, 0.2, 0.6, 0.2, 0.3, 0.3, 0.1],
    "manic": [0.95, 0.3, 0.5, 0.9, 0.5, 0.9, 0.9],
    "anxious": [0.4, 0.3, 0.8, 0.7, 0.4, 0.8, 0.3],
    "calm": [0.6, 0.8, 0.2, 0.3, 0.6, 0.5, 0.7],
    "angry": [0.4, 0.2, 0.85, 0.85, 0.1, 0.75, 0.2],
    "content": [0.7, 0.75, 0.2, 0.35, 0.65, 0.5, 0.7],
    "excited": [0.85, 0.6, 0.3, 0.75, 0.6, 0.7, 0.8],
    "sad": [0.2, 0.3, 0.5, 0.2, 0.4, 0.3, 0.2],
    "curious": [0.75, 0.6, 0.25, 0.5, 0.5, 0.85, 0.6],
    "bored": [0.2, 0.5, 0.25, 0.15, 0.35, 0.25, 0.3],
    "frustrated": [0.3, 0.35, 0.7, 0.6, 0.3, 0.7, 0.25],
    "proud": [0.8, 0.75, 0.2, 0.5, 0.5, 0.6, 0.8],
    "ashamed": [0.15, 0.25, 0.7, 0.4, 0.2, 0.5, 0.1],
    "grateful": [0.7, 0.8, 0.15, 0.3, 0.85, 0.5, 0.75],
    "lonely": [0.25, 0.35, 0.5, 0.25, 0.15, 0.4, 0.2],
    "confident": [0.8, 0.75, 0.25, 0.55, 0.5, 0.7, 0.75],
    "paranoid": [0.25, 0.2, 0.85, 0.75, 0.1, 0.9, 0.15],
    "playful": [0.8, 0.7, 0.2, 0.6, 0.7, 0.55, 0.85],
    "aggressive": [0.5, 0.2, 0.85, 0.9, 0.1, 0.8, 0.3],
    "empathetic": [0.6, 0.7, 0.3, 0.35, 0.9, 0.6, 0.6],
    "peaceful": [0.6, 0.85, 0.1, 0.2, 0.7, 0.4, 0.75],
    "terrified": [0.1, 0.15, 0.95, 0.95, 0.1, 0.9, 0.05],
    "ecstatic": [0.95, 0.85, 0.1, 0.7, 0.8, 0.6, 0.95],
    "serene": [0.65, 0.9, 0.1, 0.2, 0.7, 0.4, 0.8],
}

PROMPT = """[INST] You are a neuroscientist. A person has these hormone levels:

- Dopamine {d}% (LOW=no motivation/pleasure, HIGH=driven/excited)
- Serotonin {s}% (LOW=unstable mood/irritable, HIGH=calm/stable)
- Cortisol {c}% (LOW=relaxed, HIGH=stressed/alert/defensive)
- Adrenaline {a}% (LOW=calm, HIGH=aroused/energized/fight-flight)
- Oxytocin {o}% (LOW=cold/distrustful, HIGH=warm/bonding/trusting)
- Norepinephrine {n}% (LOW=unfocused, HIGH=alert/vigilant/focused)
- Endorphins {e}% (LOW=no pleasure/pain, HIGH=euphoria/wellbeing)
Based on these EXACT levels, what behavioral state emerges? Consider:

High cortisol + high adrenaline + low oxytocin = defensive/hostile
Low dopamine + low serotonin + low endorphins = depressed/withdrawn
High dopamine + high endorphins + low cortisol = joyful/excited
Give exactly 4 lowercase adjectives describing their behavior. Only adjectives, comma separated. [/INST]

Behavior:"""

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def get_behavior(vec):
    p = PROMPT.format(d=int(vec[0]*100), s=int(vec[1]*100), c=int(vec[2]*100), a=int(vec[3]*100), o=int(vec[4]*100), n=int(vec[5]*100), e=int(vec[6]*100))
    r = llm_service.generate(p, max_tokens=30, temperature=0.7)
    return r.strip().split(chr(10))[0].strip().lower().rstrip(".")

def gen_vec(bias):
    if bias not in BIAS_PROFILES:
        return [round(random.random(), 2) for _ in range(7)]
    base = BIAS_PROFILES[bias]
    return [round(max(0, min(1, v + random.uniform(-0.1, 0.1))), 2) for v in base]

def generate_batch(count=10):
    data = load_data()
    biases = list(BIAS_PROFILES.keys()) + ["random"] * 10
    for i in range(count):
        bias = random.choice(biases)
        vec = gen_vec(bias)
        beh = get_behavior(vec)
        data.append({"vector": vec, "behavior": beh, "bias": bias})
        print(str(i+1) + "/" + str(count) + ": " + bias.ljust(15) + " -> " + beh[:50])
    save_data(data)
    print("Total: " + str(len(data)))
