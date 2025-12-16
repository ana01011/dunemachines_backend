
import numpy as np
from typing import Dict, List
 
BEHAVIOR_MAPPINGS = [
    ([0.15, 0.25, 0.80, 0.70, 0.05, 0.85, 0.10], "cold, defensive, hyper-vigilant, dismissive"),
    ([0.20, 0.30, 0.70, 0.60, 0.10, 0.80, 0.15], "alert, suspicious, terse, distant"),
    ([0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20], "wary, measured, cautious, reserved"),
    ([0.30, 0.20, 0.75, 0.75, 0.10, 0.70, 0.15], "irritated, confrontational, sharp, intense"),
    ([0.75, 0.65, 0.25, 0.45, 0.55, 0.80, 0.65], "curious, engaged, analytical, focused"),
    ([0.70, 0.60, 0.30, 0.40, 0.50, 0.85, 0.60], "inquisitive, attentive, thoughtful, eager"),
    ([0.90, 0.75, 0.15, 0.55, 0.70, 0.55, 0.90], "joyful, warm, expressive, vibrant"),
    ([0.85, 0.70, 0.20, 0.50, 0.65, 0.60, 0.85], "pleased, upbeat, friendly, energetic"),
    ([0.70, 0.80, 0.15, 0.30, 0.85, 0.50, 0.75], "appreciative, warm, gracious, sincere"),
    ([0.35, 0.60, 0.40, 0.25, 0.85, 0.45, 0.40], "compassionate, gentle, supportive, empathetic"),
    ([0.40, 0.55, 0.45, 0.30, 0.80, 0.50, 0.45], "sympathetic, patient, nurturing, caring"),
    ([0.45, 0.40, 0.65, 0.60, 0.50, 0.75, 0.35], "tense, alert, restless, on-edge"),
    ([0.80, 0.70, 0.20, 0.50, 0.70, 0.50, 0.85], "playful, witty, lighthearted, humorous"),
    ([0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50], "balanced, neutral, steady, composed"),
    ([0.30, 0.50, 0.25, 0.20, 0.35, 0.35, 0.35], "detached, indifferent, minimal, brief"),
    ([0.70, 0.70, 0.35, 0.50, 0.45, 0.70, 0.65], "confident, assertive, direct, commanding"),
    ([0.20, 0.40, 0.70, 0.50, 0.10, 0.90, 0.20], "suspicious, guarded, penetrating, dismissive"),
]
 
class BehaviorDecoder:
    def __init__(self):
        self.vectors = np.array([m[0] for m in BEHAVIOR_MAPPINGS])
        self.behaviors = [m[1] for m in BEHAVIOR_MAPPINGS]
    
    def decode(self, hormone_vector):
        vec = np.array(hormone_vector)
        distances = np.linalg.norm(self.vectors - vec, axis=1)
        closest_idx = np.argmin(distances)
        return self.behaviors[closest_idx]
 
behavior_decoder = BehaviorDecoder()
 
def decode_hormones_to_behavior(hormones):
    vec = [hormones.get("dopamine", 0.5), hormones.get("serotonin", 0.5),
           hormones.get("cortisol", 0.3), hormones.get("adrenaline", 0.3),
           hormones.get("oxytocin", 0.5), hormones.get("norepinephrine", 0.5),
           hormones.get("endorphins", 0.5)]
    return behavior_decoder.decode(vec)
