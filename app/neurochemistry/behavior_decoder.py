import numpy as np
 
BEHAVIOR_MAPPINGS = [
    ([0.15, 0.25, 0.80, 0.70, 0.05, 0.85, 0.10], "cold, defensive, hypervigilant, dismissive"),
    ([0.20, 0.30, 0.70, 0.60, 0.10, 0.80, 0.15], "alert, suspicious, terse, guarded"),
    ([0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20], "wary, measured, cautious, reserved"),
    ([0.30, 0.20, 0.75, 0.75, 0.10, 0.70, 0.15], "irritated, confrontational, sharp, intense"),
    ([0.40, 0.25, 0.80, 0.80, 0.10, 0.75, 0.20], "angry, aggressive, hostile, fierce"),
    ([0.10, 0.15, 0.95, 0.95, 0.10, 0.90, 0.05], "terrified, panicked, frozen, overwhelmed"),
    ([0.15, 0.20, 0.85, 0.85, 0.15, 0.80, 0.10], "fearful, anxious, trembling, alert"),
    ([0.10, 0.20, 0.60, 0.20, 0.30, 0.30, 0.10], "depressed, hopeless, empty, numb"),
    ([0.15, 0.25, 0.55, 0.25, 0.35, 0.35, 0.15], "sad, melancholic, low, withdrawn"),
    ([0.25, 0.35, 0.45, 0.30, 0.15, 0.40, 0.25], "lonely, isolated, yearning, hollow"),
    ([0.20, 0.45, 0.30, 0.15, 0.35, 0.30, 0.30], "bored, disengaged, listless, passive"),
    ([0.75, 0.65, 0.25, 0.45, 0.55, 0.80, 0.65], "curious, engaged, analytical, focused"),
    ([0.70, 0.60, 0.30, 0.40, 0.50, 0.85, 0.60], "inquisitive, attentive, thoughtful, eager"),
    ([0.90, 0.75, 0.15, 0.55, 0.70, 0.55, 0.90], "joyful, warm, expressive, vibrant"),
    ([0.85, 0.70, 0.20, 0.50, 0.65, 0.60, 0.85], "happy, upbeat, friendly, energetic"),
    ([0.95, 0.85, 0.10, 0.70, 0.80, 0.60, 0.95], "ecstatic, elated, radiant, exuberant"),
    ([0.70, 0.80, 0.15, 0.30, 0.85, 0.50, 0.75], "grateful, warm, gracious, sincere"),
    ([0.35, 0.60, 0.40, 0.25, 0.85, 0.45, 0.40], "compassionate, gentle, supportive, empathetic"),
    ([0.75, 0.70, 0.15, 0.40, 0.90, 0.50, 0.80], "loving, affectionate, devoted, warm"),
    ([0.80, 0.70, 0.20, 0.60, 0.70, 0.55, 0.85], "playful, witty, lighthearted, humorous"),
    ([0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50], "balanced, neutral, steady, composed"),
    ([0.60, 0.85, 0.10, 0.20, 0.70, 0.40, 0.75], "peaceful, serene, tranquil, relaxed"),
    ([0.80, 0.75, 0.25, 0.55, 0.50, 0.70, 0.75], "confident, assertive, commanding, decisive"),
    ([0.85, 0.60, 0.30, 0.75, 0.55, 0.70, 0.80], "excited, energized, eager, animated"),
    ([0.95, 0.40, 0.45, 0.90, 0.50, 0.90, 0.90], "manic, hyperactive, scattered, intense"),
    ([0.25, 0.20, 0.85, 0.75, 0.10, 0.90, 0.15], "paranoid, suspicious, hypervigilant, distrustful"),
    ([0.20, 0.40, 0.70, 0.50, 0.10, 0.90, 0.20], "guarded, skeptical, cold, dismissive"),
    ([0.30, 0.35, 0.60, 0.55, 0.25, 0.70, 0.30], "frustrated, stuck, impatient, agitated"),
]
 
class BehaviorDecoder:
    def __init__(self):
        self.vectors = np.array([m[0] for m in BEHAVIOR_MAPPINGS])
        self.behaviors = [m[1] for m in BEHAVIOR_MAPPINGS]
    
    def decode(self, vec):
        v = np.array(vec)
        dist = np.linalg.norm(self.vectors - v, axis=1)
        return self.behaviors[np.argmin(dist)]
 
behavior_decoder = BehaviorDecoder()
 
def decode_hormones_to_behavior(h):
    vec = [
        h.get('dopamine', 0.5),
        h.get('serotonin', 0.5),
        h.get('cortisol', 0.3),
        h.get('adrenaline', 0.3),
        h.get('oxytocin', 0.5),
        h.get('norepinephrine', 0.5),
        h.get('endorphins', 0.5)
    ]
    return behavior_decoder.decode(vec)
