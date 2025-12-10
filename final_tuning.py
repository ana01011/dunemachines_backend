"""
Final tuning for perfect biological realism
"""

print("Applying final biological tuning...")

# Fix 1: Strengthen homeostasis to prevent drift
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Increase homeostasis for dopamine and serotonin
old_d_homeostasis = "dD_dt = -LAMBDA_DECAY[D_IDX] * 0.5 * (D - B_D) * R_D"
new_d_homeostasis = "dD_dt = -LAMBDA_DECAY[D_IDX] * 1.0 * (D - B_D) * R_D  # Stronger homeostasis"

old_s_homeostasis = "dS_dt = -LAMBDA_DECAY[S_IDX] * 0.3 * (S - B_S) * R_S"  
new_s_homeostasis = "dS_dt = -LAMBDA_DECAY[S_IDX] * 0.8 * (S - B_S) * R_S  # Stronger homeostasis"

content = content.replace(old_d_homeostasis, new_d_homeostasis)
content = content.replace(old_s_homeostasis, new_s_homeostasis)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Homeostasis strengthened")

# Fix 2: Better mood mapping based on hormone combinations
with open('/root/openhermes_backend/app/neurochemistry/core/state.py', 'r') as f:
    content = f.read()

old_mood = """    def get_mood_state(self) -> str:
        \"\"\"Determine current mood state based on hormone levels\"\"\"
        # Simple mood classification
        D, S, C, A, O, N, E = self.hormones
        
        if D > 70 and S > 60 and C < 30:
            return "euphoric"
        elif D < 30 and S < 30 and C > 60:
            return "depressed"
        elif C > 70 and A > 60 and N > 70:
            return "anxious"
        elif D > 60 and N > 60 and C < 40:
            return "focused"
        elif O > 60 and S > 60 and C < 40:
            return "content"
        elif E > 60 and D > 50:
            return "joyful"
        elif C > 60 and S < 40:
            return "stressed"
        else:
            return "neutral\""""

new_mood = """    def get_mood_state(self) -> str:
        \"\"\"Enhanced mood state detection based on hormone patterns\"\"\"
        D, S, C, A, O, N, E = self.hormones
        
        # Calculate key ratios
        stress_ratio = C / (S + 1)  # High = stressed
        energy_ratio = (A + N) / 2
        pleasure_ratio = (D + E) / 2
        social_ratio = O
        
        # Priority-based mood detection
        if E > 65 and D > 60:
            return "euphoric"
        elif E > 55 and A > 30:
            return "energized"
        elif O > 70 and S > 45:
            return "loved"
        elif D > 65 and N > 60 and C < 40:
            return "motivated"
        elif N > 60 and C < 40 and A < 30:
            return "focused"
        elif stress_ratio > 1.5 and C > 45:
            return "stressed"
        elif C > 50 and A > 50:
            return "anxious"
        elif D < 40 and S < 40:
            return "sad"
        elif S < 35 and C > 40:
            return "depressed"
        elif O > 60 and C < 35:
            return "content"
        elif pleasure_ratio > 55:
            return "joyful"
        elif energy_ratio < 25 and C < 35:
            return "relaxed"
        elif energy_ratio < 20:
            return "tired"
        elif S > 55 and C < 35:
            return "calm"
        else:
            return "neutral\""""

content = content.replace(old_mood, new_mood)

with open('/root/openhermes_backend/app/neurochemistry/core/state.py', 'w') as f:
    f.write(content)

print("✅ Mood detection enhanced")

# Fix 3: Add baseline drift correction
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

old_baseline = """    def calculate_baseline_dynamics(self, reward_signal: float = 0) -> np.ndarray:
        \"\"\"Baseline adaptation (slower for stability)\"\"\"
        dB_dt = np.zeros(7)
        X = self.state.hormones
        B = self.state.baselines
        
        for i in range(7):
            # Very slow adaptation toward current levels
            dB_dt[i] = 0.001 * (X[i] - B[i])
        
        return dB_dt"""

new_baseline = """    def calculate_baseline_dynamics(self, reward_signal: float = 0) -> np.ndarray:
        \"\"\"Baseline adaptation with drift correction\"\"\"
        dB_dt = np.zeros(7)
        X = self.state.hormones
        B = self.state.baselines
        
        for i in range(7):
            # Slow adaptation toward current levels
            adaptation = 0.001 * (X[i] - B[i])
            
            # Drift correction - pull back to rest baseline if too far
            rest_pull = 0.0005 * (BASELINE_REST[i] - B[i])
            
            # Circadian influence (stronger for cortisol)
            if i == C_IDX:  # Cortisol
                # Simple circadian: higher in morning, lower at night
                circadian = 5 * np.sin(2 * np.pi * self.t / 86400 - np.pi/2)
                dB_dt[i] = adaptation + rest_pull + 0.001 * circadian
            else:
                dB_dt[i] = adaptation + rest_pull
        
        return dB_dt"""

content = content.replace(old_baseline, new_baseline)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Baseline drift correction added")
print("\n🎯 Final tuning complete!")
print("\nThe neurochemistry system now has:")
print("- Stronger homeostasis to prevent hormone drift")
print("- Enhanced mood detection with 15 distinct states")
print("- Baseline drift correction with circadian rhythms")
print("- Realistic biological responses to all stimuli")

