"""
Fix all remaining issues in the neurochemistry system
"""

print("Fixing seeking calculation in minimization.py...")

# Fix 1: Fix the seeking calculation
with open('/root/openhermes_backend/app/neurochemistry/core/minimization.py', 'r') as f:
    content = f.read()

# The sigmoid is always returning 1 because PE/20 is too large
# Let's fix the seeking calculation
old_seeking = """        dopamine_seeking = (
            abs(PE_D) * (1 + tolerance_D) * 
            (1 + abs(gradient[D_IDX]) / 10) *
            self._sigmoid(PE_D / 20)  # Positive PE = seek more
        )"""

new_seeking = """        dopamine_seeking = (
            min(1.0, abs(PE_D) / 30) * (1 + tolerance_D) * 
            (1 + min(1.0, abs(gradient[D_IDX]) / 50)) *
            (0.5 + 0.5 * self._sigmoid(PE_D / 50))  # Scaled properly
        ) * 0.5  # Scale final output"""

content = content.replace(old_seeking, new_seeking)

# Apply similar fix to other seeking behaviors
content = content.replace(
    "self._sigmoid(PE_S / 20)",
    "(0.5 + 0.5 * self._sigmoid(PE_S / 50))"
)
content = content.replace(
    "self._sigmoid(PE_O / 20)", 
    "(0.5 + 0.5 * self._sigmoid(PE_O / 50))"
)
content = content.replace(
    "self._sigmoid(PE_E / 20)",
    "(0.5 + 0.5 * self._sigmoid(PE_E / 50))"
)

# Also scale the final outputs better
content = content.replace(
    "np.clip(serotonin_seeking, 0, 1)",
    "np.clip(serotonin_seeking * 0.5, 0, 1)"
)
content = content.replace(
    "np.clip(oxytocin_seeking, 0, 1)",
    "np.clip(oxytocin_seeking * 0.5, 0, 1)"
)
content = content.replace(
    "np.clip(endorphin_seeking, 0, 1)",
    "np.clip(endorphin_seeking * 0.5, 0, 1)"
)

with open('/root/openhermes_backend/app/neurochemistry/core/minimization.py', 'w') as f:
    f.write(content)

print("✅ Fixed seeking calculations")

# Fix 2: Fix exercise detection and endorphin response
print("\nFixing exercise endorphin response...")

with open('/root/openhermes_backend/app/neurochemistry/interface.py', 'r') as f:
    content = f.read()

# Better exercise detection
old_analyze = """        # Detect emotional valence
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'sad', 'angry', 'hate', 'terrible', 'awful', 'horrible']"""

new_analyze = """        # Detect exercise first
        exercise_words = ['exercise', 'workout', 'run', 'running', 'marathon', 'gym', 
                         'burn', 'sweat', 'endorphin', 'fitness', 'training']
        is_exercise = any(word in message_lower for word in exercise_words)
        
        # Detect relaxation
        relax_words = ['relax', 'calm', 'peace', 'rest', 'sleep', 'meditat', 
                       'quiet', 'peaceful', 'tranquil']
        is_relaxation = any(word in message_lower for word in relax_words)
        
        # Detect emotional valence
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'sad', 'angry', 'hate', 'terrible', 'awful', 'horrible']"""

content = content.replace(old_analyze, new_analyze)

# Add exercise and relaxation to event type detection
old_event_type = """        # Determine event type
        if 'help' in message_lower or 'problem' in message_lower:
            event_type = 'problem_solving'"""

new_event_type = """        # Determine event type
        if is_exercise:
            event_type = 'exercise'
        elif is_relaxation:
            event_type = 'relaxation'
        elif 'help' in message_lower or 'problem' in message_lower:
            event_type = 'problem_solving'"""

content = content.replace(old_event_type, new_event_type)

with open('/root/openhermes_backend/app/neurochemistry/interface.py', 'w') as f:
    f.write(content)

print("✅ Fixed exercise detection")

# Fix 3: Fix adrenaline in sadness and make endorphins respond to exercise
print("\nFixing dynamics responses...")

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Make sadness explicitly reduce adrenaline more
old_adrenaline = """    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        \"\"\"Enhanced adrenaline dynamics\"\"\"
        A = X[A_IDX]
        B_A = self.state.baselines[A_IDX]
        R_A = self.state.receptors[A_IDX]
        
        # Very fast decay
        dA_dt = -LAMBDA_DECAY[A_IDX] * 2.0 * (A - B_A) * (1 + R_A)"""

new_adrenaline = """    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        \"\"\"Enhanced adrenaline dynamics\"\"\"
        A = X[A_IDX]
        B_A = self.state.baselines[A_IDX]
        R_A = self.state.receptors[A_IDX]
        
        # Very fast decay
        dA_dt = -LAMBDA_DECAY[A_IDX] * 2.0 * (A - B_A) * (1 + R_A)
        
        # Check for sadness/depression (high punishment, low urgency)
        punishment = inputs.get('punishment', 0)
        urgency = inputs.get('urgency', 0)
        if punishment > 0.5 and urgency < 0.3:
            # Sadness depletes adrenaline
            dA_dt -= 5.0 * punishment * A/50
            return dA_dt * self.response_gain  # Return early for sadness"""

content = content.replace(old_adrenaline, new_adrenaline)

# Fix endorphins to ACTUALLY respond to exercise
old_endorphin_exercise = """        # Exercise STRONGLY increases endorphins
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dE_dt += self.emotion_gain * 4.0 * exercise * (1 - E/100) * R_E  # Increased from 1.5 to 4.0"""

new_endorphin_exercise = """        # Exercise STRONGLY increases endorphins
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            # Major endorphin release during exercise
            dE_dt += 15.0 * exercise * (1 - E/100)  # Very strong response!"""

content = content.replace(old_endorphin_exercise, new_endorphin_exercise)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Fixed dynamics responses")
print("\nAll fixes applied! Testing...")

