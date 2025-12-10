"""
Fix specific dynamics issues for more realistic responses
"""

# Read the current dynamics file
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Fix 1: Exercise should STRONGLY boost endorphins
old_exercise_endorphins = """        # Exercise STRONGLY increases endorphins
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dE_dt += self.emotion_gain * 1.5 * exercise * (1 - E/100) * R_E"""

new_exercise_endorphins = """        # Exercise STRONGLY increases endorphins
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dE_dt += self.emotion_gain * 4.0 * exercise * (1 - E/100) * R_E  # Increased from 1.5 to 4.0"""

content = content.replace(old_exercise_endorphins, new_exercise_endorphins)

# Fix 2: Sadness should NOT increase adrenaline
old_sadness_check = """        # SHARP urgency/threat response
        urgency = inputs.get('urgency', 0)
        threat = inputs.get('threat', 0)
        fight_flight = inputs.get('fight_flight', 0)
        
        emergency = urgency + threat + fight_flight"""

new_sadness_check = """        # SHARP urgency/threat response
        urgency = inputs.get('urgency', 0)
        threat = inputs.get('threat', 0)
        fight_flight = inputs.get('fight_flight', 0)
        
        # Reduce adrenaline if it's sadness (punishment without urgency)
        punishment = inputs.get('punishment', 0)
        if punishment > 0 and urgency < 0.3:
            dA_dt -= 2.0 * punishment * A/100  # Sadness depletes adrenaline
        
        emergency = urgency + threat + fight_flight"""

content = content.replace(old_sadness_check, new_sadness_check)

# Fix 3: Relaxation should strongly DECREASE adrenaline  
old_relaxation = """        # Relaxation decreases adrenaline
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dA_dt -= 4.0 * sleep * A/100"""

new_relaxation = """        # Relaxation strongly decreases adrenaline
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dA_dt -= 8.0 * sleep * A/100  # Increased relaxation effect
        
        # Also check for low arousal (relaxation without sleep)
        arousal = inputs.get('urgency', 0) + inputs.get('attention', 0)
        if arousal < 0.2:  # Very low arousal = relaxation
            dA_dt -= 3.0 * (1 - arousal) * A/100"""

content = content.replace(old_relaxation, new_relaxation)

# Write the fixed content back
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Dynamics fixed!")
print("\nChanges made:")
print("1. Exercise now boosts endorphins 4x instead of 1.5x")
print("2. Sadness now decreases adrenaline instead of increasing it")
print("3. Relaxation now decreases adrenaline 2x more effectively")
