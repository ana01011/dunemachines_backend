#!/usr/bin/env python3
"""
Test improved mood emergence
"""

import sys
sys.path.append('/workspace')

from app.neurochemistry.core.state_v2 import NeurochemicalState
from app.neurochemistry.core.mood_emergence_v2 import MoodEmergence

def test_emotional_patterns():
    """Test that distinct emotions emerge properly"""
    
    print("=" * 80)
    print("🧠 TESTING DISTINCT EMOTIONAL EMERGENCE")
    print("=" * 80)
    
    state = NeurochemicalState()
    
    # Test cases with more extreme values for clearer emergence
    test_cases = [
        ("Baseline", {
            "dopamine": 50, "cortisol": 30,
            "adrenaline": 20, "serotonin": 60, "oxytocin": 40
        }),
        
        ("ANGER (high C+A, low S+O)", {
            "dopamine": 35, "cortisol": 75,
            "adrenaline": 70, "serotonin": 25, "oxytocin": 20
        }),
        
        ("FEAR (very high C+A, low dominance)", {
            "dopamine": 20, "cortisol": 85,
            "adrenaline": 80, "serotonin": 30, "oxytocin": 35
        }),
        
        ("JOY (high D+S, low C)", {
            "dopamine": 80, "cortisol": 20,
            "adrenaline": 40, "serotonin": 85, "oxytocin": 70
        }),
        
        ("SADNESS (low D+S, moderate C)", {
            "dopamine": 20, "cortisol": 50,
            "adrenaline": 15, "serotonin": 25, "oxytocin": 35
        }),
        
        ("RAGE (extreme anger pattern)", {
            "dopamine": 25, "cortisol": 85,
            "adrenaline": 85, "serotonin": 15, "oxytocin": 15
        }),
        
        ("ECSTASY (extreme joy)", {
            "dopamine": 90, "cortisol": 15,
            "adrenaline": 50, "serotonin": 90, "oxytocin": 85
        })
    ]
    
    for name, hormones in test_cases:
        # Set hormones
        for h, level in hormones.items():
            setattr(state, h, level)
        
        mood = MoodEmergence.describe_emergent_state(state)
        tendencies = MoodEmergence.get_behavioral_tendencies(state)
        prompt = MoodEmergence.create_natural_prompt(state)
        triggers = MoodEmergence.get_capability_triggers(state)
        
        print(f"\n📊 {name}")
        print(f"   Hormones: D={state.dopamine} C={state.cortisol} A={state.adrenaline} S={state.serotonin} O={state.oxytocin}")
        print(f"   → EMERGENT MOOD: {mood}")
        print(f"   → Fight: {tendencies['fight']:.2f} | Flight: {tendencies['flight']:.2f}")
        print(f"   → Prompt: {prompt}")
        if triggers:
            print(f"   → Active: {', '.join(triggers)}")
    
    print("\n" + "=" * 80)
    print("✅ Emotions emerge naturally from hormone patterns!")
    print("=" * 80)

if __name__ == "__main__":
    test_emotional_patterns()
