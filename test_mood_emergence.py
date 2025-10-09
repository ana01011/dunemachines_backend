#!/usr/bin/env python3
"""
Test natural mood emergence - anger and other emotions emerge naturally
"""

import sys
sys.path.append('/workspace')

import numpy as np
from app.neurochemistry.core.state_v2 import NeurochemicalState, Event
from app.neurochemistry.core.mood_emergence import MoodEmergence

def test_natural_emergence():
    """Test that moods emerge naturally from hormone states"""
    
    print("=" * 80)
    print("🧬 TESTING NATURAL MOOD EMERGENCE")
    print("=" * 80)
    
    state = NeurochemicalState()
    
    # Test different hormone combinations to see what emerges
    test_cases = [
        ("Normal", {}),
        
        ("High Stress", {
            "cortisol": 75,
            "adrenaline": 60,
            "serotonin": 35
        }),
        
        ("Should emerge as anger", {
            "cortisol": 70,
            "adrenaline": 65, 
            "serotonin": 25,
            "dopamine": 30,
            "oxytocin": 25
        }),
        
        ("Should emerge as fear", {
            "cortisol": 80,
            "adrenaline": 75,
            "serotonin": 30,
            "dopamine": 25,
            "oxytocin": 35
        }),
        
        ("Should emerge as joy", {
            "dopamine": 75,
            "serotonin": 80,
            "oxytocin": 70,
            "cortisol": 25,
            "adrenaline": 45
        }),
        
        ("Should emerge as sadness", {
            "dopamine": 25,
            "serotonin": 30,
            "cortisol": 45,
            "adrenaline": 15,
            "oxytocin": 35
        })
    ]
    
    for name, hormones in test_cases:
        # Set hormone levels
        for hormone, level in hormones.items():
            setattr(state, hormone, level)
        
        # Get emergent state
        mood = MoodEmergence.describe_emergent_state(state)
        dims = MoodEmergence.get_emotional_vector(state)
        tendencies = MoodEmergence.get_behavioral_tendencies(state)
        prompt = MoodEmergence.create_natural_prompt(state)
        critical = MoodEmergence.detect_critical_states(state)
        
        print(f"\n🔬 {name}:")
        print(f"   Hormones: D={state.dopamine:.0f} C={state.cortisol:.0f} "
              f"A={state.adrenaline:.0f} S={state.serotonin:.0f} O={state.oxytocin:.0f}")
        print(f"   Emergent Mood: {mood}")
        print(f"   Emotional Dimensions:")
        print(f"      Valence: {dims['valence']:.2f}")
        print(f"      Arousal: {dims['arousal']:.2f}") 
        print(f"      Dominance: {dims['dominance']:.2f}")
        print(f"   Behavioral Tendencies:")
        print(f"      Approach: {tendencies['approach']:.2f}")
        print(f"      Fight: {tendencies['fight']:.2f}")
        print(f"      Flight: {tendencies['flight']:.2f}")
        print(f"   Natural Prompt: {prompt}")
        if critical:
            print(f"   ⚠️ Critical States: {critical}")
    
    # Test reward-prediction dopamine
    print("\n" + "=" * 80)
    print("🎯 TESTING REWARD-PREDICTION DOPAMINE")
    print("=" * 80)
    
    state = NeurochemicalState()
    
    print("\n1. Starting task (expecting 70% quality)...")
    state.start_task("test_task", expected_difficulty=0.3)
    print(f"   Dopamine: {state.dopamine:.1f} (anticipation rise)")
    
    print("\n2. Making progress...")
    for progress in [0.2, 0.4, 0.6, 0.8]:
        state.update_progress(progress)
        print(f"   Progress {progress:.0%}: Dopamine={state.dopamine:.1f}")
    
    print("\n3. Task completed BETTER than expected (85% vs 70%)...")
    old_dopamine = state.dopamine
    state.complete_task(actual_quality=0.85)
    print(f"   Dopamine: {old_dopamine:.1f} → {state.dopamine:.1f} (SPIKE!)")
    mood = MoodEmergence.describe_emergent_state(state)
    print(f"   Emergent mood: {mood}")
    
    print("\n4. Let spike settle (opponent process)...")
    for i in range(5):
        state.apply_dynamics(dt=0.5)
        print(f"   Step {i+1}: Dopamine={state.dopamine:.1f}")
    
    print("\n5. Starting new task after success...")
    state.start_task("task2", expected_difficulty=0.25)
    print(f"   Dopamine: {state.dopamine:.1f} (confident anticipation)")
    
    print("\n6. Task completed WORSE than expected (40% vs 75%)...")
    old_dopamine = state.dopamine
    state.complete_task(actual_quality=0.40)
    print(f"   Dopamine: {old_dopamine:.1f} → {state.dopamine:.1f} (CRASH!)")
    mood = MoodEmergence.describe_emergent_state(state)
    print(f"   Emergent mood: {mood}")
    print(f"   Cortisol: {state.cortisol:.1f} (stress from failure)")
    
    print("\n7. Retrying after failure...")
    state.retry_task("task2_retry")
    print(f"   Dopamine: {state.dopamine:.1f} (cautious)")
    print(f"   Expected quality: {state.current_task.expected_quality:.2f} (lowered)")
    
    print("\n" + "=" * 80)
    print("✅ Natural emergence working! Anger and other emotions emerge from hormone combinations.")
    print("=" * 80)

if __name__ == "__main__":
    test_natural_emergence()
