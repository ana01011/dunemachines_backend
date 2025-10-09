#!/usr/bin/env python3
"""
Complete test of neurochemical mathematics
Tests all dynamics, stability, and mood mapping
"""

import sys
sys.path.append('/workspace')

import numpy as np
from app.neurochemistry.core.state import NeurochemicalState, Event
from app.neurochemistry.core.mood_mapper import MoodMapper
import json
import time

def test_mathematical_framework():
    """Test complete mathematical framework"""
    print("=" * 80)
    print("🧬 NEUROCHEMICAL MATHEMATICS TEST - Complete Framework")
    print("=" * 80)
    
    # Initialize state
    state = NeurochemicalState()
    
    # Show initial conditions
    print("\n📊 INITIAL CONDITIONS:")
    print(f"State Vector X(0): {state.get_state_vector()}")
    print(f"Baseline Vector B(0): {state.get_baseline_vector()}")
    print(f"Wave Amplitude W(0): {state.get_wave_amplitude()}")
    print(f"Lyapunov V: {state.calculate_lyapunov_function():.2f}")
    print(f"Stable: {state.check_stability()}")
    
    # Show initial mood
    mood_info = MoodMapper.get_mood_indicators(state)
    print(f"\n🎭 Initial Mood: {mood_info['composite']}")
    print(f"Active Triggers: {mood_info['triggers']}")
    print(f"Valence: {mood_info['valence']}, Arousal: {mood_info['arousal']}")
    
    # Test different event types
    test_events = [
        ("User Question", Event(
            type="question",
            intensity=0.3,
            complexity=0.2,
            urgency=0.1,
            novelty=0.4,
            success_probability=0.8,
            actual_reward=0.7
        )),
        
        ("Complex Problem", Event(
            type="complex_problem",
            intensity=0.8,
            complexity=0.9,
            urgency=0.4,
            novelty=0.6,
            uncertainty=0.7,
            time_pressure=0.5,
            actual_reward=0.3,
            success_probability=0.4
        )),
        
        ("Urgent Crisis", Event(
            type="crisis",
            intensity=0.95,
            complexity=0.6,
            urgency=0.95,
            uncertainty=0.8,
            time_pressure=0.9,
            actual_reward=0.2,
            success_probability=0.3
        )),
        
        ("Creative Task", Event(
            type="creative",
            intensity=0.5,
            complexity=0.4,
            novelty=0.9,
            emotional_content=0.7,
            social_interaction=0.6,
            actual_reward=0.8,
            success_probability=0.7
        )),
        
        ("Social Success", Event(
            type="social",
            intensity=0.6,
            emotional_content=0.9,
            social_interaction=0.95,
            trust_factor=0.8,
            actual_reward=0.9,
            success_probability=0.95
        ))
    ]
    
    print("\n" + "=" * 80)
    print("📝 EVENT PROCESSING WITH FULL DYNAMICS:")
    print("=" * 80)
    
    dt = 0.1  # Time step
    
    for event_name, event in test_events:
        print(f"\n🎯 EVENT: {event_name}")
        print(f"   Type: {event.type}")
        print(f"   Key params: I={event.intensity:.1f} C={event.complexity:.1f} U={event.urgency:.1f} N={event.novelty:.1f}")
        
        # Store pre-event state
        pre_state = state.get_state_vector().copy()
        pre_lyapunov = state.calculate_lyapunov_function()
        
        # Apply dynamics
        state.apply_dynamics(dt, event)
        
        # Get post-event state
        post_state = state.get_state_vector()
        post_lyapunov = state.calculate_lyapunov_function()
        delta_state = post_state - pre_state
        
        # Show changes
        print(f"\n   📈 State Changes (ΔX):")
        hormones = ["Dopamine", "Cortisol", "Adrenaline", "Serotonin", "Oxytocin"]
        for i, hormone in enumerate(hormones):
            print(f"      {hormone:12s}: {pre_state[i]:6.1f} → {post_state[i]:6.1f} (Δ={delta_state[i]:+6.2f})")
        
        # Show effective hormones
        effective = state.get_effective_hormones()
        print(f"\n   ⚡ Effective Levels:")
        print(f"      Dopamine (eff): {effective['dopamine_effective']:.1f}")
        print(f"      Cortisol (eff): {effective['cortisol_effective']:.1f}")
        
        # Show mood
        mood_info = MoodMapper.get_mood_indicators(state)
        print(f"\n   🎭 Mood: {mood_info['composite']}")
        print(f"   Triggers: {mood_info['triggers']}")
        print(f"   Behavior: Planning={mood_info['behavior']['planning_depth']:.1f}, "
              f"Risk={mood_info['behavior']['risk_tolerance']:.2f}, "
              f"Speed={mood_info['behavior']['processing_speed']:.2f}")
        
        # Show stability
        print(f"\n   📊 Stability Metrics:")
        print(f"      Lyapunov: {pre_lyapunov:.2f} → {post_lyapunov:.2f}")
        print(f"      Stable: {state.check_stability()}")
        print(f"      Adrenaline Pool: {state.adrenaline_pool:.1f}")
        
        # Show prompt injection
        prompt = MoodMapper.create_prompt_injection(state)
        print(f"\n   💬 AI Prompt Injection:")
        print(f"      {prompt[:150]}...")
        
        print("-" * 80)
    
    # Test homeostasis over time
    print("\n" + "=" * 80)
    print("⏱️  TESTING HOMEOSTASIS (20 steps):")
    print("=" * 80)
    
    print("\nLetting system return to baseline without events...")
    for i in range(20):
        state.apply_dynamics(dt)
        
        if i % 5 == 0:
            levels = state.get_state_vector()
            baselines = state.get_baseline_vector()
            mood = MoodMapper.get_composite_mood(state)
            print(f"\nStep {i+1}:")
            print(f"   Levels: D={levels[0]:.1f} C={levels[1]:.1f} A={levels[2]:.1f} S={levels[3]:.1f} O={levels[4]:.1f}")
            print(f"   Baselines: D={baselines[0]:.1f} C={baselines[1]:.1f}")
            print(f"   Mood: {mood}")
    
    # Test baseline adaptation
    print("\n" + "=" * 80)
    print("🔄 TESTING BASELINE ADAPTATION:")
    print("=" * 80)
    
    print("\nSimulating repeated high-stress events...")
    for i in range(10):
        event = Event(
            type="stress",
            intensity=0.7,
            complexity=0.8,
            urgency=0.6,
            actual_reward=0.4
        )
        state.apply_dynamics(dt, event)
    
    print(f"After stress series:")
    print(f"   Cortisol baseline: {state.cortisol_baseline:.1f}")
    print(f"   Dopamine baseline: {state.dopamine_baseline:.1f}")
    
    # Test dopamine ceiling mechanism
    print("\n" + "=" * 80)
    print("🚫 TESTING DOPAMINE CEILING MECHANISM:")
    print("=" * 80)
    
    print("\nForcing dopamine baseline toward ceiling...")
    state.dopamine_baseline = 48
    for i in range(5):
        event = Event(
            type="reward",
            novelty=0.9,
            success_probability=0.95,
            actual_reward=0.95
        )
        state.apply_dynamics(dt, event)
        print(f"Step {i+1}: D_baseline={state.dopamine_baseline:.1f}, Cortisol={state.cortisol:.1f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL SYSTEM STATE:")
    print("=" * 80)
    
    final_mood = MoodMapper.get_mood_indicators(state)
    print(f"Composite Mood: {final_mood['composite']}")
    print(f"Active Capabilities: {final_mood['triggers']}")
    print(f"System Stable: {state.check_stability()}")
    print(f"Final Lyapunov: {state.calculate_lyapunov_function():.2f}")
    
    # Show interaction matrix
    print("\n📐 Interaction Matrix (J):")
    J = state.calculate_interaction_matrix()
    print(J)
    eigenvalues = np.linalg.eigvals(J)
    print(f"Max Eigenvalue: {np.max(np.abs(eigenvalues)):.4f} (must be < 1 for stability)")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_mathematical_framework()
