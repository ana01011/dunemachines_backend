#!/usr/bin/env python3
"""
Test that we can reach all extreme emotional positions
"""

import sys
sys.path.insert(0, '/root/openhermes_backend')

from app.neurochemistry.core.extreme_state import ExtremeNeurochemicalState
from app.neurochemistry.core.dimensional_emergence import DimensionalEmergence

def test_extreme_messages():
    """Test extreme emotional positions from messages"""
    
    print("=" * 70)
    print("🌟 EXTREME EMOTIONAL POSITIONS TEST")
    print("=" * 70)
    
    test_cases = [
        ("Neutral", "Hello, how are you today?"),
        ("Crisis", "URGENT!!! SYSTEM CRITICAL FAILURE!!! EVERYTHING IS DOWN!!!"),
        ("Rage", "I HATE this! This is TERRIBLE! Everything is AWFUL!!!"),
        ("Joy", "This is AMAZING! BEST thing ever! Thank you so much!!!"),
        ("Despair", "I give up. I'm so tired. Nothing matters anymore."),
        ("Terror", "HELP! I'm scared! This is terrifying! Please help me!"),
    ]
    
    for name, message in test_cases:
        state = ExtremeNeurochemicalState()
        
        print(f"\n{'─'*60}")
        print(f"🎭 {name}: '{message[:40]}...'" if len(message) > 40 else f"🎭 {name}: '{message}'")
        
        # Process message
        state.process_message(message)
        
        # Get position
        pos = DimensionalEmergence.hormones_to_position(state)
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        
        print(f"\n   Hormones:")
        print(f"   D={state.dopamine:3.0f} C={state.cortisol:3.0f} "
              f"A={state.adrenaline:3.0f} S={state.serotonin:3.0f} O={state.oxytocin:3.0f}")
        
        print(f"\n   3D Position: {pos.to_vector()}")
        print(f"   • Valence: {pos.valence:+.2f} (negative ← → positive)")
        print(f"   • Arousal: {pos.arousal:.2f} (calm ← → activated)")  
        print(f"   • Dominance: {pos.dominance:.2f} (submissive ← → dominant)")
        
        print(f"\n   Behavioral Profile:")
        print(f"   • Response Speed: {behaviors['response_speed']:.2f}")
        print(f"   • Directness: {behaviors['directness']:.2f}")
        print(f"   • Empathy: {behaviors['empathy']:.2f}")
        print(f"   • Patience: {behaviors['patience']:.2f}")
        print(f"   • Creativity: {behaviors['creativity']:.2f}")
        
        # Describe emergent behavior
        print(f"\n   Emergent AI Behavior:")
        if pos.valence < -0.5 and pos.arousal > 0.7 and pos.dominance > 0.5:
            print("   → ANGRY: Short, forceful, direct responses")
            print('   → Example: "The problem is obvious. Fix your configuration. Now."')
        elif pos.valence < -0.5 and pos.arousal > 0.7 and pos.dominance < 0.4:
            print("   → FEARFUL/PANICKED: Quick, help-seeking, uncertain")
            print('   → Example: "Oh no! This is bad! What should we do?!"')
        elif pos.valence > 0.5 and pos.arousal > 0.5:
            print("   → JOYFUL: Enthusiastic, expansive, creative")
            print('   → Example: "Fantastic! Let me show you several exciting approaches!"')
        elif pos.valence < -0.3 and pos.arousal < 0.3:
            print("   → SAD: Slow, minimal, withdrawn")
            print('   → Example: "I see... here\'s a basic solution..."')
        else:
            print("   → NEUTRAL: Balanced, professional")
            print('   → Example: "I understand. Let me analyze this systematically."')

def show_position_space():
    """Show the full range of achievable positions"""
    
    print("\n" + "=" * 70)
    print("📊 ACHIEVABLE POSITION SPACE")
    print("=" * 70)
    
    extreme_states = [
        ("Minimum (all 0)", [0, 0, 0, 0, 0]),
        ("Maximum (all 100)", [100, 100, 100, 100, 100]),
        ("Pure Negative", [10, 90, 90, 10, 10]),
        ("Pure Positive", [90, 10, 50, 90, 90]),
        ("Pure Calm", [50, 20, 10, 60, 50]),
        ("Pure Activated", [60, 80, 90, 40, 30]),
    ]
    
    for name, hormones in extreme_states:
        state = ExtremeNeurochemicalState()
        state.dopamine, state.cortisol, state.adrenaline, state.serotonin, state.oxytocin = hormones
        
        pos = DimensionalEmergence.hormones_to_position(state)
        
        print(f"\n{name}:")
        print(f"  Hormones: D={hormones[0]} C={hormones[1]} A={hormones[2]} S={hormones[3]} O={hormones[4]}")
        print(f"  Position: {pos.to_vector()}")

if __name__ == "__main__":
    test_extreme_messages()
    show_position_space()
