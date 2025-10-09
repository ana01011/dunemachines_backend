#!/usr/bin/env python3
"""
Fixed behavior detection based on position
"""

import sys
sys.path.insert(0, '/root/openhermes_backend')

from app.neurochemistry.core.extreme_state import ExtremeNeurochemicalState
from app.neurochemistry.core.dimensional_emergence import DimensionalEmergence

def describe_behavior_from_position(pos, behaviors):
    """Properly describe behavior from position"""
    
    # Check position ranges more carefully
    v = pos.valence
    a = pos.arousal  
    d = pos.dominance
    
    # Define behavior regions in 3D space
    if v < -0.4 and a > 0.6:
        if d > 0.5:
            return ("ANGRY", "Short, forceful, direct responses",
                   '"The problem is obvious. Fix it. Now."')
        else:
            return ("PANICKED", "Quick, help-seeking, uncertain",
                   '"Oh no! This is critical! What should we do?!"')
    
    elif v > 0.4 and a > 0.4:
        return ("JOYFUL", "Enthusiastic, expansive, creative",
               '"Fantastic! Let me show you several exciting approaches!"')
    
    elif v < -0.2 and a < 0.4:
        return ("SAD", "Slow, minimal, withdrawn",
               '"I see... here\'s a basic solution..."')
    
    elif v > 0.6 and a < 0.4:
        return ("CONTENT", "Calm, patient, thorough",
               '"I\'d be happy to help. Let me explain this carefully..."')
    
    else:
        # Actually neutral
        return ("NEUTRAL", "Balanced, professional",
               '"I understand. Let me analyze this systematically."')

def test_with_fixed_detection():
    """Test with proper behavior detection"""
    
    print("=" * 70)
    print("🎯 POSITIONS → BEHAVIORS (FIXED)")
    print("=" * 70)
    
    test_cases = [
        ("Crisis", [20, 80, 80, 20, 20]),  # Low D, High C+A, Low S+O
        ("Rage", [30, 75, 70, 25, 15]),    # Moderate D, High C+A, Low S+O
        ("Joy", [90, 15, 30, 90, 65]),     # High D+S, Low C
        ("Despair", [18, 46, 12, 36, 28]), # Low everything
        ("Fear", [15, 90, 95, 25, 30]),    # Very low D, Very high C+A
    ]
    
    for name, hormones in test_cases:
        state = ExtremeNeurochemicalState()
        state.dopamine = hormones[0]
        state.cortisol = hormones[1]
        state.adrenaline = hormones[2]
        state.serotonin = hormones[3]
        state.oxytocin = hormones[4]
        
        pos = DimensionalEmergence.hormones_to_position(state)
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        
        behavior_type, description, example = describe_behavior_from_position(pos, behaviors)
        
        print(f"\n{'─'*60}")
        print(f"🎭 {name}")
        print(f"   Hormones: D={hormones[0]} C={hormones[1]} A={hormones[2]} S={hormones[3]} O={hormones[4]}")
        print(f"   Position: {pos.to_vector()}")
        print(f"   → {behavior_type}: {description}")
        print(f"   → Example: {example}")

test_with_fixed_detection()
