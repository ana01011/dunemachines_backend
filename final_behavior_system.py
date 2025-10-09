#!/usr/bin/env python3
"""
Final refined behavior detection system
More nuanced position → behavior mapping
"""

import sys
sys.path.insert(0, '/root/openhermes_backend')

from app.neurochemistry.core.extreme_state import ExtremeNeurochemicalState
from app.neurochemistry.core.dimensional_emergence import DimensionalEmergence
import numpy as np

def get_behavior_from_position(pos):
    """
    More nuanced behavior detection
    Based on distance from prototypical emotional positions
    """
    
    v, a, d = pos.valence, pos.arousal, pos.dominance
    
    # Define prototypical positions for each behavior
    prototypes = {
        'ANGRY': {'v': -0.5, 'a': 0.7, 'd': 0.6},      # Negative, aroused, dominant
        'FEARFUL': {'v': -0.5, 'a': 0.8, 'd': 0.3},    # Negative, aroused, submissive
        'SAD': {'v': -0.4, 'a': 0.2, 'd': 0.3},        # Negative, calm, submissive
        'JOYFUL': {'v': 0.7, 'a': 0.6, 'd': 0.7},      # Positive, aroused, dominant
        'CONTENT': {'v': 0.5, 'a': 0.3, 'd': 0.5},     # Positive, calm, balanced
        'EXCITED': {'v': 0.6, 'a': 0.8, 'd': 0.6},     # Positive, very aroused
        'ANXIOUS': {'v': -0.3, 'a': 0.7, 'd': 0.4},    # Negative, aroused, uncertain
        'NEUTRAL': {'v': 0.0, 'a': 0.4, 'd': 0.5}      # Balanced everything
    }
    
    # Calculate distance to each prototype
    distances = {}
    for name, proto in prototypes.items():
        dist = np.sqrt((v - proto['v'])**2 + (a - proto['a'])**2 + (d - proto['d'])**2)
        distances[name] = dist
    
    # Find closest prototype
    closest = min(distances, key=distances.get)
    
    # Define behaviors for each state
    behaviors = {
        'ANGRY': {
            'description': 'Forceful, direct, impatient',
            'example': '"This is unacceptable. Fix it immediately."',
            'traits': 'Short sentences, no pleasantries, commanding tone'
        },
        'FEARFUL': {
            'description': 'Urgent, help-seeking, uncertain',
            'example': '"This is really bad! I need help! What should I do?"',
            'traits': 'Questions, exclamations, seeking reassurance'
        },
        'SAD': {
            'description': 'Slow, minimal, withdrawn',
            'example': '"I understand... Here\'s what you need..."',
            'traits': 'Short responses, low energy, minimal elaboration'
        },
        'JOYFUL': {
            'description': 'Enthusiastic, creative, expansive',
            'example': '"Wonderful! Let me show you several exciting approaches!"',
            'traits': 'Exclamations, detailed explanations, multiple ideas'
        },
        'CONTENT': {
            'description': 'Calm, patient, thorough',
            'example': '"I\'d be happy to help. Let me explain this step by step..."',
            'traits': 'Measured pace, complete explanations, friendly tone'
        },
        'EXCITED': {
            'description': 'Very energetic, rapid, intense',
            'example': '"Oh wow! This is amazing! Quick, let me show you!"',
            'traits': 'Rapid delivery, many exclamations, jumping between ideas'
        },
        'ANXIOUS': {
            'description': 'Worried, cautious, overthinking',
            'example': '"Hmm, this could be problematic... We should be careful..."',
            'traits': 'Hedging language, considering risks, uncertain'
        },
        'NEUTRAL': {
            'description': 'Balanced, professional, measured',
            'example': '"I understand your request. Here\'s my analysis..."',
            'traits': 'Standard professional tone, systematic approach'
        }
    }
    
    return closest, behaviors[closest], distances[closest]

def test_complete_system():
    """Test the complete emotional range with nuanced detection"""
    
    print("=" * 70)
    print("🎯 COMPLETE EMOTIONAL POSITION SYSTEM")
    print("=" * 70)
    
    # Test with specific hormone configurations
    test_cases = [
        ("Rage", [25, 80, 75, 20, 15]),     # More extreme anger
        ("Fear", [15, 85, 85, 25, 25]),     # Clear fear
        ("Deep Joy", [95, 10, 45, 95, 80]), # Extreme joy
        ("Sadness", [15, 55, 10, 25, 30]),  # Clear sadness
        ("Excitement", [75, 40, 70, 75, 60]), # High energy positive
        ("Anxiety", [35, 65, 60, 35, 35]),  # Worried state
        ("Neutral", [50, 30, 20, 60, 40]),  # Baseline
    ]
    
    for name, hormones in test_cases:
        state = ExtremeNeurochemicalState()
        state.dopamine = hormones[0]
        state.cortisol = hormones[1]
        state.adrenaline = hormones[2]
        state.serotonin = hormones[3]
        state.oxytocin = hormones[4]
        
        pos = DimensionalEmergence.hormones_to_position(state)
        behavior_type, behavior_info, distance = get_behavior_from_position(pos)
        
        print(f"\n{'─'*60}")
        print(f"🎭 {name}")
        print(f"   Hormones: D={hormones[0]:3} C={hormones[1]:3} A={hormones[2]:3} "
              f"S={hormones[3]:3} O={hormones[4]:3}")
        
        print(f"\n   3D Position: {pos.to_vector()}")
        print(f"   • Valence:   {pos.valence:+.2f} {'←negative  positive→':^20}")
        print(f"   • Arousal:   {pos.arousal:.2f}  {'calm← →activated':^20}")
        print(f"   • Dominance: {pos.dominance:.2f}  {'←submissive  dominant→':^20}")
        
        print(f"\n   Detected State: {behavior_type} (distance: {distance:.2f})")
        print(f"   Behavior: {behavior_info['description']}")
        print(f"   Example: {behavior_info['example']}")
        print(f"   Traits: {behavior_info['traits']}")

def show_3d_map():
    """Show where different emotions exist in 3D space"""
    
    print("\n" + "=" * 70)
    print("📊 EMOTIONAL MAP IN 3D SPACE")
    print("=" * 70)
    
    print("""
    High Arousal (1.0)
           ↑
      FEAR | ANGER
           |
    ───────┼──────→ Positive Valence (1.0)
           |
      SAD  | JOY
           ↓
    Low Arousal (0.0)
    
    With Dominance as depth:
    • Front (high dominance): ANGER, JOY
    • Back (low dominance): FEAR, SAD
    
    The AI's behavior emerges from its position in this space.
    """)

def demonstrate_prompt_injection():
    """Show how the AI would actually use these positions"""
    
    print("\n" + "=" * 70)
    print("💉 PROMPT INJECTION EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("Angry", "V-0.50A0.70D0.60", "Why is this code not working?"),
        ("Joyful", "V+0.70A0.60D0.70", "Can you teach me something new?"),
        ("Sad", "V-0.40A0.20D0.30", "I keep failing at this..."),
        ("Neutral", "V+0.00A0.40D0.50", "Please explain recursion."),
    ]
    
    for emotion, vector, message in examples:
        print(f"\n{emotion} State:")
        print(f"  User: {message}")
        print(f"  AI receives: [{vector}] {message}")
        print(f"  AI behavior emerges from position {vector}")
        
        if "V-0.50" in vector:
            print("  → Response: 'The issue is obvious. Line 42. Fix the type.'")
        elif "V+0.70" in vector:
            print("  → Response: 'Absolutely! I'm excited to share this with you!'")
        elif "V-0.40" in vector and "A0.20" in vector:
            print("  → Response: 'I see... Let me give you the basics...'")
        else:
            print("  → Response: 'I'll explain this systematically for you.'")

if __name__ == "__main__":
    test_complete_system()
    show_3d_map()
    demonstrate_prompt_injection()
