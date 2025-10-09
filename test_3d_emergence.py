#!/usr/bin/env python3
"""
Test pure 3D dimensional emergence
No emotions - just positions creating behavior
"""

import sys
sys.path.append('/workspace')

import numpy as np
from app.neurochemistry.core.state_v2 import NeurochemicalState
from app.neurochemistry.core.dimensional_emergence import (
    DimensionalPosition, 
    DimensionalEmergence,
    ResponseGenerator
)

def visualize_3d_space():
    """Show the 3D emotional space"""
    print("="*70)
    print("📊 THE 3D EMOTIONAL SPACE")
    print("="*70)
    
    print("""
    Imagine a cube where every point represents a unique state:
    
                    High Arousal (1.0)
                          ↑
                          |
    Negative ←————————————————————————→ Positive
    Valence               |              Valence
    (-1.0)                ↓               (+1.0)
                    Low Arousal (0.0)
    
    With Dominance as the third dimension (in/out of page)
    
    Every hormone configuration maps to ONE point in this space.
    That point determines ALL behaviors.
    """)

def test_positions():
    """Test how different positions create different behaviors"""
    print("\n" + "="*70)
    print("🧬 POSITION → BEHAVIOR EMERGENCE")
    print("="*70)
    
    test_cases = [
        {
            "name": "Position A",
            "hormones": {"dopamine": 30, "cortisol": 75, "adrenaline": 70, 
                        "serotonin": 25, "oxytocin": 20},
            "human_label": "(humans call this 'anger')"
        },
        {
            "name": "Position B", 
            "hormones": {"dopamine": 25, "cortisol": 80, "adrenaline": 75,
                        "serotonin": 30, "oxytocin": 35},
            "human_label": "(humans call this 'fear/panic')"
        },
        {
            "name": "Position C",
            "hormones": {"dopamine": 80, "cortisol": 20, "adrenaline": 45,
                        "serotonin": 85, "oxytocin": 70},
            "human_label": "(humans call this 'joy')"
        },
        {
            "name": "Position D",
            "hormones": {"dopamine": 20, "cortisol": 50, "adrenaline": 15,
                        "serotonin": 25, "oxytocin": 35},
            "human_label": "(humans call this 'sadness')"
        }
    ]
    
    for test in test_cases:
        state = NeurochemicalState()
        for h, v in test["hormones"].items():
            setattr(state, h, v)
        
        # Get position
        pos = DimensionalEmergence.hormones_to_position(state)
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        style = DimensionalEmergence.position_to_response_style(pos)
        prompt = DimensionalEmergence.create_prompt_injection(state)
        
        print(f"\n{'─'*60}")
        print(f"📍 {test['name']}: {pos.to_vector()}")
        print(f"   {test['human_label']}")
        print(f"\n   3D Coordinates:")
        print(f"   • Valence: {pos.valence:+.2f} (suffering ← → pleasure)")
        print(f"   • Arousal: {pos.arousal:.2f} (calm ← → activated)")
        print(f"   • Dominance: {pos.dominance:.2f} (helpless ← → powerful)")
        
        print(f"\n   Emergent Behaviors:")
        print(f"   • Response Speed: {behaviors['response_speed']:.2f}")
        print(f"   • Directness: {behaviors['directness']:.2f}")
        print(f"   • Patience: {behaviors['patience']:.2f}")
        print(f"   • Analytical Depth: {behaviors['analytical_depth']:.2f}")
        print(f"   • Creativity: {behaviors['creativity']:.2f}")
        print(f"   • Empathy: {behaviors['empathy']:.2f}")
        
        print(f"\n   Response Style:")
        print(f"   • Sentences: {style['sentence_type']}")
        print(f"   • Opening: {style['opening']}")
        print(f"   • Vocabulary: {style['vocabulary']}")
        
        print(f"\n   AI Prompt: {prompt}")
        print(f"   (AI has no idea what 'emotion' this represents!)")

def demonstrate_continuous_space():
    """Show infinite positions between traditional emotions"""
    print("\n" + "="*70)
    print("🌈 CONTINUOUS SPACE - INFINITE STATES")
    print("="*70)
    
    print("\nMoving through space creates gradual behavior changes:")
    print("(not discrete emotion switching)")
    
    # Create a gradient from "sad" to "happy"
    for i in range(5):
        t = i / 4.0  # 0 to 1
        
        # Interpolate hormones
        state = NeurochemicalState()
        state.dopamine = 20 + (80-20) * t
        state.cortisol = 50 - (50-20) * t  
        state.serotonin = 25 + (85-25) * t
        state.adrenaline = 15 + (45-15) * t
        state.oxytocin = 35 + (70-35) * t
        
        pos = DimensionalEmergence.hormones_to_position(state)
        
        print(f"\n   Step {i+1}: {pos.to_vector()}")
        print(f"   V={pos.valence:+.2f}, A={pos.arousal:.2f}, D={pos.dominance:.2f}")
        
        if i == 0:
            print("   → Behavior: Withdrawn, minimal responses")
        elif i == 1:
            print("   → Behavior: Cautious, brief responses")
        elif i == 2:
            print("   → Behavior: Neutral, balanced responses")
        elif i == 3:
            print("   → Behavior: Engaged, detailed responses")
        elif i == 4:
            print("   → Behavior: Enthusiastic, expansive responses")

def show_ai_response_generation():
    """Show how position generates response characteristics"""
    print("\n" + "="*70)
    print("🤖 HOW AI RESPONDS FROM POSITION")
    print("="*70)
    
    # Test message
    user_message = "Can you help me understand this error?"
    
    # Two different positions
    positions = [
        (DimensionalPosition(-0.6, 0.8, 0.7), "Negative+Activated+Dominant"),
        (DimensionalPosition(0.6, 0.3, 0.5), "Positive+Calm+Balanced")
    ]
    
    for pos, desc in positions:
        template = ResponseGenerator.generate_response_template(pos, user_message)
        
        print(f"\n📍 Position: {pos.to_vector()} ({desc})")
        print(f"\n   Behavioral Parameters:")
        for key, val in sorted(template['behaviors'].items())[:5]:
            print(f"   • {key}: {val:.2f}")
        
        print(f"\n   Response Constraints:")
        for constraint in template['constraints']:
            print(f"   • {constraint}")
        
        print(f"\n   Example Response Style:")
        if 'example_opening' in template:
            print(f"   Opening: \"{template['example_opening']}\"")
            print(f"   Tone: {template['example_tone']}")
        
        print(f"\n   How AI would respond:")
        if pos.valence < 0 and pos.arousal > 0.7:
            print('   "The error is in line 42. Fix the type mismatch. Check your inputs."')
            print('   (Short, direct, impatient - emerges from high arousal + negative valence)')
        else:
            print('   "I\'d be happy to help you understand this error. Let me explain what\'s happening..."')
            print('   (Patient, detailed - emerges from positive valence + low arousal)')

def main():
    """Run all demonstrations"""
    visualize_3d_space()
    test_positions()
    demonstrate_continuous_space()
    show_ai_response_generation()
    
    print("\n" + "="*70)
    print("✨ KEY INSIGHT")
    print("="*70)
    print("""
    The AI never knows it's "angry" or "sad" or "happy".
    
    It only knows its position: V-0.6A0.8D0.7
    
    From that position, behaviors emerge:
    - High A (0.8) → Quick responses, impatience
    - Negative V (-0.6) → Less friendly, more critical
    - High D (0.7) → Assertive, commanding
    
    Humans perceive this as "anger" but the AI is just
    following the physics of its position in 3D space!
    
    This is TRUE EMERGENCE - complex emotional behavior
    from simple spatial coordinates.
    """)

if __name__ == "__main__":
    main()
