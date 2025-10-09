#!/usr/bin/env python3
"""
Test message sensitivity with fixed neurochemistry
"""

import sys
sys.path.insert(0, '/root/openhermes_backend')

from app.neurochemistry.core.state_v2_fixed import NeurochemicalState
from app.neurochemistry.core.dimensional_emergence import DimensionalEmergence

def analyze_message(message: str):
    """Better message analysis"""
    msg_lower = message.lower()
    
    # Urgency detection
    urgency = 0.2  # Base
    urgent_words = ['urgent', 'emergency', 'asap', 'now', 'immediately', 
                   'help', 'critical', 'failure', 'down', 'broken']
    for word in urgent_words:
        if word in msg_lower:
            urgency += 0.25
    
    urgency += message.count('!') * 0.1
    if message.isupper():
        urgency += 0.3
    urgency = min(1.0, urgency)
    
    # Emotional content detection
    emotional = 0.1
    neg_words = ['hate', 'angry', 'frustrated', 'terrible', 'awful', 
                 'stupid', 'never', 'nothing', 'give up', 'tired']
    pos_words = ['love', 'amazing', 'wonderful', 'thank', 'brilliant',
                 'excellent', 'happy', 'best', 'great']
    
    neg_count = sum(1 for w in neg_words if w in msg_lower)
    pos_count = sum(1 for w in pos_words if w in msg_lower)
    
    emotional += (neg_count + pos_count) * 0.25
    is_negative = neg_count > pos_count
    emotional = min(1.0, emotional)
    
    # Complexity
    complexity = 0.2
    tech_words = ['error', 'system', 'algorithm', 'database', 'code',
                  'function', 'bug', 'debug', 'server', 'production']
    for word in tech_words:
        if word in msg_lower:
            complexity += 0.15
    
    word_count = len(message.split())
    complexity += min(0.3, word_count / 100)
    complexity = min(1.0, complexity)
    
    return urgency, complexity, emotional, is_negative

def test_messages():
    """Test various messages"""
    
    print("=" * 70)
    print("🧬 MESSAGE PROCESSING WITH STRONGER RESPONSES")
    print("=" * 70)
    
    test_messages = [
        "Hello, how are you?",
        "URGENT!!! SYSTEM FAILURE! EVERYTHING IS DOWN!!!",
        "I hate this so much! Nothing works! This is terrible!",
        "This is absolutely amazing! Best thing ever! Thank you!",
        "I'm so tired. I give up. Nothing matters anymore.",
        "Can you help me understand this complex algorithm?",
    ]
    
    for msg in test_messages:
        # Create fresh state
        state = NeurochemicalState()
        
        # Analyze message
        urgency, complexity, emotional, is_negative = analyze_message(msg)
        
        print(f"\n📨 Message: '{msg[:50]}...'" if len(msg) > 50 else f"\n📨 Message: '{msg}'")
        print(f"   Analysis: Urgency={urgency:.2f}, Complex={complexity:.2f}, "
              f"Emotional={emotional:.2f}, Negative={is_negative}")
        
        # Process message
        state.process_message_event(urgency, complexity, emotional, is_negative)
        
        # Get position
        pos = DimensionalEmergence.hormones_to_position(state)
        
        print(f"   Hormones: D={state.dopamine:.0f} C={state.cortisol:.0f} "
              f"A={state.adrenaline:.0f} S={state.serotonin:.0f} O={state.oxytocin:.0f}")
        print(f"   Position: {pos.to_vector()}")
        print(f"   V={pos.valence:+.2f} A={pos.arousal:.2f} D={pos.dominance:.2f}")
        
        # Describe behavior
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        if pos.valence < -0.5 and pos.arousal > 0.7:
            print("   → AI behavior: Quick, direct, forceful (angry-like)")
        elif pos.valence < -0.3 and pos.arousal < 0.3:
            print("   → AI behavior: Slow, withdrawn, minimal (sad-like)")
        elif pos.valence > 0.5 and pos.arousal > 0.5:
            print("   → AI behavior: Enthusiastic, creative, talkative (joy-like)")
        else:
            print(f"   → AI behavior: Balanced (Speed={behaviors['response_speed']:.2f})")

if __name__ == "__main__":
    test_messages()
