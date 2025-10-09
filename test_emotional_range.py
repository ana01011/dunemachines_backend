#!/usr/bin/env python3
"""
Test the full emotional range of the neurochemical system
"""

import sys
import os
sys.path.insert(0, '/root/openhermes_backend')

from app.neurochemistry.core.state_v2 import NeurochemicalState, Event
from app.neurochemistry.core.dimensional_emergence import DimensionalEmergence
from app.neurochemistry.integrated_system import NeurochemicalAgent

def test_extreme_states():
    """Test that we can reach extreme emotional positions"""
    
    print("=" * 70)
    print("🧬 TESTING FULL EMOTIONAL RANGE")
    print("=" * 70)
    
    agent = NeurochemicalAgent("test_user")
    
    # Test 1: Create high stress state (should map to "anger/panic")
    print("\n🔴 TEST 1: Extreme Stress")
    print("Applying multiple high-stress events...")
    
    for i in range(5):
        event = Event(
            type="crisis",
            urgency=0.95,
            complexity=0.9,
            uncertainty=0.8,
            threat_level=0.8,
            emotional_content=0.8
        )
        agent.state.apply_dynamics(0.1, event)
    
    pos1 = DimensionalEmergence.hormones_to_position(agent.state)
    print(f"Position: {pos1.to_vector()}")
    print(f"Hormones: D={agent.state.dopamine:.0f} C={agent.state.cortisol:.0f} "
          f"A={agent.state.adrenaline:.0f} S={agent.state.serotonin:.0f}")
    print(f"Should be: Negative valence, High arousal")
    
    # Test 2: Create joy state
    print("\n🟢 TEST 2: Extreme Joy")
    agent = NeurochemicalAgent("test_user2")  # Fresh agent
    
    for i in range(5):
        agent.state.start_task(f"task{i}", 0.3)
        agent.state.complete_task(0.95)  # Much better than expected
        agent.state.apply_dynamics(0.1)
    
    pos2 = DimensionalEmergence.hormones_to_position(agent.state)
    print(f"Position: {pos2.to_vector()}")
    print(f"Hormones: D={agent.state.dopamine:.0f} C={agent.state.cortisol:.0f} "
          f"A={agent.state.adrenaline:.0f} S={agent.state.serotonin:.0f}")
    print(f"Should be: Positive valence, Moderate arousal")
    
    # Test 3: Create sad state
    print("\n🔵 TEST 3: Deep Sadness")
    agent = NeurochemicalAgent("test_user3")
    
    for i in range(5):
        agent.state.start_task(f"task{i}", 0.2)
        agent.state.complete_task(0.1)  # Much worse than expected
        agent.state.apply_dynamics(0.1)
    
    pos3 = DimensionalEmergence.hormones_to_position(agent.state)
    print(f"Position: {pos3.to_vector()}")
    print(f"Hormones: D={agent.state.dopamine:.0f} C={agent.state.cortisol:.0f} "
          f"A={agent.state.adrenaline:.0f} S={agent.state.serotonin:.0f}")
    print(f"Should be: Negative valence, Low arousal")
    
    # Test 4: Manually set extreme values
    print("\n⚡ TEST 4: Manual Extreme Positions")
    
    test_states = [
        ("Extreme Anger", {"dopamine": 20, "cortisol": 85, "adrenaline": 80, 
                           "serotonin": 20, "oxytocin": 15}),
        ("Extreme Joy", {"dopamine": 90, "cortisol": 15, "adrenaline": 50,
                        "serotonin": 90, "oxytocin": 85}),
        ("Extreme Fear", {"dopamine": 15, "cortisol": 90, "adrenaline": 95,
                         "serotonin": 25, "oxytocin": 30}),
        ("Extreme Sadness", {"dopamine": 10, "cortisol": 60, "adrenaline": 10,
                            "serotonin": 20, "oxytocin": 25})
    ]
    
    for name, hormones in test_states:
        agent = NeurochemicalAgent("test")
        for h, v in hormones.items():
            setattr(agent.state, h, v)
        
        pos = DimensionalEmergence.hormones_to_position(agent.state)
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        
        print(f"\n{name}:")
        print(f"  Position: {pos.to_vector()}")
        print(f"  Behaviors: Speed={behaviors['response_speed']:.2f}, "
              f"Directness={behaviors['directness']:.2f}, "
              f"Empathy={behaviors['empathy']:.2f}")

def test_message_processing():
    """Test how messages affect position"""
    
    print("\n" + "=" * 70)
    print("📨 TESTING MESSAGE → POSITION MAPPING")
    print("=" * 70)
    
    agent = NeurochemicalAgent("msg_test")
    
    test_messages = [
        ("Calm greeting", "Hello, how are you?"),
        ("Urgent crisis", "EMERGENCY!!! SYSTEM FAILURE! HELP NOW!!!"),
        ("Deep frustration", "I hate this! Nothing ever works! This is terrible!"),
        ("Pure joy", "This is amazing! Best thing ever! Thank you!!!"),
        ("Sadness", "I give up. Nothing matters. I'm so tired...")
    ]
    
    for name, msg in test_messages:
        # Reset to baseline
        agent.state = NeurochemicalState()
        
        # Process message
        result = agent.process_message(msg)
        
        print(f"\n{name}: '{msg[:40]}...'")
        print(f"  Position: {result['position']}")
        print(f"  V={result['coordinates']['valence']:+.2f} "
              f"A={result['coordinates']['arousal']:.2f} "
              f"D={result['coordinates']['dominance']:.2f}")

def show_behavioral_differences():
    """Show how different positions create different AI responses"""
    
    print("\n" + "=" * 70)
    print("🤖 BEHAVIORAL DIFFERENCES BY POSITION")
    print("=" * 70)
    
    from app.neurochemistry.core.dimensional_emergence import DimensionalPosition, ResponseGenerator
    
    positions = [
        (DimensionalPosition(-0.8, 0.9, 0.8), "Extreme Negative+Aroused+Dominant"),
        (DimensionalPosition(0.8, 0.8, 0.8), "Extreme Positive+Aroused+Dominant"),
        (DimensionalPosition(-0.7, 0.2, 0.2), "Extreme Negative+Calm+Submissive"),
        (DimensionalPosition(0.0, 0.5, 0.5), "Perfect Neutral")
    ]
    
    user_msg = "Can you help me fix this error?"
    
    for pos, desc in positions:
        template = ResponseGenerator.generate_response_template(pos, user_msg)
        
        print(f"\n📍 {desc}")
        print(f"   Position: {pos.to_vector()}")
        print(f"   Constraints: {template['constraints'][:3]}")
        
        if template.get('example_opening'):
            print(f"   Opening style: {template['example_opening']}")
            print(f"   Tone: {template.get('example_tone', 'neutral')}")

if __name__ == "__main__":
    test_extreme_states()
    test_message_processing()
    show_behavioral_differences()
