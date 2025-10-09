"""
Standalone Neurochemistry System Test
Tests mathematical calculations and mood mapping
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import math
import random

# Hormone mood mappings
MOOD_MAPPINGS = {
    "dopamine": {
        (0, 10): "unmotivated",
        (11, 20): "neutral",
        (21, 30): "interested",
        (31, 40): "motivated", 
        (41, 50): "engaged",
        (51, 60): "excited",
        (61, 70): "enthusiastic",
        (71, 80): "driven",
        (81, 90): "euphoric",
        (91, 100): "manic"
    },
    "cortisol": {
        (0, 10): "relaxed",
        (11, 20): "calm",
        (21, 30): "alert",
        (31, 40): "attentive",
        (41, 50): "focused",
        (51, 60): "vigilant",
        (61, 70): "stressed",
        (71, 80): "anxious",
        (81, 90): "overwhelmed",
        (91, 100): "panicked"
    },
    "adrenaline": {
        (0, 10): "sluggish",
        (11, 20): "rested",
        (21, 30): "ready",
        (31, 40): "energized",
        (41, 50): "active",
        (51, 60): "urgent",
        (61, 70): "intense",
        (71, 80): "hyperactive",
        (81, 90): "frantic",
        (91, 100): "overdrive"
    },
    "serotonin": {
        (0, 10): "depressed",
        (11, 20): "low",
        (21, 30): "uncertain",
        (31, 40): "neutral",
        (41, 50): "content",
        (51, 60): "confident",
        (61, 70): "happy",
        (71, 80): "optimistic",
        (81, 90): "joyful",
        (91, 100): "blissful"
    },
    "oxytocin": {
        (0, 10): "detached",
        (11, 20): "distant",
        (21, 30): "professional",
        (31, 40): "friendly",
        (41, 50): "warm",
        (51, 60): "caring",
        (61, 70): "empathetic",
        (71, 80): "compassionate",
        (81, 90): "deeply connected",
        (91, 100): "boundlessly loving"
    }
}

@dataclass
class Event:
    type: str
    intensity: float = 0.5
    complexity: float = 0.5
    urgency: float = 0.5
    emotional_content: float = 0.5
    novelty: float = 0.5
    success_probability: float = 0.5

@dataclass
class NeurochemicalState:
    dopamine: float = 50.0
    cortisol: float = 30.0
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Dynamic baselines
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    def get_mood(self, hormone: str, level: float) -> str:
        """Get mood descriptor for hormone level"""
        for (min_val, max_val), mood in MOOD_MAPPINGS[hormone].items():
            if min_val <= level <= max_val:
                return mood
        return "unknown"
    
    def get_composite_mood(self) -> Dict[str, any]:
        """Get composite mood from all hormones"""
        moods = {
            "dopamine_mood": self.get_mood("dopamine", self.dopamine),
            "cortisol_mood": self.get_mood("cortisol", self.cortisol),
            "adrenaline_mood": self.get_mood("adrenaline", self.adrenaline),
            "serotonin_mood": self.get_mood("serotonin", self.serotonin),
            "oxytocin_mood": self.get_mood("oxytocin", self.oxytocin)
        }
        
        # Create composite description
        composite = f"{moods['cortisol_mood']}+{moods['dopamine_mood']}+{moods['serotonin_mood']}"
        
        # Determine triggers
        triggers = []
        if self.cortisol > 40 and self.dopamine > 30:
            triggers.append("deep_research")
        if self.cortisol > 60 and self.adrenaline > 50:
            triggers.append("urgent_problem_solving")
        if self.dopamine > 50 and self.serotonin > 50:
            triggers.append("creative_brainstorming")
        if 30 <= self.cortisol <= 50 and self.adrenaline < 40:
            triggers.append("methodical_analysis")
            
        return {
            "moods": moods,
            "composite": composite,
            "triggers": triggers,
            "levels": {
                "dopamine": round(self.dopamine, 2),
                "cortisol": round(self.cortisol, 2),
                "adrenaline": round(self.adrenaline, 2),
                "serotonin": round(self.serotonin, 2),
                "oxytocin": round(self.oxytocin, 2)
            }
        }
    
    def process_event(self, event: Event):
        """Process event and update hormone levels"""
        print(f"\n📊 Processing {event.type} event:")
        print(f"   Intensity: {event.intensity:.2f}")
        print(f"   Complexity: {event.complexity:.2f}")
        print(f"   Urgency: {event.urgency:.2f}")
        
        # Calculate hormone changes
        # Dopamine: reward and novelty
        dopamine_change = (event.novelty * 10) + (event.success_probability * 5)
        
        # Cortisol: stress and complexity
        cortisol_change = (event.complexity * 8) + (event.urgency * 12)
        
        # Adrenaline: urgency
        adrenaline_change = event.urgency * 15
        
        # Serotonin: success and low stress
        serotonin_change = (event.success_probability * 10) - (event.complexity * 5)
        
        # Oxytocin: emotional content
        oxytocin_change = event.emotional_content * 10
        
        # Apply changes with decay toward baseline
        self.dopamine = self.dopamine * 0.9 + dopamine_change
        self.cortisol = self.cortisol * 0.9 + cortisol_change  
        self.adrenaline = self.adrenaline * 0.85 + adrenaline_change  # Faster decay
        self.serotonin = self.serotonin * 0.95 + serotonin_change
        self.oxytocin = self.oxytocin * 0.9 + oxytocin_change
        
        # Apply homeostasis (pull toward baseline)
        self.apply_homeostasis()
        
        # Clamp values
        self.clamp_values()
        
    def apply_homeostasis(self):
        """Apply homeostatic pressure toward baselines"""
        homeostasis_rate = 0.05
        
        self.dopamine += (self.dopamine_baseline - self.dopamine) * homeostasis_rate
        self.cortisol += (self.cortisol_baseline - self.cortisol) * homeostasis_rate
        self.adrenaline += (self.adrenaline_baseline - self.adrenaline) * homeostasis_rate * 2  # Faster return
        self.serotonin += (self.serotonin_baseline - self.serotonin) * homeostasis_rate
        self.oxytocin += (self.oxytocin_baseline - self.oxytocin) * homeostasis_rate
        
    def clamp_values(self):
        """Clamp hormone values between 0 and 100"""
        self.dopamine = max(0, min(100, self.dopamine))
        self.cortisol = max(0, min(100, self.cortisol))
        self.adrenaline = max(0, min(100, self.adrenaline))
        self.serotonin = max(0, min(100, self.serotonin))
        self.oxytocin = max(0, min(100, self.oxytocin))

def test_neurochemistry():
    """Test the neurochemistry system"""
    print("=" * 60)
    print("🧬 NEUROCHEMISTRY SYSTEM TEST")
    print("=" * 60)
    
    # Initialize state
    state = NeurochemicalState()
    
    # Show initial state
    print("\n📍 Initial State:")
    mood = state.get_composite_mood()
    print(json.dumps(mood, indent=2))
    
    # Test various events
    events = [
        Event("user_question", intensity=0.3, complexity=0.2, urgency=0.1),
        Event("complex_coding", intensity=0.7, complexity=0.9, urgency=0.4),
        Event("urgent_error", intensity=0.9, complexity=0.6, urgency=0.95),
        Event("successful_completion", intensity=0.6, success_probability=0.9, emotional_content=0.7),
        Event("creative_task", intensity=0.5, novelty=0.8, complexity=0.4),
    ]
    
    for event in events:
        state.process_event(event)
        mood = state.get_composite_mood()
        print(f"\n🎭 After {event.type}:")
        print(f"   Composite Mood: {mood['composite']}")
        print(f"   Triggers: {mood['triggers']}")
        print(f"   Levels: D:{mood['levels']['dopamine']:.1f} C:{mood['levels']['cortisol']:.1f} " +
              f"A:{mood['levels']['adrenaline']:.1f} S:{mood['levels']['serotonin']:.1f} " +
              f"O:{mood['levels']['oxytocin']:.1f}")
    
    # Test homeostasis over time
    print("\n\n⏱️  Testing Homeostasis (10 iterations):")
    print("-" * 40)
    for i in range(10):
        state.apply_homeostasis()
        state.clamp_values()
        if i % 2 == 0:
            mood = state.get_composite_mood()
            print(f"Iteration {i+1}: {mood['composite']}")
            print(f"   D:{mood['levels']['dopamine']:.1f} C:{mood['levels']['cortisol']:.1f} " +
                  f"A:{mood['levels']['adrenaline']:.1f}")

if __name__ == "__main__":
    test_neurochemistry()
