"""
Neurochemical State with strong event responses
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NeurochemicalState:
    """Core neurochemical state for a user"""
    
    # User identification
    user_id: str
    
    # Current hormone levels
    dopamine: float = 50.0
    cortisol: float = 30.0
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Baselines (what hormones return to)
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    def process_message_event(self, event):
        """Process a message event with strong responses"""
        # Strong responses based on event properties
        if event.urgency > 0.7:
            self.cortisol += event.urgency * 30
            self.adrenaline += event.urgency * 35
        
        if event.threat_level > 0.5:
            self.cortisol += event.threat_level * 25
            self.adrenaline += event.threat_level * 20
            self.dopamine -= event.threat_level * 10
        
        if event.emotional_content > 0:  # Positive
            self.dopamine += event.emotional_content * 20
            self.serotonin += event.emotional_content * 15
            self.oxytocin += event.emotional_content * 10
        elif event.emotional_content < 0:  # Negative
            self.dopamine += event.emotional_content * 15  # Decrease
            self.cortisol -= event.emotional_content * 20   # Increase
            self.serotonin += event.emotional_content * 10  # Decrease
        
        if event.complexity > 0.7:
            self.cortisol += event.complexity * 15
            self.dopamine += event.complexity * 10  # Challenge can be rewarding
        
        # Clamp values
        self.clamp_hormones()
    
    def apply_homeostasis(self, rate: float = 0.1):
        """Return hormones toward baseline"""
        self.dopamine += (self.dopamine_baseline - self.dopamine) * rate
        self.cortisol += (self.cortisol_baseline - self.cortisol) * rate
        self.adrenaline += (self.adrenaline_baseline - self.adrenaline) * rate
        self.serotonin += (self.serotonin_baseline - self.serotonin) * rate
        self.oxytocin += (self.oxytocin_baseline - self.oxytocin) * rate
        
        self.clamp_hormones()
    
    def clamp_hormones(self):
        """Keep hormones in valid range"""
        self.dopamine = max(0, min(100, self.dopamine))
        self.cortisol = max(0, min(100, self.cortisol))
        self.adrenaline = max(0, min(100, self.adrenaline))
        self.serotonin = max(0, min(100, self.serotonin))
        self.oxytocin = max(0, min(100, self.oxytocin))
