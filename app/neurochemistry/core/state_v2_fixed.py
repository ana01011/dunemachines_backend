"""
Fixed neurochemical state with stronger event responses
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from app.neurochemistry.core.constants import *

@dataclass
class Event:
    """Event that affects neurochemistry"""
    type: str
    intensity: float = 0.5
    complexity: float = 0.5
    urgency: float = 0.5
    emotional_content: float = 0.5
    novelty: float = 0.5
    social_interaction: float = 0.5
    uncertainty: float = 0.5
    trust_factor: float = 0.5
    threat_level: float = 0.0
    success_probability: float = 0.5
    actual_reward: float = 0.5

@dataclass
class NeurochemicalState:
    """Neurochemical state with stronger responses"""
    
    # Current hormone levels
    dopamine: float = 50.0
    cortisol: float = 30.0
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Baselines
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    # History
    dopamine_spikes: List[float] = field(default_factory=list)
    recent_outcomes: List[float] = field(default_factory=list)
    adrenaline_pool: float = 100.0
    time: float = 0.0
    
    def apply_dynamics(self, dt: float, event: Optional[Event] = None):
        """Apply dynamics with STRONGER event responses"""
        
        if event:
            # MUCH STRONGER responses to events
            response_multiplier = 20.0  # Increased from implicit ~5
            
            # Urgency massively affects cortisol and adrenaline
            if event.urgency > 0.7:
                self.cortisol += event.urgency * 30
                self.adrenaline += event.urgency * 35
                
            # Threat increases cortisol, decreases serotonin
            if event.threat_level > 0.5:
                self.cortisol += event.threat_level * 25
                self.serotonin -= event.threat_level * 20
                self.oxytocin -= event.threat_level * 15
                
            # Emotional content affects oxytocin and serotonin
            if event.emotional_content > 0.7:
                if event.threat_level > 0.5:  # Negative emotion
                    self.serotonin -= event.emotional_content * 15
                    self.dopamine -= event.emotional_content * 10
                else:  # Positive emotion
                    self.oxytocin += event.emotional_content * 20
                    self.serotonin += event.emotional_content * 15
                    
            # Complexity increases cortisol
            if event.complexity > 0.5:
                self.cortisol += event.complexity * 15
                
            # Social interaction affects oxytocin
            self.oxytocin += event.social_interaction * 10
        
        # Homeostasis (but slower, so events have more impact)
        homeostasis_rate = 0.02  # Reduced from 0.05
        
        self.dopamine += (self.dopamine_baseline - self.dopamine) * homeostasis_rate * dt
        self.cortisol += (self.cortisol_baseline - self.cortisol) * homeostasis_rate * dt
        self.adrenaline += (self.adrenaline_baseline - self.adrenaline) * homeostasis_rate * 2 * dt
        self.serotonin += (self.serotonin_baseline - self.serotonin) * homeostasis_rate * dt
        self.oxytocin += (self.oxytocin_baseline - self.oxytocin) * homeostasis_rate * dt
        
        # Clamp values
        self.dopamine = np.clip(self.dopamine, 0, 100)
        self.cortisol = np.clip(self.cortisol, 0, 100)
        self.adrenaline = np.clip(self.adrenaline, 0, 100)
        self.serotonin = np.clip(self.serotonin, 0, 100)
        self.oxytocin = np.clip(self.oxytocin, 0, 100)
        
        self.time += dt
    
    def process_message_event(self, urgency: float, complexity: float, 
                            emotional_content: float, is_negative: bool = False):
        """Process a message as an event with strong effects"""
        
        # Create event with proper threat level for negative messages
        threat = 0.8 if is_negative else 0.0
        
        event = Event(
            type="message",
            urgency=urgency,
            complexity=complexity,
            emotional_content=emotional_content,
            threat_level=threat,
            intensity=max(urgency, emotional_content, complexity)
        )
        
        self.apply_dynamics(0.1, event)
    
    def get_state_vector(self) -> np.ndarray:
        return np.array([self.dopamine, self.cortisol, self.adrenaline, 
                        self.serotonin, self.oxytocin])
    
    def get_baseline_vector(self) -> np.ndarray:
        return np.array([self.dopamine_baseline, self.cortisol_baseline,
                        self.adrenaline_baseline, self.serotonin_baseline,
                        self.oxytocin_baseline])
