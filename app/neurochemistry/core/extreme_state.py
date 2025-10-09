"""
Neurochemical state with extreme responses for full emotional range
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class ExtremeNeurochemicalState:
    """State that can reach extreme positions"""
    
    # Current levels
    dopamine: float = 50.0
    cortisol: float = 30.0  
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Baselines (for reference)
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    def apply_extreme_event(self, event_type: str, intensity: float = 1.0):
        """Apply extreme changes based on event type"""
        
        if event_type == "extreme_crisis":
            # Massive stress response
            self.cortisol = min(100, self.cortisol + 50 * intensity)
            self.adrenaline = min(100, self.adrenaline + 60 * intensity)
            self.dopamine = max(0, self.dopamine - 30 * intensity)
            self.serotonin = max(0, self.serotonin - 40 * intensity)
            self.oxytocin = max(0, self.oxytocin - 20 * intensity)
            
        elif event_type == "extreme_anger":
            # High cortisol + adrenaline, crashed serotonin
            self.cortisol = min(100, self.cortisol + 45 * intensity)
            self.adrenaline = min(100, self.adrenaline + 50 * intensity)
            self.serotonin = max(0, self.serotonin - 35 * intensity)
            self.oxytocin = max(0, self.oxytocin - 25 * intensity)
            self.dopamine = max(0, self.dopamine - 20 * intensity)
            
        elif event_type == "extreme_joy":
            # High dopamine + serotonin, low cortisol
            self.dopamine = min(100, self.dopamine + 40 * intensity)
            self.serotonin = min(100, self.serotonin + 30 * intensity)
            self.oxytocin = min(100, self.oxytocin + 25 * intensity)
            self.cortisol = max(0, self.cortisol - 15 * intensity)
            self.adrenaline = self.adrenaline + 10 * intensity  # Some excitement
            
        elif event_type == "extreme_sadness":
            # Crashed dopamine + serotonin
            self.dopamine = max(0, self.dopamine - 40 * intensity)
            self.serotonin = max(0, self.serotonin - 30 * intensity)
            self.cortisol = min(100, self.cortisol + 20 * intensity)
            self.adrenaline = max(0, self.adrenaline - 10 * intensity)
            self.oxytocin = max(0, self.oxytocin - 15 * intensity)
            
        elif event_type == "extreme_fear":
            # Maxed cortisol + adrenaline, low dominance hormones
            self.cortisol = min(100, self.cortisol + 60 * intensity)
            self.adrenaline = min(100, self.adrenaline + 70 * intensity)
            self.dopamine = max(0, self.dopamine - 35 * intensity)
            self.serotonin = max(0, self.serotonin - 25 * intensity)
            
        # Clamp all values
        self.dopamine = np.clip(self.dopamine, 0, 100)
        self.cortisol = np.clip(self.cortisol, 0, 100)
        self.adrenaline = np.clip(self.adrenaline, 0, 100)
        self.serotonin = np.clip(self.serotonin, 0, 100)
        self.oxytocin = np.clip(self.oxytocin, 0, 100)
    
    def process_message(self, message: str):
        """Process message with extreme responses"""
        msg_lower = message.lower()
        
        # Check for extreme patterns
        if any(word in msg_lower for word in ['emergency', 'urgent', 'critical', 'failure', 'down']):
            if '!!!' in message or message.isupper():
                self.apply_extreme_event("extreme_crisis", 1.0)
            else:
                self.apply_extreme_event("extreme_crisis", 0.6)
                
        elif any(word in msg_lower for word in ['hate', 'angry', 'furious', 'terrible', 'awful']):
            intensity = 0.7 + (message.count('!') * 0.1)
            self.apply_extreme_event("extreme_anger", min(1.0, intensity))
            
        elif any(word in msg_lower for word in ['amazing', 'wonderful', 'best', 'brilliant', 'thank']):
            intensity = 0.6 + (message.count('!') * 0.1)
            self.apply_extreme_event("extreme_joy", min(1.0, intensity))
            
        elif any(word in msg_lower for word in ['tired', 'give up', 'hopeless', 'nothing matters']):
            self.apply_extreme_event("extreme_sadness", 0.8)
            
        elif any(word in msg_lower for word in ['scared', 'terrified', 'help', 'afraid']):
            self.apply_extreme_event("extreme_fear", 0.8)
        
        # Default mild response for neutral messages
        else:
            pass  # Stay at current state
