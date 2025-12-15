"""
Behavioral Prompt Generator
Converts 7D hormone vector directly to behavioral percentages for OMNIUS
Simple, clean, biologically grounded.
"""
 
from typing import Dict, Tuple
from dataclasses import dataclass
 
 
@dataclass
class BehavioralState:
    """
    7D Behavioral state mapped directly from hormones
    Each hormone IS a behavioral dimension
    """
    motivation: int      # Dopamine → Curiosity, drive, reward-seeking
    patience: int        # Serotonin → Calm, content, tolerant
    stress: int          # Cortisol → Alert, anxious, vigilant
    energy: int          # Adrenaline → Excited, urgent, action-ready
    warmth: int          # Oxytocin → Empathetic, trusting, bonding
    focus: int           # Norepinephrine → Attentive, concentrated
    joy: int             # Endorphins → Playful, pleasant, comfortable
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'motivation': self.motivation,
            'patience': self.patience,
            'stress': self.stress,
            'energy': self.energy,
            'warmth': self.warmth,
            'focus': self.focus,
            'joy': self.joy
        }
    
    def to_prompt_string(self) -> str:
        """
        Generate simple percentage string for OMNIUS prompt
        """
        return (
            f"motivation:{self.motivation}% "
            f"patience:{self.patience}% "
            f"stress:{self.stress}% "
            f"energy:{self.energy}% "
            f"warmth:{self.warmth}% "
            f"focus:{self.focus}% "
            f"joy:{self.joy}%"
        )
 
 
def hormones_to_behavior(hormones: Dict[str, float]) -> BehavioralState:
    """
    Convert 7D hormone dict to behavioral percentages
    
    Args:
        hormones: Dict with keys: dopamine, serotonin, cortisol, 
                  adrenaline, oxytocin, norepinephrine, endorphins
                  Values should be 0.0 to 1.0
    
    Returns:
        BehavioralState with percentage values (0-100)
    """
    # Direct mapping: hormone value (0-1) → percentage (0-100)
    return BehavioralState(
        motivation=int(hormones.get('dopamine', 0.5) * 100),
        patience=int(hormones.get('serotonin', 0.5) * 100),
        stress=int(hormones.get('cortisol', 0.3) * 100),
        energy=int(hormones.get('adrenaline', 0.3) * 100),
        warmth=int(hormones.get('oxytocin', 0.5) * 100),
        focus=int(hormones.get('norepinephrine', 0.5) * 100),
        joy=int(hormones.get('endorphins', 0.5) * 100)
    )
 
 
def get_behavior_prompt(hormones: Dict[str, float]) -> str:
    """
    Main function: Convert hormones to OMNIUS behavior prompt
    
    Args:
        hormones: 7D hormone dict (values 0-1)
        
    Returns:
        Simple behavior string for prompt injection
    """
    state = hormones_to_behavior(hormones)
    return state.to_prompt_string()
 
 
def get_behavior_context(hormones: Dict[str, float], user_emotion: str = None) -> str:
    """
    Generate full behavioral context for OMNIUS prompt
    
    Args:
        hormones: 7D hormone dict
        user_emotion: Optional detected user emotion
        
    Returns:
        Context string to inject into OMNIUS prompt
    """
    state = hormones_to_behavior(hormones)
    
    context = f"[BEHAVIOR STATE: {state.to_prompt_string()}]"
    
    if user_emotion:
        context += f" [USER EMOTION: {user_emotion}]"
    
    return context
