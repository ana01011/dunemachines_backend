"""
Event class for neurochemical system
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Event:
    """Represents an event that affects neurochemistry"""
    
    type: str
    intensity: float = 0.5
    complexity: float = 0.5
    urgency: float = 0.2
    emotional_content: float = 0.0  # -1 to +1
    threat_level: float = 0.1
    timestamp: Optional[float] = None
