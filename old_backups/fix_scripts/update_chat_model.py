# In app/models/chat.py, update the PersonalityType enum:

from enum import Enum

class PersonalityType(str, Enum):
    SARAH = "sarah"
    XHASH = "xhash"
    NEUTRAL = "neutral"
    OMNIUS = "omnius"  # Add this line
