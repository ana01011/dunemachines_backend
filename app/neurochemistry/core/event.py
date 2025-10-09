"""
Fixed Event class with all required attributes
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from enum import Enum
import hashlib


class EventType(str, Enum):
    """Types of events that can trigger neurochemical responses"""
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    USER_PRAISE = "user_praise"
    USER_CRITICISM = "user_criticism"
    HIGH_COMPLEXITY = "high_complexity"
    TIME_PRESSURE = "time_pressure"
    LEARNING = "learning"
    SOCIAL_INTERACTION = "social_interaction"
    ERROR_OCCURRED = "error_occurred"
    ROUTINE_TASK = "routine_task"
    CREATIVE_TASK = "creative_task"
    UNKNOWN = "unknown"


@dataclass
class Event:
    """Represents an event that triggers neurochemical responses"""
    type: EventType
    magnitude: float  # 0.0 to 1.0
    timestamp: datetime
    event_id: str
    
    # Required attributes for hormone calculations
    complexity: float = 0.5  # 0.0 to 1.0
    urgency: float = 0.0  # 0.0 to 1.0
    emotional_content: float = 0.0  # 0.0 to 1.0
    novelty: float = 0.0  # 0.0 to 1.0
    success_probability: float = 0.5  # 0.0 to 1.0
    
    # Context
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_message(cls, message: str, context: Dict[str, Any]) -> 'Event':
        """Create an event from a user message"""
        
        # Analyze message for event characteristics
        message_lower = message.lower()
        
        # Determine event type
        event_type = EventType.UNKNOWN
        magnitude = 0.5
        complexity = 0.5
        urgency = 0.0
        emotional_content = 0.0
        novelty = 0.3
        
        # Check for stress/urgency indicators
        if any(word in message_lower for word in ['urgent', 'emergency', 'asap', 'quickly', 'hurry', 'deadline', 'stressed']):
            event_type = EventType.TIME_PRESSURE
            urgency = 0.8
            magnitude = 0.7
            emotional_content = 0.6
        
        # Check for complexity
        if any(word in message_lower for word in ['complex', 'difficult', 'hard', 'complicated', 'advanced']):
            event_type = EventType.HIGH_COMPLEXITY
            complexity = 0.8
            magnitude = 0.7
            novelty = 0.6
        
        # Check for praise
        if any(word in message_lower for word in ['good', 'great', 'excellent', 'amazing', 'thank']):
            event_type = EventType.USER_PRAISE
            magnitude = 0.8
            emotional_content = 0.7
        
        # Check for criticism
        if any(word in message_lower for word in ['wrong', 'bad', 'terrible', 'stupid', 'useless']):
            event_type = EventType.USER_CRITICISM
            magnitude = 0.6
            emotional_content = 0.8
        
        # Check for creative tasks
        if any(word in message_lower for word in ['create', 'design', 'imagine', 'innovate']):
            event_type = EventType.CREATIVE_TASK
            novelty = 0.7
            complexity = 0.6
        
        # Check for learning
        if any(word in message_lower for word in ['explain', 'understand', 'learn', 'teach', 'how']):
            if event_type == EventType.UNKNOWN:
                event_type = EventType.LEARNING
            complexity = 0.5
            novelty = 0.4
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        return cls(
            type=event_type,
            magnitude=magnitude,
            timestamp=datetime.now(),
            event_id=event_id,
            complexity=complexity,
            urgency=urgency,
            emotional_content=emotional_content,
            novelty=novelty,
            success_probability=0.7,  # Default optimistic
            user_id=context.get('user_id'),
            metadata={
                'message': message,
                'context': context
            }
        )
    
    @classmethod
    def from_execution_result(cls, success: bool, execution_time: float, 
                             error: Optional[str] = None) -> 'Event':
        """Create an event from code execution results"""
        
        if success:
            event_type = EventType.TASK_SUCCESS
            magnitude = 0.8
            emotional_content = 0.3
        else:
            event_type = EventType.TASK_FAILURE
            magnitude = 0.6
            emotional_content = 0.5
        
        # Fast execution is less complex
        complexity = min(1.0, execution_time / 10.0)
        
        return cls(
            type=event_type,
            magnitude=magnitude,
            timestamp=datetime.now(),
            event_id=str(uuid.uuid4()),
            complexity=complexity,
            urgency=0.0,
            emotional_content=emotional_content,
            novelty=0.2,
            success_probability=0.5,
            metadata={
                'execution_time': execution_time,
                'error': error
            }
        )

    @property
    def context(self):
        """Get context from metadata"""
        return self.metadata.get('context', {})
