"""
Event processor for neurochemical responses
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import asyncio
from datetime import datetime
import logging

from ..core.constants import EventType, Hormone
from ..core.event import Event
from ..hormones import BaseHormone

logger = logging.getLogger(__name__)

class EventProcessor:
    """
    Processes events and triggers appropriate neurochemical responses
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize event processor
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        self.event_history = []
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        
        # Event weighting factors
        self.event_weights = {
            EventType.TASK_SUCCESS: 1.2,
            EventType.TASK_FAILURE: 1.5,
            EventType.SOCIAL_POSITIVE: 1.3,
            EventType.SOCIAL_NEGATIVE: 1.4,
            EventType.HIGH_COMPLEXITY: 1.0,
            EventType.TIME_PRESSURE: 1.1,
            EventType.UNCERTAINTY: 0.9,
            EventType.NOVELTY: 0.8,
            EventType.ROUTINE: 0.6,
            EventType.ERROR: 1.3,
            EventType.LEARNING: 0.7
        }
        
        # Cooldown tracking to prevent spam
        self.last_event_time = {}
        self.cooldown_period = 1.0  # seconds
        
    async def process(self, event: Event) -> Dict[str, float]:
        """
        Process an event through all hormones
        
        Args:
            event: Event to process
            
        Returns:
            Dictionary of hormone responses
        """
        # Check cooldown
        if not self._check_cooldown(event):
            logger.debug(f"Event {event.type} in cooldown period")
            return {}
        
        # Add to queue
        await self.processing_queue.put(event)
        
        # Process if not already processing
        if not self.is_processing:
            return await self._process_queue()
        
        return {}
    
    async def _process_queue(self) -> Dict[str, float]:
        """Process all queued events"""
        self.is_processing = True
        responses = {}
        
        try:
            while not self.processing_queue.empty():
                event = await self.processing_queue.get()
                event_responses = await self._process_single_event(event)
                
                # Aggregate responses
                for hormone, response in event_responses.items():
                    if hormone in responses:
                        responses[hormone] += response
                    else:
                        responses[hormone] = response
                        
        finally:
            self.is_processing = False
            
        return responses
    
    async def _process_single_event(self, event: Event) -> Dict[str, float]:
        """
        Process a single event
        
        Args:
            event: Event to process
            
        Returns:
            Dictionary of hormone responses
        """
        responses = {}
        
        # Get event weight
        weight = self.event_weights.get(event.type, 1.0)
        
        # Process through each hormone
        for hormone_name, hormone in self.state.hormones.items():
            # Calculate base response
            response = hormone.calculate_response(
                event.type.value if isinstance(event.type, EventType) else event.type,
                event.magnitude * weight,
                event.to_dict()
            )
            
            # Apply user-specific modulation
            if hasattr(self.state, 'pattern_recognizer'):
                pattern_mod = self.state.pattern_recognizer.get_response_modifier(
                    event.user_id, event.type, hormone_name
                )
                response *= pattern_mod
            
            responses[hormone_name] = response
            
            logger.debug(f"{hormone_name} response to {event.type}: {response:.3f}")
        
        # Store in history
        self.event_history.append({
            'event': event,
            'responses': responses,
            'timestamp': datetime.now()
        })
        
        # Update last event time
        self.last_event_time[event.type] = datetime.now()
        
        return responses
    
    def _check_cooldown(self, event: Event) -> bool:
        """
        Check if event type is in cooldown
        
        Args:
            event: Event to check
            
        Returns:
            True if event can be processed
        """
        if event.type not in self.last_event_time:
            return True
            
        time_since = (datetime.now() - self.last_event_time[event.type]).total_seconds()
        
        # Different cooldowns for different event types
        if event.type in [EventType.SOCIAL_POSITIVE, EventType.SOCIAL_NEGATIVE]:
            # Social events have longer cooldown
            return time_since > self.cooldown_period * 2
        elif event.type == EventType.ROUTINE:
            # Routine events have shorter cooldown
            return time_since > self.cooldown_period * 0.5
        else:
            return time_since > self.cooldown_period
    
    def get_event_statistics(self) -> Dict:
        """Get statistics about processed events"""
        if not self.event_history:
            return {}
            
        stats = {
            'total_events': len(self.event_history),
            'event_types': {},
            'average_responses': {},
            'recent_trend': None
        }
        
        # Count event types
        for entry in self.event_history:
            event_type = entry['event'].type
            if event_type not in stats['event_types']:
                stats['event_types'][event_type] = 0
            stats['event_types'][event_type] += 1
        
        # Calculate average responses
        for hormone_name in self.state.hormones.keys():
            responses = [entry['responses'].get(hormone_name, 0) 
                        for entry in self.event_history]
            if responses:
                stats['average_responses'][hormone_name] = np.mean(responses)
        
        # Recent trend (last 10 events)
        if len(self.event_history) > 10:
            recent = self.event_history[-10:]
            recent_positivity = sum(1 for e in recent 
                                  if e['event'].type in [EventType.TASK_SUCCESS, 
                                                         EventType.SOCIAL_POSITIVE])
            if recent_positivity > 7:
                stats['recent_trend'] = 'positive'
            elif recent_positivity < 3:
                stats['recent_trend'] = 'negative'
            else:
                stats['recent_trend'] = 'neutral'
        
        return stats
    
    def predict_response(self, event_type: EventType, magnitude: float) -> Dict[str, float]:
        """
        Predict hormone responses without actually processing
        
        Args:
            event_type: Type of event
            magnitude: Event magnitude
            
        Returns:
            Predicted responses
        """
        predictions = {}
        weight = self.event_weights.get(event_type, 1.0)
        
        for hormone_name, hormone in self.state.hormones.items():
            # Get baseline prediction
            prediction = hormone.calculate_response(
                event_type.value,
                magnitude * weight,
                {}
            )
            predictions[hormone_name] = prediction
            
        return predictions
