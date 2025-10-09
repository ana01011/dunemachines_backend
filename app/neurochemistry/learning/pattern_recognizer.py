"""
Pattern recognition for learning user-specific neurochemical responses
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging

from ..core.constants import EventType, Hormone
from ..core.event import Event

logger = logging.getLogger(__name__)

class PatternRecognizer:
    """
    Learns patterns in user behavior and neurochemical responses
    """
    
    def __init__(self, user_id: str):
        """
        Initialize pattern recognizer for a specific user
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        
        # Pattern storage
        self.event_patterns = defaultdict(lambda: {
            'count': 0,
            'total_magnitude': 0.0,
            'hormone_responses': defaultdict(list),
            'contexts': [],
            'timestamps': deque(maxlen=100)
        })
        
        # Sequence patterns (what follows what)
        self.sequence_patterns = defaultdict(lambda: defaultdict(int))
        self.last_event_type = None
        
        # Time patterns (when things happen)
        self.time_patterns = {
            'hourly': defaultdict(list),
            'daily': defaultdict(list),
            'weekly': defaultdict(list)
        }
        
        # User preferences learned over time
        self.user_preferences = {
            'preferred_complexity': 0.5,
            'risk_tolerance': 0.5,
            'social_preference': 0.5,
            'learning_style': 'balanced',  # visual, verbal, kinesthetic, balanced
            'patience_level': 0.5,
            'detail_preference': 0.5
        }
        
        # Success/failure patterns
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        
        # Interaction style patterns
        self.interaction_patterns = {
            'message_length_avg': 0,
            'question_frequency': 0,
            'clarification_needed': 0,
            'preferred_response_length': 'medium',
            'technical_level': 0.5
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.memory_window = 1000  # Number of events to remember
        self.confidence_threshold = 10  # Minimum events before making predictions
        
    def observe(self, event: Event):
        """
        Observe and learn from an event
        
        Args:
            event: Event to learn from
        """
        event_type = event.type.value if isinstance(event.type, EventType) else event.type
        
        # Update basic patterns
        pattern = self.event_patterns[event_type]
        pattern['count'] += 1
        pattern['total_magnitude'] += event.magnitude
        pattern['timestamps'].append(datetime.now())
        
        # Store context
        if event.context:
            pattern['contexts'].append({
                'time': datetime.now(),
                'context': event.context,
                'magnitude': event.magnitude
            })
        
        # Update sequence patterns
        if self.last_event_type:
            self.sequence_patterns[self.last_event_type][event_type] += 1
        self.last_event_type = event_type
        
        # Update time patterns
        now = datetime.now()
        self.time_patterns['hourly'][now.hour].append(event_type)
        self.time_patterns['daily'][now.weekday()].append(event_type)
        self.time_patterns['weekly'][now.isocalendar()[1] % 4].append(event_type)
        
        # Learn from success/failure
        if event_type == EventType.TASK_SUCCESS:
            self._learn_from_success(event)
        elif event_type == EventType.TASK_FAILURE:
            self._learn_from_failure(event)
            
        # Update user preferences
        self._update_user_preferences(event)
        
        logger.debug(f"Observed {event_type} for user {self.user_id}")
    
    def _learn_from_success(self, event: Event):
        """Learn from successful interactions"""
        context = event.context or {}
        
        self.success_patterns['magnitudes'].append(event.magnitude)
        self.success_patterns['contexts'].append(context)
        
        # Increase preference for things that led to success
        if 'complexity' in context:
            self.user_preferences['preferred_complexity'] = \
                0.9 * self.user_preferences['preferred_complexity'] + 0.1 * context['complexity']
                
        if 'approach' in context:
            self.success_patterns['approaches'].append(context['approach'])
            
        # Build confidence
        self.user_preferences['risk_tolerance'] = min(0.9, 
            self.user_preferences['risk_tolerance'] + 0.01)
    
    def _learn_from_failure(self, event: Event):
        """Learn from failures"""
        context = event.context or {}
        
        self.failure_patterns['magnitudes'].append(event.magnitude)
        self.failure_patterns['contexts'].append(context)
        
        # Adjust preferences to avoid failure patterns
        if 'complexity' in context and context['complexity'] > 0.7:
            # High complexity led to failure, reduce preference
            self.user_preferences['preferred_complexity'] *= 0.95
            
        # Reduce risk tolerance after failure
        self.user_preferences['risk_tolerance'] = max(0.1,
            self.user_preferences['risk_tolerance'] - 0.02)
    
    def _update_user_preferences(self, event: Event):
        """Update learned user preferences"""
        if not event.context:
            return
            
        context = event.context
        
        # Learn patience level from time pressure responses
        if event.type == EventType.TIME_PRESSURE:
            # User asking for urgent response indicates lower patience
            self.user_preferences['patience_level'] *= 0.95
            
        # Learn social preference
        if event.type in [EventType.SOCIAL_POSITIVE, EventType.SOCIAL_NEGATIVE]:
            weight = 0.1 if event.type == EventType.SOCIAL_POSITIVE else -0.1
            self.user_preferences['social_preference'] = \
                np.clip(self.user_preferences['social_preference'] + weight, 0, 1)
                
        # Learn detail preference from message length
        if 'message_length' in context:
            if context['message_length'] > 500:
                self.user_preferences['detail_preference'] = \
                    min(1.0, self.user_preferences['detail_preference'] + 0.01)
            elif context['message_length'] < 100:
                self.user_preferences['detail_preference'] = \
                    max(0.0, self.user_preferences['detail_preference'] - 0.01)
    
    def predict_next_event(self) -> Tuple[Optional[str], float]:
        """
        Predict the most likely next event type
        
        Returns:
            Tuple of (predicted_event_type, confidence)
        """
        if not self.last_event_type:
            return None, 0.0
            
        # Get sequence predictions
        next_events = self.sequence_patterns.get(self.last_event_type, {})
        
        if not next_events:
            return None, 0.0
            
        # Find most likely next event
        total_count = sum(next_events.values())
        if total_count < self.confidence_threshold:
            return None, 0.0
            
        most_likely = max(next_events.items(), key=lambda x: x[1])
        confidence = most_likely[1] / total_count
        
        return most_likely[0], confidence
    
    def get_response_modifier(self, user_id: str, event_type: str, 
                            hormone: str) -> float:
        """
        Get user-specific response modifier for a hormone
        
        Args:
            user_id: User identifier
            event_type: Type of event
            hormone: Hormone name
            
        Returns:
            Modifier value (0.5 to 1.5)
        """
        if user_id != self.user_id:
            return 1.0
            
        pattern = self.event_patterns.get(event_type)
        if not pattern or pattern['count'] < self.confidence_threshold:
            return 1.0
            
        # Calculate modifier based on learned patterns
        modifier = 1.0
        
        # Adjust based on user preferences
        if hormone == 'dopamine':
            # Users with high risk tolerance have stronger dopamine responses
            modifier *= (0.8 + 0.4 * self.user_preferences['risk_tolerance'])
            
        elif hormone == 'cortisol':
            # Patient users have lower cortisol responses
            modifier *= (1.3 - 0.6 * self.user_preferences['patience_level'])
            
        elif hormone == 'oxytocin':
            # Social users have stronger oxytocin responses
            modifier *= (0.7 + 0.6 * self.user_preferences['social_preference'])
            
        elif hormone == 'serotonin':
            # Detail-oriented users build serotonin more steadily
            modifier *= (0.9 + 0.2 * self.user_preferences['detail_preference'])
            
        return np.clip(modifier, 0.5, 1.5)
    
    def get_time_based_prediction(self) -> Dict[str, float]:
        """
        Predict hormone states based on time patterns
        
        Returns:
            Dictionary of predicted hormone modifiers
        """
        now = datetime.now()
        hour = now.hour
        day = now.weekday()
        
        predictions = {
            'dopamine': 1.0,
            'cortisol': 1.0,
            'adrenaline': 1.0,
            'serotonin': 1.0,
            'oxytocin': 1.0
        }
        
        # Check hourly patterns
        hourly_events = self.time_patterns['hourly'].get(hour, [])
        if hourly_events:
            # More events at this hour = higher baseline activity
            activity_level = len(hourly_events) / max(1, len(self.time_patterns['hourly']))
            
            if activity_level > 0.2:  # Active hour
                predictions['adrenaline'] *= 1.1
                predictions['dopamine'] *= 1.05
            else:  # Quiet hour
                predictions['serotonin'] *= 1.1
                predictions['cortisol'] *= 0.9
        
        # Check daily patterns
        daily_events = self.time_patterns['daily'].get(day, [])
        if daily_events:
            # Detect stress patterns (many failures on certain days)
            stress_events = [e for e in daily_events if e in 
                           [EventType.TASK_FAILURE, EventType.ERROR, EventType.TIME_PRESSURE]]
            if len(stress_events) > len(daily_events) * 0.4:
                predictions['cortisol'] *= 1.15
                
        return predictions
    
    def get_user_profile(self) -> Dict:
        """
        Get comprehensive user profile based on learned patterns
        
        Returns:
            User profile dictionary
        """
        profile = {
            'user_id': self.user_id,
            'preferences': self.user_preferences.copy(),
            'total_events': sum(p['count'] for p in self.event_patterns.values()),
            'most_common_events': [],
            'success_rate': 0.0,
            'avg_complexity_handled': 0.0,
            'interaction_style': self.interaction_patterns.copy(),
            'peak_activity_hours': [],
            'stress_triggers': [],
            'reward_triggers': []
        }
        
        # Find most common events
        sorted_events = sorted(self.event_patterns.items(), 
                             key=lambda x: x[1]['count'], reverse=True)
        profile['most_common_events'] = [(e, p['count']) for e, p in sorted_events[:5]]
        
        # Calculate success rate
        success_count = self.event_patterns.get(EventType.TASK_SUCCESS, {}).get('count', 0)
        failure_count = self.event_patterns.get(EventType.TASK_FAILURE, {}).get('count', 0)
        if success_count + failure_count > 0:
            profile['success_rate'] = success_count / (success_count + failure_count)
            
        # Find peak activity hours
        hourly_counts = [(h, len(events)) for h, events in self.time_patterns['hourly'].items()]
        hourly_counts.sort(key=lambda x: x[1], reverse=True)
        profile['peak_activity_hours'] = [h for h, _ in hourly_counts[:3]]
        
        # Identify stress triggers
        stress_events = [EventType.TASK_FAILURE, EventType.ERROR, 
                        EventType.TIME_PRESSURE, EventType.UNCERTAINTY]
        profile['stress_triggers'] = [e for e in stress_events 
                                     if self.event_patterns.get(e, {}).get('count', 0) > 5]
        
        # Identify reward triggers  
        reward_events = [EventType.TASK_SUCCESS, EventType.SOCIAL_POSITIVE,
                        EventType.LEARNING]
        profile['reward_triggers'] = [e for e in reward_events
                                     if self.event_patterns.get(e, {}).get('count', 0) > 5]
        
        return profile
    
    def export_patterns(self) -> str:
        """
        Export learned patterns to JSON
        
        Returns:
            JSON string of patterns
        """
        export_data = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'event_patterns': dict(self.event_patterns),
            'sequence_patterns': dict(self.sequence_patterns),
            'user_preferences': self.user_preferences,
            'success_patterns': dict(self.success_patterns),
            'failure_patterns': dict(self.failure_patterns),
            'interaction_patterns': self.interaction_patterns
        }
        
        return json.dumps(export_data, default=str, indent=2)
    
    def import_patterns(self, json_data: str):
        """
        Import previously learned patterns
        
        Args:
            json_data: JSON string of patterns
        """
        try:
            data = json.loads(json_data)
            
            if data.get('user_id') != self.user_id:
                logger.warning(f"Pattern user_id mismatch: {data.get('user_id')} != {self.user_id}")
                
            # Restore patterns
            self.event_patterns.update(data.get('event_patterns', {}))
            self.sequence_patterns.update(data.get('sequence_patterns', {}))
            self.user_preferences.update(data.get('user_preferences', {}))
            self.success_patterns.update(data.get('success_patterns', {}))
            self.failure_patterns.update(data.get('failure_patterns', {}))
            self.interaction_patterns.update(data.get('interaction_patterns', {}))
            
            logger.info(f"Imported patterns for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to import patterns: {e}")
