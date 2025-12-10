"""
Main interface for the neurochemistry system
Provides clean API for the application
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import time
import asyncio
from dataclasses import dataclass

from .core.state import NeurochemicalState
from .core.dynamics import NeurochemicalDynamics
from .core.minimization import NeurochemicalMinimization
from .core.constants import *

@dataclass
class NeurochemicalEvent:
    """Event that affects neurochemical state"""
    event_type: str
    intensity: float = 0.5
    valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.5  # 0 to 1 (calm to excited)
    social: float = 0.0   # 0 to 1 (alone to connected)
    novelty: float = 0.5  # 0 to 1 (familiar to novel)
    uncertainty: float = 0.5  # 0 to 1 (certain to uncertain)
    duration: float = 1.0  # Expected duration in seconds

class NeurochemicalSystem:
    """
    Main interface for neurochemical processing
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.state = NeurochemicalState(user_id=user_id)
        self.dynamics = NeurochemicalDynamics(self.state)
        self.minimization = NeurochemicalMinimization(self.state)
        
        # Simulation parameters
        self.dt = DT_INITIAL
        self.time_elapsed = 0.0
        
        # Event queue
        self.event_queue: List[NeurochemicalEvent] = []
        
        # Performance metrics
        self.last_update = time.time()
        self.update_count = 0
        
    def process_message(self, message: str) -> Dict:
        """
        Process a text message and return neurochemical response
        """
        # Analyze message for neurochemical triggers
        event = self._analyze_message(message)
        
        # Convert to inputs
        inputs = self._event_to_inputs(event)
        
        # Run dynamics
        self.dynamics.step(self.dt, inputs)
        
        # Calculate costs and optimization
        total_cost, cost_components = self.minimization.calculate_total_cost()
        seeking = self.minimization.calculate_seeking_intensity()
        suggestions = self.minimization.suggest_optimal_action()
        
        # Get behavioral output
        behavioral = self.state.get_behavioral_parameters()
        mood = self.state.get_mood_state()
        
        # Prepare response
        response = {
            'user_id': self.user_id,
            'mood': mood,
            'behavioral': behavioral,
            'hormones': {
                'dopamine': float(self.state.dopamine),
                'serotonin': float(self.state.serotonin),
                'cortisol': float(self.state.cortisol),
                'adrenaline': float(self.state.adrenaline),
                'oxytocin': float(self.state.oxytocin),
                'norepinephrine': float(self.state.norepinephrine),
                'endorphins': float(self.state.endorphins)
            },
            'effective': {
                HORMONES[i]: float(self.state.get_effective_hormones()[i])
                for i in range(7)
            },
            'receptors': {
                HORMONES[i]: float(self.state.receptors[i])
                for i in range(7)
            },
            'baselines': {
                HORMONES[i]: float(self.state.baselines[i])
                for i in range(7)
            },
            'cost': cost_components,
            'efficiency': self.minimization.calculate_efficiency_score(),
            'seeking': seeking,
            'suggestions': suggestions,
            'resources': {
                'tyrosine': self.state.p_tyr / P_TYR_MAX,
                'tryptophan': self.state.p_trp / P_TRP_MAX,
                'atp': self.state.e_atp / E_ATP_MAX
            },
            'allostatic_load': self.state.allostatic_load / L_MAX
        }
        
        self.update_count += 1
        self.last_update = time.time()
        
        return response
    
    def _analyze_message(self, message: str) -> NeurochemicalEvent:
        """
        Analyze message content for neurochemical triggers
        """
        message_lower = message.lower()
        
        # Detect exercise first
        exercise_words = ['exercise', 'workout', 'run', 'running', 'marathon', 'gym', 
                         'burn', 'sweat', 'endorphin', 'fitness', 'training']
        is_exercise = any(word in message_lower for word in exercise_words)
        
        # Detect relaxation
        relax_words = ['relax', 'calm', 'peace', 'rest', 'sleep', 'meditat', 
                       'quiet', 'peaceful', 'tranquil']
        is_relaxation = any(word in message_lower for word in relax_words)
        
        # Detect emotional valence
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'sad', 'angry', 'hate', 'terrible', 'awful', 'horrible']
        
        pos_count = sum(1 for word in positive_words if word in message_lower)
        neg_count = sum(1 for word in negative_words if word in message_lower)
        
        if pos_count + neg_count > 0:
            valence = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            valence = 0.0
        
        # Detect arousal
        high_arousal_words = ['urgent', 'emergency', 'now', 'immediately', 'quick', '!']
        arousal = 0.8 if any(word in message_lower for word in high_arousal_words) else 0.3
        
        # Detect social content
        social_words = ['we', 'us', 'together', 'friend', 'family', 'team', 'people']
        social = 0.8 if any(word in message_lower for word in social_words) else 0.2
        
        # Detect novelty
        novel_words = ['new', 'never', 'first', 'unique', 'strange', 'unusual']
        novelty = 0.8 if any(word in message_lower for word in novel_words) else 0.3
        
        # Detect uncertainty
        uncertain_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', '?']
        uncertainty = 0.7 if any(word in message_lower for word in uncertain_words) else 0.3
        
        # Determine event type
        if is_exercise:
            event_type = 'exercise'
        elif is_relaxation:
            event_type = 'relaxation'
        elif 'help' in message_lower or 'problem' in message_lower:
            event_type = 'problem_solving'
        elif social > 0.5:
            event_type = 'social_interaction'
        elif pos_count > neg_count:
            event_type = 'positive_feedback'
        elif neg_count > pos_count:
            event_type = 'negative_feedback'
        else:
            event_type = 'neutral_interaction'
        
        # Intensity based on message length and punctuation
        intensity = min(1.0, len(message) / 200 + message.count('!') * 0.2)
        
        event = NeurochemicalEvent(
            event_type=event_type,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            social=social,
            novelty=novelty,
            uncertainty=uncertainty,
            duration=1.0
        )
        event.message = message  # Store original message for keyword detection
        return event
    
    def _event_to_inputs(self, event: NeurochemicalEvent) -> Dict[str, float]:
        """
        Convert event to dynamics inputs - ENHANCED
        """
        # Detect exercise keywords
        message_lower = event.event_type.lower() if hasattr(event, 'message') else ""
        exercise_words = ['exercise', 'workout', 'run', 'gym', 'marathon', 'burn', 'sweat', 'endorphin']
        is_exercise = any(word in message_lower for word in exercise_words) or 'exercise' in event.event_type.lower()
        
        # Detect relaxation keywords
        relax_words = ['relax', 'calm', 'peace', 'rest', 'sleep', 'meditat', 'quiet']
        is_relaxation = any(word in message_lower for word in relax_words) or 'relax' in event.event_type.lower()
        
        inputs = {
            'reward': max(0, event.valence) * event.intensity,
            'punishment': max(0, -event.valence) * event.intensity,
            'threat': max(0, -event.valence) * event.arousal if not is_relaxation else 0,
            'urgency': event.arousal * event.intensity if not is_relaxation else 0,
            'social': event.social * event.intensity,
            'novelty': event.novelty,
            'uncertainty': event.uncertainty if not is_relaxation else 0,
            'attention': event.arousal if not is_relaxation else 0.1,
            'trust': event.social * max(0, event.valence),
            'touch': 0.0,  
            'exercise': 1.0 if is_exercise else 0.0,  # Now properly detected!
            'pain': max(0, -event.valence) * 0.5,
            'pleasure': max(0, event.valence) * 0.5,
            'fight_flight': event.arousal * max(0, -event.valence) if not is_relaxation else 0,
            'attachment': event.social * event.duration / 10,
            'nutrition': 0.5,  
            'glucose': 0.7,    
            'oxygen': 0.9,     
            'temperature': 1.0, 
            'sleep': 0.8 if is_relaxation else 0.0  # Relaxation triggers sleep-like state
        }
        
        return inputs
    
    async def process_stream(self, message_stream) -> None:
        """
        Process a stream of messages asynchronously
        """
        async for message in message_stream:
            response = self.process_message(message)
            yield response
    
    def get_state_summary(self) -> Dict:
        """
        Get a summary of current neurochemical state
        """
        return {
            'mood': self.state.get_mood_state(),
            'behavioral': self.state.get_behavioral_parameters(),
            'primary_hormones': {
                'dopamine': f"{self.state.dopamine:.1f}",
                'serotonin': f"{self.state.serotonin:.1f}",
                'cortisol': f"{self.state.cortisol:.1f}"
            },
            'efficiency': f"{self.minimization.calculate_efficiency_score():.2f}",
            'allostatic_load': f"{self.state.allostatic_load:.1f}",
            'energy': f"{self.state.e_atp:.1f}%"
        }
    
    def reset(self) -> None:
        """
        Reset to initial state
        """
        self.state = NeurochemicalState(user_id=self.user_id)
        self.dynamics = NeurochemicalDynamics(self.state)
        self.minimization = NeurochemicalMinimization(self.state)
        self.time_elapsed = 0.0
        self.update_count = 0
    
    def save_state(self) -> Dict:
        """
        Save current state for persistence
        """
        return self.state.to_dict()
    
    def load_state(self, state_dict: Dict) -> None:
        """
        Load state from saved data
        """
        self.state = NeurochemicalState.from_dict(state_dict)
        self.dynamics = NeurochemicalDynamics(self.state)
        self.minimization = NeurochemicalMinimization(self.state)
    
    def simulate_time_passage(self, seconds: float, rest: bool = False) -> None:
        """
        Simulate passage of time without events
        Useful for modeling decay, recovery, circadian rhythms
        """
        steps = int(seconds / self.dt)
        
        # Default inputs for rest or normal activity
        if rest:
            inputs = {
                'reward': 0, 'threat': 0, 'social': 0.1,
                'urgency': 0, 'attention': 0.1, 'sleep': 0.8,
                'nutrition': 0.6, 'glucose': 0.8, 'oxygen': 0.95
            }
        else:
            inputs = {
                'reward': 0.1, 'threat': 0.1, 'social': 0.3,
                'urgency': 0.2, 'attention': 0.4, 'sleep': 0,
                'nutrition': 0.5, 'glucose': 0.7, 'oxygen': 0.9
            }
        
        for _ in range(steps):
            self.dynamics.step(self.dt, inputs)
            self.time_elapsed += self.dt
    
    def get_diagnostics(self) -> Dict:
        """
        Get detailed diagnostics for debugging
        """
        return {
            'update_count': self.update_count,
            'time_elapsed': self.time_elapsed,
            'last_update': time.time() - self.last_update,
            'cost_trajectory': self.minimization.get_cost_trajectory(),
            'gradient': self.minimization.calculate_cost_gradient().tolist(),
            'optimal_baseline_shift': self.minimization.calculate_optimal_baseline_shift().tolist(),
            'state_valid': self._validate_state()
        }
    
    def _validate_state(self) -> bool:
        """
        Validate that state is within biological bounds
        """
        valid = True
        
        # Check hormones
        if np.any(self.state.hormones < HORMONE_MIN) or np.any(self.state.hormones > HORMONE_MAX):
            valid = False
        
        # Check receptors
        if np.any(self.state.receptors < RECEPTOR_MIN) or np.any(self.state.receptors > RECEPTOR_MAX):
            valid = False
        
        # Check resources
        if self.state.p_tyr < 0 or self.state.p_trp < 0 or self.state.e_atp < 0:
            valid = False
        
        # Check for NaN
        if np.any(np.isnan(self.state.hormones)) or np.any(np.isnan(self.state.receptors)):
            valid = False
        
        return valid
