"""
Adrenaline hormone implementation - Urgency and energy
"""
from typing import Dict, Optional
import numpy as np
import time

from .base_hormone import BaseHormone
from ..core.constants import EventType

class Adrenaline(BaseHormone):
    """
    Adrenaline: The urgency and performance hormone
    
    Characteristics:
    - Spikes with challenges, novelty, and time pressure
    - Depletes quickly (limited pool)
    - Enhances processing speed but can reduce accuracy
    - Creates "rush" feeling for complex problems
    """
    
    def __init__(self, config: Dict):
        """Initialize adrenaline with specific configuration"""
        super().__init__('adrenaline', config)
        
        # Adrenaline-specific parameters
        self.depletion_rate = float(config.get('depletion_rate', 0.1))
        self.max_pool = float(config.get('max_pool', 100))
        self.regen_rate = float(config.get('regen_rate', 5.0))
        
        # Current pool (adrenaline is limited resource)
        self.current_pool = self.max_pool
        
        # Activity tracking
        self.last_spike_time = 0
        self.refractory_period = 10.0  # Seconds before can spike again
        self.consecutive_spikes = 0
        
        # Performance modulation
        self.accuracy_penalty_threshold = 70  # Above this, accuracy drops
        self.speed_boost_factor = 0.0
        
        # Exhaustion tracking
        self.exhaustion_level = 0.0
        self.is_exhausted = False
        
    def calculate_response(self, event_type: str, magnitude: float,
                          context: Optional[Dict] = None) -> float:
        """
        Calculate adrenaline response to events
        
        Adrenaline responds to:
        - Novel challenges (excitement)
        - Time pressure (urgency)
        - High stakes situations
        - Competition or performance pressure
        - Emergency situations
        """
        response = 0.0
        
        # Check if still in refractory period
        time_since_spike = time.time() - self.last_spike_time
        if time_since_spike < self.refractory_period:
            # Reduced response during refractory
            magnitude *= (time_since_spike / self.refractory_period)
        
        # Check if pool is depleted
        if self.current_pool < 20:
            # Can't spike if exhausted
            return 0.0
        
        if event_type == EventType.NOVELTY:
            # Novel situations trigger excitement
            response = self.sensitivity * magnitude * 0.4
            
            # Higher response for very novel situations
            if context and context.get('novelty_score', 0) > 0.8:
                response *= 1.5
                
        elif event_type == EventType.TIME_PRESSURE:
            # Urgency triggers adrenaline
            urgency = context.get('time_remaining', 1.0) if context else 1.0
            
            # More adrenaline as deadline approaches
            if urgency < 0.2:  # Less than 20% time remaining
                response = self.sensitivity * 0.7
            elif urgency < 0.5:
                response = self.sensitivity * 0.5
            else:
                response = self.sensitivity * 0.3
                
        elif event_type == EventType.HIGH_COMPLEXITY:
            # Complex challenges can be exciting
            if context and context.get('confidence', 0.5) > 0.6:
                # Confident about complex task = excitement
                response = self.sensitivity * magnitude * 0.45
            else:
                # Not confident = stress without adrenaline
                response = self.sensitivity * magnitude * 0.2
                
        elif event_type == EventType.TASK_SUCCESS:
            # Success after challenge maintains adrenaline
            if self.current_level > self.baseline + 20:
                response = 0.1 * magnitude  # Small boost to maintain
            else:
                response = -0.2 * magnitude  # Come down if not already high
                
        elif event_type == EventType.TASK_FAILURE:
            # Failure can trigger fight-or-flight
            if context and context.get('recoverable', True):
                # Recoverable failure = rally
                response = self.sensitivity * magnitude * 0.35
            else:
                # Catastrophic failure = crash
                response = -0.5 * magnitude
        
        # Deplete from pool
        if response > 0:
            depletion = response * self.depletion_rate
            self.current_pool -= depletion
            self.current_pool = max(0, self.current_pool)
            
            # Track spike
            if response > 20:
                self.last_spike_time = time.time()
                self.consecutive_spikes += 1
        
        return response
    
    def calculate_interaction(self, other_hormones: Dict[str, BaseHormone]) -> float:
        """
        Calculate interactions with other hormones
        
        Adrenaline:
        - Is triggered by cortisol (stress → action)
        - Amplifies dopamine (excitement + reward)
        - Is suppressed by serotonin (contentment → calm)
        - Depletes faster with high activity
        """
        interaction = 0.0
        
        # Cortisol trigger (stress → action)
        if 'cortisol' in other_hormones:
            cortisol = other_hormones['cortisol']
            if cortisol.current_level > cortisol.baseline + 20:
                # High stress triggers adrenaline
                trigger = 0.02 * (cortisol.current_level - cortisol.baseline - 20)
                interaction += trigger * 30
                
        # Dopamine synergy (excitement + anticipation)
        if 'dopamine' in other_hormones:
            dopamine = other_hormones['dopamine']
            # Rising dopamine creates excitement
            if dopamine.current_level > dopamine.previous_level:
                dopamine_rate = dopamine.current_level - dopamine.previous_level
                synergy = 0.1 * dopamine_rate
                interaction += synergy * 20
                
        # Serotonin suppression (calm → reduced adrenaline)
        if 'serotonin' in other_hormones:
            serotonin = other_hormones['serotonin']
            if serotonin.current_level > 70:
                suppression = -0.02 * (serotonin.current_level - 70)
                interaction += suppression * self.current_level / 100
        
        return interaction
    
    def update(self, dt: float, event: Optional[Dict] = None,
               other_hormones: Optional[Dict] = None) -> None:
        """
        Extended update with pool management and exhaustion
        """
        # Regular update
        super().update(dt, event, other_hormones)
        
        # Regenerate pool slowly
        if self.current_pool < self.max_pool:
            regen_amount = self.regen_rate * dt
            
            # Slower regen if exhausted
            if self.exhaustion_level > 0.5:
                regen_amount *= 0.5
                
            self.current_pool += regen_amount
            self.current_pool = min(self.max_pool, self.current_pool)
        
        # Update exhaustion level
        if self.current_pool < 30:
            self.exhaustion_level += 0.01 * dt
        else:
            self.exhaustion_level -= 0.02 * dt
        self.exhaustion_level = np.clip(self.exhaustion_level, 0, 1)
        
        # Check if exhausted
        self.is_exhausted = self.exhaustion_level > 0.7 or self.current_pool < 10
        
        # Calculate performance modulation
        if self.current_level > self.baseline:
            excess = self.current_level - self.baseline
            # Speed boost proportional to adrenaline
            self.speed_boost_factor = min(1.0, excess / 50)
        else:
            self.speed_boost_factor = 0.0
        
        # Reset consecutive spikes if been calm for a while
        if self.current_level < self.baseline + 10:
            time_since_spike = time.time() - self.last_spike_time
            if time_since_spike > 60:  # 1 minute of calm
                self.consecutive_spikes = 0
    
    def get_performance_modifiers(self) -> Dict[str, float]:
        """
        Get performance modifiers based on adrenaline level
        
        Returns:
            Dictionary of performance modifiers
        """
        modifiers = {
            'processing_speed': 1.0,
            'accuracy': 1.0,
            'parallel_thinking': 1.0,
            'risk_taking': 1.0
        }
        
        if self.current_level > self.baseline:
            excess = self.current_level - self.baseline
            
            # Speed boost
            modifiers['processing_speed'] = 1.0 + self.speed_boost_factor * 0.5
            
            # Parallel thinking boost
            modifiers['parallel_thinking'] = 1.0 + min(0.5, excess / 100)
            
            # Risk taking increase
            modifiers['risk_taking'] = 1.0 + min(0.3, excess / 150)
            
            # Accuracy penalty if too high
            if self.current_level > self.accuracy_penalty_threshold:
                overage = self.current_level - self.accuracy_penalty_threshold
                modifiers['accuracy'] = 1.0 - min(0.3, overage / 100)
        
        # Exhaustion penalties
        if self.is_exhausted:
            modifiers['processing_speed'] *= 0.7
            modifiers['accuracy'] *= 0.8
            modifiers['parallel_thinking'] *= 0.5
        
        return modifiers
    
    def can_spike(self) -> bool:
        """Check if adrenaline can spike"""
        time_since_spike = time.time() - self.last_spike_time
        return (not self.is_exhausted and 
                self.current_pool > 30 and 
                time_since_spike > self.refractory_period)
    
    def get_state_summary(self) -> Dict:
        """Extended state summary with adrenaline-specific info"""
        summary = super().get_state_summary()
        performance = self.get_performance_modifiers()
        summary.update({
            'current_pool': round(self.current_pool, 1),
            'pool_percentage': round(self.current_pool / self.max_pool * 100, 1),
            'exhaustion_level': round(self.exhaustion_level, 3),
            'is_exhausted': self.is_exhausted,
            'can_spike': self.can_spike(),
            'consecutive_spikes': self.consecutive_spikes,
            'speed_boost': round(performance['processing_speed'], 2),
            'accuracy_modifier': round(performance['accuracy'], 2),
            'time_since_spike': round(time.time() - self.last_spike_time, 1)
        })
        return summary
