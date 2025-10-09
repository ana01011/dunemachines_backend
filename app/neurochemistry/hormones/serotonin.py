"""
Serotonin hormone implementation - Satisfaction and well-being
"""
from typing import Dict, Optional, List
import numpy as np
import time
from collections import deque

from .base_hormone import BaseHormone
from ..core.constants import EventType

class Serotonin(BaseHormone):
    """
    Serotonin: The satisfaction and well-being hormone
    
    Characteristics:
    - Builds slowly from consistent success
    - Provides stability and confidence
    - Dampens volatility in other hormones
    - Associated with contentment and wisdom
    - Depletes with chronic stress or repeated failures
    """
    
    def __init__(self, config: Dict):
        """Initialize serotonin with specific configuration"""
        super().__init__('serotonin', config)
        
        # Serotonin-specific parameters
        self.accumulation_rate = float(config.get('accumulation_rate', 0.05))
        self.stability_factor = float(config.get('stability_factor', 0.8))
        
        # Success tracking
        self.recent_successes = deque(maxlen=20)
        self.success_streak = 0
        self.consistency_score = 0.5
        
        # Well-being metrics
        self.life_satisfaction = 0.5  # Long-term satisfaction
        self.confidence_level = 0.5
        self.wisdom_score = 0.0  # Builds from experience
        
        # Depletion tracking
        self.depletion_factors = []
        self.is_depleted = False
        
        # Memory of positive experiences
        self.positive_memories = deque(maxlen=50)
        self.memory_influence = 0.0
        
    def calculate_response(self, event_type: str, magnitude: float,
                          context: Optional[Dict] = None) -> float:
        """
        Calculate serotonin response to events
        
        Serotonin responds to:
        - Consistent success (gradual building)
        - Task completion (satisfaction)
        - Social validation (positive feedback)
        - Learning and growth
        - Stability and routine
        """
        response = 0.0
        
        if event_type == EventType.TASK_SUCCESS:
            # Success builds serotonin slowly
            response = self.sensitivity * magnitude * 0.2
            
            # Bonus for streak
            if self.success_streak > 3:
                response *= (1 + min(0.5, self.success_streak * 0.1))
            
            # Track success
            self.recent_successes.append({
                'time': time.time(),
                'magnitude': magnitude
            })
            self.success_streak += 1
            
            # Add to positive memories
            if magnitude > 0.7:
                self.positive_memories.append({
                    'time': time.time(),
                    'type': 'achievement',
                    'strength': magnitude
                })
                
        elif event_type == EventType.SOCIAL_POSITIVE:
            # Positive social feedback boosts serotonin
            response = self.sensitivity * magnitude * 0.25
            
            # Social validation is particularly powerful
            self.positive_memories.append({
                'time': time.time(),
                'type': 'social',
                'strength': magnitude
            })
            
        elif event_type == EventType.LEARNING:
            # Learning and growth increase serotonin
            response = self.sensitivity * magnitude * 0.15
            
            # Build wisdom
            self.wisdom_score += 0.01 * magnitude
            self.wisdom_score = min(1.0, self.wisdom_score)
            
        elif event_type == EventType.ROUTINE:
            # Routine and stability maintain serotonin
            if self.current_level > self.baseline:
                # Maintain current level
                response = 0.05 * magnitude
            else:
                # Slowly build if below baseline
                response = 0.1 * magnitude
                
        elif event_type == EventType.TASK_FAILURE:
            # Failure depletes serotonin
            response = -0.15 * magnitude
            
            # Break success streak
            self.success_streak = 0
            
            # Track depletion
            self.depletion_factors.append({
                'time': time.time(),
                'reason': 'failure',
                'magnitude': magnitude
            })
            
        elif event_type == EventType.SOCIAL_NEGATIVE:
            # Negative social feedback hurts serotonin
            response = -0.3 * magnitude
            
            self.depletion_factors.append({
                'time': time.time(),
                'reason': 'social_rejection',
                'magnitude': magnitude
            })
        
        # Apply wisdom modifier (experience dampens volatility)
        if self.wisdom_score > 0.3:
            if response < 0:
                # Wisdom reduces negative impact
                response *= (1 - self.wisdom_score * 0.3)
            else:
                # Wisdom provides steady gains
                response *= (1 + self.wisdom_score * 0.2)
        
        return response
    
    def calculate_interaction(self, other_hormones: Dict[str, BaseHormone]) -> float:
        """
        Calculate interactions with other hormones
        
        Serotonin:
        - Is depleted by chronic cortisol (stress depletes well-being)
        - Is boosted by oxytocin (social bonding)
        - Stabilizes all other hormones (damping effect)
        - Competes with dopamine (contentment vs seeking)
        """
        interaction = 0.0
        
        # Cortisol depletion (chronic stress hurts well-being)
        if 'cortisol' in other_hormones:
            cortisol = other_hormones['cortisol']
            # Check for chronic elevation
            if hasattr(cortisol, 'chronic_stress_level'):
                if cortisol.chronic_stress_level > 0.5:
                    depletion = -0.01 * cortisol.chronic_stress_level
                    interaction += depletion * self.current_level / 100
            
            # Acute stress also affects but less
            if cortisol.current_level > 70:
                interaction -= 0.005 * (cortisol.current_level - 70) / 100
        
        # Oxytocin synergy (social bonding enhances well-being)
        if 'oxytocin' in other_hormones:
            oxytocin = other_hormones['oxytocin']
            if oxytocin.current_level > oxytocin.baseline:
                boost = 0.01 * (oxytocin.current_level - oxytocin.baseline)
                interaction += boost
        
        # Dopamine competition (can't be fully content while seeking)
        if 'dopamine' in other_hormones:
            dopamine = other_hormones['dopamine']
            # High dopamine seeking suppresses serotonin slightly
            if dopamine.current_level > dopamine.baseline + 30:
                suppression = -0.005 * (dopamine.current_level - dopamine.baseline - 30)
                interaction += suppression
        
        return interaction
    
    def get_stability_influence(self, other_hormones: Dict[str, BaseHormone]) -> Dict[str, float]:
        """
        Calculate serotonin's stabilizing influence on other hormones
        
        High serotonin dampens volatility in all other systems
        
        Returns:
            Dictionary of damping factors for each hormone
        """
        influences = {}
        
        # Only provide stability when serotonin is healthy
        if self.current_level > 40:
            # Calculate damping based on serotonin level
            damping_strength = min(0.5, (self.current_level - 40) / 100)
            
            for hormone_name in other_hormones:
                # Serotonin dampens all hormone volatility
                influences[hormone_name] = -damping_strength * 0.1
        
        return influences
    
    def update(self, dt: float, event: Optional[Dict] = None,
               other_hormones: Optional[Dict] = None) -> None:
        """
        Extended update with consistency tracking and memory influence
        """
        # Regular update
        super().update(dt, event, other_hormones)
        
        # Update consistency score based on recent successes
        if len(self.recent_successes) > 5:
            recent_times = [s['time'] for s in self.recent_successes]
            recent_mags = [s['magnitude'] for s in self.recent_successes]
            
            # Calculate variance in success
            if len(recent_mags) > 1:
                variance = np.var(recent_mags)
                # Lower variance = higher consistency
                self.consistency_score = 1.0 / (1.0 + variance)
            
            # Apply consistency bonus to baseline
            if self.consistency_score > 0.7:
                consistency_boost = self.accumulation_rate * self.consistency_score * dt
                self.current_level += consistency_boost
        
        # Calculate memory influence
        current_time = time.time()
        memory_boost = 0.0
        for memory in self.positive_memories:
            age = current_time - memory['time']
            if age < 3600:  # Memories within last hour
                # Recent memories have more influence
                influence = memory['strength'] * np.exp(-age / 1800)
                memory_boost += influence * 0.01
        
        self.memory_influence = min(0.1, memory_boost)
        self.current_level += self.memory_influence * dt
        
        # Update life satisfaction (very slow moving average)
        self.life_satisfaction = 0.999 * self.life_satisfaction + 0.001 * (self.current_level / 100)
        
        # Update confidence based on success rate
        if len(self.recent_successes) > 0:
            recent_rate = len([s for s in self.recent_successes if s['magnitude'] > 0.5]) / len(self.recent_successes)
            self.confidence_level = 0.9 * self.confidence_level + 0.1 * recent_rate
        
        # Check if depleted
        self.is_depleted = self.current_level < 30 or self.life_satisfaction < 0.3
        
        # Clean old depletion factors
        self.depletion_factors = [d for d in self.depletion_factors 
                                  if current_time - d['time'] < 1800]
    
    def get_confidence_modifier(self) -> float:
        """
        Get confidence modifier for decision making
        
        Returns:
            Confidence modifier (0.5 - 1.5)
        """
        base_confidence = self.confidence_level
        
        # Boost from high serotonin
        if self.current_level > 60:
            serotonin_boost = (self.current_level - 60) / 40 * 0.3
            base_confidence += serotonin_boost
        
        # Boost from wisdom
        base_confidence += self.wisdom_score * 0.2
        
        # Penalty if depleted
        if self.is_depleted:
            base_confidence *= 0.7
        
        return np.clip(base_confidence, 0.5, 1.5)
    
    def get_state_summary(self) -> Dict:
        """Extended state summary with serotonin-specific info"""
        summary = super().get_state_summary()
        summary.update({
            'success_streak': self.success_streak,
            'consistency_score': round(self.consistency_score, 3),
            'life_satisfaction': round(self.life_satisfaction, 3),
            'confidence_level': round(self.confidence_level, 3),
            'wisdom_score': round(self.wisdom_score, 3),
            'memory_influence': round(self.memory_influence, 3),
            'confidence_modifier': round(self.get_confidence_modifier(), 2),
            'is_depleted': self.is_depleted,
            'positive_memory_count': len(self.positive_memories),
            'depletion_factor_count': len(self.depletion_factors)
        })
        return summary
