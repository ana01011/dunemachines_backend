"""
Dopamine hormone implementation - Reward and motivation
"""
from typing import Dict, Optional
import numpy as np
import time

from .base_hormone import BaseHormone
from ..core.constants import EventType

class Dopamine(BaseHormone):
    """
    Dopamine: The reward and motivation hormone
    
    Characteristics:
    - Spikes with success and anticipation
    - Crashes below baseline after spikes (opponent process)
    - Builds tolerance over time
    - Regulated by cortisol when baseline > 50
    """
    
    def __init__(self, config: Dict):
        """Initialize dopamine with specific configuration"""
        super().__init__('dopamine', config)
        
        # Dopamine-specific parameters
        self.crash_factor = float(config.get('crash_factor', 0.4))
        self.tolerance = 0.0  # Builds with repeated rewards
        self.tolerance_buildup = float(config.get('tolerance_buildup', 0.01))
        
        # Reward prediction
        self.expected_reward = 0.5
        self.reward_history = []
        
        # Post-spike state
        self.post_spike_crash_pending = False
        self.spike_peak = 0.0
        
    def calculate_response(self, event_type: str, magnitude: float,
                          context: Optional[Dict] = None) -> float:
        """
        Calculate dopamine response to events
        
        Dopamine responds strongly to:
        - Success (especially unexpected)
        - Anticipation of reward
        - Novelty
        
        Dopamine drops with:
        - Failure
        - Disappointment (expected reward not received)
        """
        response = 0.0
        
        if event_type == EventType.TASK_SUCCESS:
            # Reward prediction error is key
            actual_reward = magnitude
            expected = context.get('expected', self.expected_reward) if context else self.expected_reward
            
            # RPE = actual - expected
            rpe = actual_reward - expected
            
            # Response is combination of absolute reward and prediction error
            response = self.sensitivity * (0.4 * actual_reward + 0.6 * max(0, rpe))
            
            # Apply tolerance (repeated rewards have less impact)
            response *= (1 - self.tolerance)
            
            # Update expectation for next time
            self.expected_reward = 0.7 * self.expected_reward + 0.3 * actual_reward
            
            # Build tolerance
            self.tolerance = min(0.5, self.tolerance + self.tolerance_buildup)
            
            # Mark for post-spike crash
            if response > 20:
                self.post_spike_crash_pending = True
                self.spike_peak = self.current_level + response
                
        elif event_type == EventType.ANTICIPATION:
            # Anticipation creates smaller rise
            response = self.sensitivity * magnitude * 0.3
            
        elif event_type == EventType.TASK_FAILURE:
            # Failure causes drop
            expected = context.get('expected', 0.5) if context else 0.5
            
            # Bigger drop if we expected success
            disappointment_factor = 1 + expected
            response = -self.crash_factor * magnitude * disappointment_factor
            
        elif event_type == EventType.NOVELTY:
            # Novel situations trigger curiosity (mild dopamine)
            response = self.sensitivity * magnitude * 0.25
            
            # Reset some tolerance for new experiences
            self.tolerance *= 0.9
            
        elif event_type == EventType.ROUTINE:
            # Routine tasks barely register
            response = self.sensitivity * magnitude * 0.1 * (1 - self.tolerance)
            
        return response
    
    def calculate_interaction(self, other_hormones: Dict[str, BaseHormone]) -> float:
        """
        Calculate interactions with other hormones
        
        Dopamine is:
        - Suppressed by cortisol (stress inhibits reward)
        - Enhanced by adrenaline (excitement amplifies reward)
        - Stabilized by serotonin (contentment reduces seeking)
        """
        interaction = 0.0
        
        # Cortisol suppression
        if 'cortisol' in other_hormones:
            cortisol = other_hormones['cortisol']
            # Cortisol above baseline suppresses dopamine
            if cortisol.current_level > cortisol.baseline:
                suppression = -0.02 * (cortisol.current_level - cortisol.baseline)
                # Stronger suppression if dopamine baseline is high (regulation)
                if self.baseline > 50:
                    suppression *= 1.5
                interaction += suppression * self.current_level / 100
        
        # Adrenaline enhancement
        if 'adrenaline' in other_hormones:
            adrenaline = other_hormones['adrenaline']
            if adrenaline.current_level > adrenaline.baseline:
                enhancement = 0.01 * (adrenaline.current_level - adrenaline.baseline)
                interaction += enhancement * self.current_level / 100
        
        # Serotonin stabilization
        if 'serotonin' in other_hormones:
            serotonin = other_hormones['serotonin']
            # High serotonin reduces dopamine volatility
            if serotonin.current_level > 60:
                damping = -0.005 * (serotonin.current_level - 60)
                interaction += damping * abs(self.current_level - self.baseline) / 50
        
        return interaction
    
    def update(self, dt: float, event: Optional[Dict] = None,
               other_hormones: Optional[Dict] = None) -> None:
        """
        Extended update to handle post-spike crashes
        """
        # Regular update
        super().update(dt, event, other_hormones)
        
        # Handle post-spike crash (opponent process)
        if self.post_spike_crash_pending:
            # Check if we've peaked and starting to fall
            if self.current_level < self.previous_level and \
               self.current_level < self.spike_peak - 5:
                # Initiate crash below baseline
                crash_magnitude = (self.spike_peak - self.baseline) * self.crash_factor
                self.current_level = self.baseline - crash_magnitude
                
                # Record wave amplitude
                self.wave_amplitudes.append(self.spike_peak - self.baseline)
                
                # Reset crash flag
                self.post_spike_crash_pending = False
                self.spike_peak = 0.0
                
                # Reduce tolerance slightly after crash
                self.tolerance *= 0.95
    
    def update_baseline(self, wave_analysis_result: Optional[float] = None) -> float:
        """
        Special baseline update with cortisol brake mechanism
        """
        if len(self.wave_amplitudes) < 3:
            return 0.0
        
        # Calculate wave deltas
        recent_waves = list(self.wave_amplitudes)[-10:]
        deltas = [recent_waves[i] - recent_waves[i-1] 
                 for i in range(1, len(recent_waves))]
        
        avg_delta = np.mean(deltas) if deltas else 0.0
        
        # CRITICAL: Cortisol brake mechanism at baseline 50
        if self.baseline >= 50:
            # Above 50, reverse positive changes to prevent inflation
            if avg_delta > 0:
                avg_delta *= -1  # Force downward
                
        # Apply shift with strict limits
        shift = avg_delta * 0.1  # Slow adaptation
        shift = np.clip(shift, -2.0, 2.0)
        
        # Apply shift
        self.baseline += shift
        self.baseline = np.clip(self.baseline, 10, 90)
        
        # Record
        self.baseline_history.append({
            'time': time.time(),
            'baseline': self.baseline,
            'shift': shift,
            'regulated': self.baseline >= 50
        })
        
        return shift
    
    def get_state_summary(self) -> Dict:
        """Extended state summary with dopamine-specific info"""
        summary = super().get_state_summary()
        summary.update({
            'tolerance': round(self.tolerance, 3),
            'expected_reward': round(self.expected_reward, 2),
            'post_spike_crash_pending': self.post_spike_crash_pending,
            'baseline_regulated': self.baseline >= 50
        })
        return summary
