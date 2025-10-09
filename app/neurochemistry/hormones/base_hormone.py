"""
Base class for all hormones in the neurochemical system
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import time

from ..core.constants import (
    HORMONE_MIN, HORMONE_MAX, BASELINE_MIN, BASELINE_MAX,
    UPPER_SOFT_BOUND, LOWER_SOFT_BOUND, MAX_WAVE_HISTORY
)

class BaseHormone(ABC):
    """
    Abstract base class for all hormones.
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, name: str, config: Dict):
        """
        Initialize a hormone with configuration
        
        Args:
            name: Hormone identifier
            config: Configuration dictionary from YAML
        """
        self.name = name
        
        # Current state
        self.current_level = float(config.get('initial_level', 40.0))
        self.baseline = float(config.get('initial_baseline', 40.0))
        self.previous_level = self.current_level
        
        # Configuration
        self.decay_rate = float(config.get('decay_rate', 0.1))
        self.sensitivity = float(config.get('sensitivity', 1.0))
        
        # Bounds
        self.min_value = HORMONE_MIN
        self.max_value = HORMONE_MAX
        self.baseline_min = BASELINE_MIN
        self.baseline_max = BASELINE_MAX
        
        # History tracking
        self.history = deque(maxlen=1000)  # Recent levels
        self.wave_amplitudes = deque(maxlen=MAX_WAVE_HISTORY)  # Recent spike amplitudes
        self.baseline_history = deque(maxlen=100)  # Baseline changes
        
        # Timing
        self.last_update = time.time()
        self.last_spike_time = 0
        self.last_crash_time = 0
        
        # State flags
        self.is_spiking = False
        self.is_crashing = False
        self.is_stable = True
        
    @abstractmethod
    def calculate_response(self, event_type: str, magnitude: float, 
                          context: Optional[Dict] = None) -> float:
        """
        Calculate hormone response to an event
        
        Args:
            event_type: Type of event
            magnitude: Event magnitude (0-1)
            context: Optional context information
            
        Returns:
            Change in hormone level
        """
        pass
    
    @abstractmethod
    def calculate_interaction(self, other_hormones: Dict[str, 'BaseHormone']) -> float:
        """
        Calculate interaction effects from other hormones
        
        Args:
            other_hormones: Dictionary of other hormone instances
            
        Returns:
            Interaction effect on this hormone
        """
        pass
    
    def update(self, dt: float, event: Optional[Dict] = None, 
               other_hormones: Optional[Dict] = None) -> None:
        """
        Update hormone level based on dynamics
        
        Args:
            dt: Time delta in seconds
            event: Optional event dictionary
            other_hormones: Optional dictionary of other hormones
        """
        # Store previous level
        self.previous_level = self.current_level
        
        # Base decay toward baseline
        decay = -self.decay_rate * (self.current_level - self.baseline) * dt
        
        # Event response
        event_response = 0
        if event:
            event_response = self.calculate_response(
                event.get('type'),
                event.get('magnitude', 0.5),
                event.get('context')
            )
        
        # Hormone interactions
        interaction = 0
        if other_hormones:
            interaction = self.calculate_interaction(other_hormones)
        
        # Calculate total change
        total_change = decay + event_response + interaction
        
        # Apply change with stability check
        self.current_level += self._apply_stability_limit(total_change, dt)
        
        # Apply bounds
        self._apply_bounds()
        
        # Update state flags
        self._update_state_flags()
        
        # Record history
        self.history.append({
            'time': time.time(),
            'level': self.current_level,
            'baseline': self.baseline,
            'event': event.get('type') if event else None
        })
        
        self.last_update = time.time()
    
    def _apply_bounds(self) -> None:
        """Apply soft sigmoid bounds to hormone level"""
        if self.current_level > UPPER_SOFT_BOUND:
            # Soft compression above 80
            excess = self.current_level - UPPER_SOFT_BOUND
            compressed = UPPER_SOFT_BOUND + (HORMONE_MAX - UPPER_SOFT_BOUND) * np.tanh(excess / 20)
            self.current_level = min(compressed, HORMONE_MAX - 0.01)
            
        elif self.current_level < LOWER_SOFT_BOUND:
            # Soft expansion below 20
            deficit = LOWER_SOFT_BOUND - self.current_level
            expanded = LOWER_SOFT_BOUND - LOWER_SOFT_BOUND * np.tanh(deficit / 20)
            self.current_level = max(expanded, HORMONE_MIN + 0.01)
        
        # Hard bounds as final check
        self.current_level = np.clip(self.current_level, self.min_value, self.max_value)
        
    def _apply_stability_limit(self, change: float, dt: float) -> float:
        """
        Limit rate of change for stability
        
        Args:
            change: Proposed change
            dt: Time delta
            
        Returns:
            Limited change value
        """
        max_change = 50 * dt  # Max 50 points per second
        return np.clip(change, -max_change, max_change)
    
    def _update_state_flags(self) -> None:
        """Update state flags based on current level and history"""
        # Check if spiking (rising rapidly above baseline)
        if self.current_level > self.baseline + 15 and \
           self.current_level > self.previous_level + 5:
            self.is_spiking = True
            self.is_crashing = False
            self.last_spike_time = time.time()
            
        # Check if crashing (falling rapidly below baseline)
        elif self.current_level < self.baseline - 15 and \
             self.current_level < self.previous_level - 5:
            self.is_crashing = True
            self.is_spiking = False
            self.last_crash_time = time.time()
            
        else:
            self.is_spiking = False
            self.is_crashing = False
        
        # Check stability (near baseline)
        self.is_stable = abs(self.current_level - self.baseline) < 10
    
    def update_baseline(self, wave_analysis_result: Optional[float] = None) -> float:
        """
        Update baseline based on wave analysis
        
        Args:
            wave_analysis_result: Optional external wave analysis
            
        Returns:
            Baseline shift amount
        """
        if wave_analysis_result is not None:
            shift = wave_analysis_result
        else:
            shift = self._calculate_baseline_shift()
        
        # Apply shift with limits
        shift = np.clip(shift, -2.0, 2.0)
        self.baseline += shift
        
        # Keep baseline in bounds
        self.baseline = np.clip(self.baseline, self.baseline_min, self.baseline_max)
        
        # Record baseline change
        self.baseline_history.append({
            'time': time.time(),
            'baseline': self.baseline,
            'shift': shift
        })
        
        return shift
    
    def _calculate_baseline_shift(self) -> float:
        """
        Calculate baseline shift from wave history
        
        Returns:
            Suggested baseline shift
        """
        if len(self.wave_amplitudes) < 3:
            return 0.0
        
        # Get recent wave amplitudes
        recent_waves = list(self.wave_amplitudes)[-10:]
        
        # Calculate trend
        if len(recent_waves) >= 2:
            deltas = [recent_waves[i] - recent_waves[i-1] 
                     for i in range(1, len(recent_waves))]
            avg_delta = np.mean(deltas)
            
            # Scale by baseline adaptation rate
            return avg_delta * 0.1
        
        return 0.0
    
    def get_amplitude(self) -> float:
        """Get current amplitude (distance from baseline)"""
        return self.current_level - self.baseline
    
    def get_state_summary(self) -> Dict:
        """Get summary of current hormone state"""
        return {
            'name': self.name,
            'level': round(self.current_level, 2),
            'baseline': round(self.baseline, 2),
            'amplitude': round(self.get_amplitude(), 2),
            'is_spiking': self.is_spiking,
            'is_crashing': self.is_crashing,
            'is_stable': self.is_stable,
            'sensitivity': self.sensitivity
        }
    
    def reset(self) -> None:
        """Reset hormone to baseline state"""
        self.current_level = self.baseline
        self.previous_level = self.baseline
        self.is_spiking = False
        self.is_crashing = False
        self.is_stable = True
