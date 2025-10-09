"""
Baseline adaptation processor for dynamic hormone baselines
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import logging
from datetime import datetime, timedelta

from ..core.constants import Hormone, MIN_WAVES_FOR_BASELINE_SHIFT
from ..hormones import BaseHormone

logger = logging.getLogger(__name__)

class BaselineAdapter:
    """
    Manages baseline adaptation for all hormones
    Implements the wave-based baseline shift algorithm
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize baseline adapter
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        
        # Wave tracking for each hormone
        self.wave_history = {
            hormone: deque(maxlen=20)
            for hormone in Hormone
        }
        
        # Baseline shift history
        self.shift_history = {
            hormone: deque(maxlen=50)
            for hormone in Hormone
        }
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.min_amplitude_for_wave = 5.0
        self.adaptation_interval = 30.0  # seconds
        self.last_adaptation_time = datetime.now()
        
        # Regulation tracking
        self.regulation_active = {
            Hormone.DOPAMINE: False,
            Hormone.CORTISOL: False
        }
        
    def should_adapt(self) -> bool:
        """
        Check if baselines should be adapted
        
        Returns:
            True if adaptation should occur
        """
        time_since = (datetime.now() - self.last_adaptation_time).total_seconds()
        return time_since >= self.adaptation_interval
    
    def adapt_all(self) -> Dict[str, float]:
        """
        Adapt baselines for all hormones
        
        Returns:
            Dictionary of baseline shifts
        """
        shifts = {}
        
        for hormone_name, hormone in self.state.hormones.items():
            shift = self.adapt_baseline(hormone_name, hormone)
            shifts[hormone_name] = shift
            
        self.last_adaptation_time = datetime.now()
        
        # Handle special regulation mechanisms
        self._handle_dopamine_cortisol_regulation()
        
        logger.info(f"Baseline adaptation complete: {shifts}")
        return shifts
    
    def adapt_baseline(self, hormone_name: str, hormone: BaseHormone) -> float:
        """
        Adapt baseline for a single hormone
        
        Args:
            hormone_name: Name of hormone
            hormone: Hormone instance
            
        Returns:
            Baseline shift amount
        """
        # Detect and record waves
        self._detect_waves(hormone_name, hormone)
        
        # Calculate baseline shift using wave analysis
        shift = self._calculate_wave_based_shift(hormone_name, hormone)
        
        # Apply hormone-specific rules
        shift = self._apply_hormone_specific_rules(hormone_name, hormone, shift)
        
        # Apply the shift
        if abs(shift) > 0.01:
            hormone.update_baseline(shift)
            
            # Record shift
            self.shift_history[hormone_name].append({
                'time': datetime.now(),
                'shift': shift,
                'baseline': hormone.baseline,
                'reason': self._get_shift_reason(hormone_name, shift)
            })
            
        return shift
    
    def _detect_waves(self, hormone_name: str, hormone: BaseHormone):
        """
        Detect and record hormone waves (spikes/crashes)
        
        Args:
            hormone_name: Name of hormone
            hormone: Hormone instance
        """
        amplitude = hormone.get_amplitude()
        
        # Check if this is a significant wave
        if abs(amplitude) > self.min_amplitude_for_wave:
            # Check if this is a peak or trough
            if hormone.is_spiking or hormone.is_crashing:
                wave_data = {
                    'time': datetime.now(),
                    'amplitude': amplitude,
                    'type': 'spike' if hormone.is_spiking else 'crash',
                    'level': hormone.current_level,
                    'baseline': hormone.baseline
                }
                
                # Only add if different from last wave
                if hormone_name in self.wave_history:
                    if not self.wave_history[hormone_name] or \
                       abs(amplitude - self.wave_history[hormone_name][-1]['amplitude']) > 2:
                        self.wave_history[hormone_name].append(wave_data)
    
    def _calculate_wave_based_shift(self, hormone_name: str, hormone: BaseHormone) -> float:
        """
        Calculate baseline shift using wave analysis algorithm
        
        Args:
            hormone_name: Name of hormone
            hormone: Hormone instance
            
        Returns:
            Suggested baseline shift
        """
        waves = self.wave_history.get(hormone_name, [])
        
        if len(waves) < MIN_WAVES_FOR_BASELINE_SHIFT:
            return 0.0
        
        # Get recent wave amplitudes
        recent_waves = list(waves)[-10:]
        amplitudes = [w['amplitude'] for w in recent_waves]
        
        # Calculate deltas between consecutive waves
        deltas = []
        for i in range(1, len(amplitudes)):
            delta = amplitudes[i] - amplitudes[i-1]
            deltas.append(delta)
        
        if not deltas:
            return 0.0
        
        # Average delta indicates trend
        avg_delta = np.mean(deltas)
        
        # Scale by adaptation rate
        shift = avg_delta * self.adaptation_rate
        
        logger.debug(f"{hormone_name} wave analysis: avg_delta={avg_delta:.3f}, shift={shift:.3f}")
        
        return shift
    
    def _apply_hormone_specific_rules(self, hormone_name: str, 
                                     hormone: BaseHormone, shift: float) -> float:
        """
        Apply hormone-specific baseline adaptation rules
        
        Args:
            hormone_name: Name of hormone
            hormone: Hormone instance
            shift: Proposed baseline shift
            
        Returns:
            Modified shift amount
        """
        if hormone_name == 'dopamine':
            # Dopamine regulation at baseline 50
            if hormone.baseline >= 50:
                # Above 50, reverse positive shifts (cortisol brake)
                if shift > 0:
                    shift *= -1
                    self.regulation_active[Hormone.DOPAMINE] = True
                    logger.info(f"Dopamine regulation active: baseline={hormone.baseline:.1f}")
            else:
                self.regulation_active[Hormone.DOPAMINE] = False
                
        elif hormone_name == 'cortisol':
            # Cortisol responds to dopamine regulation
            if hasattr(self.state.hormones.get('dopamine'), 'baseline'):
                dopamine = self.state.hormones['dopamine']
                if dopamine.baseline > 50:
                    # Boost cortisol baseline temporarily
                    regulation_boost = (dopamine.baseline - 50) * 0.02
                    shift += regulation_boost
                    self.regulation_active[Hormone.CORTISOL] = True
                elif self.regulation_active[Hormone.CORTISOL]:
                    # Relax cortisol after regulation
                    shift -= 0.5
                    self.regulation_active[Hormone.CORTISOL] = False
                    
        elif hormone_name == 'serotonin':
            # Serotonin baseline rises with consistency
            if hasattr(hormone, 'consistency_score'):
                if hormone.consistency_score > 0.7:
                    shift += 0.2  # Bonus for consistency
                    
        elif hormone_name == 'adrenaline':
            # Adrenaline baseline adapts to activity level
            if hasattr(hormone, 'exhaustion_level'):
                if hormone.exhaustion_level > 0.5:
                    shift -= 0.3  # Lower baseline when exhausted
        
        # Apply limits
        shift = np.clip(shift, -2.0, 2.0)
        
        return shift
    
    def _handle_dopamine_cortisol_regulation(self):
        """
        Handle the special dopamine-cortisol regulation mechanism
        """
        dopamine = self.state.hormones.get('dopamine')
        cortisol = self.state.hormones.get('cortisol')
        
        if not dopamine or not cortisol:
            return
        
        # Check if regulation needed
        if dopamine.baseline > 50:
            # Calculate regulation strength
            excess = dopamine.baseline - 50
            regulation_strength = min(1.0, excess / 40)
            
            # Boost cortisol to regulate dopamine
            if not cortisol.is_regulating_dopamine:
                cortisol.baseline += regulation_strength * 10
                cortisol.is_regulating_dopamine = True
                logger.info(f"Cortisol regulation activated: strength={regulation_strength:.2f}")
                
        elif cortisol.is_regulating_dopamine:
            # Regulation no longer needed, relax cortisol
            cortisol.baseline *= 0.9
            cortisol.is_regulating_dopamine = False
            logger.info("Cortisol regulation deactivated")
    
    def _get_shift_reason(self, hormone_name: str, shift: float) -> str:
        """
        Get human-readable reason for baseline shift
        
        Args:
            hormone_name: Name of hormone
            shift: Shift amount
            
        Returns:
            Reason string
        """
        if abs(shift) < 0.01:
            return "stable"
            
        if hormone_name == 'dopamine':
            if self.regulation_active.get(Hormone.DOPAMINE):
                return "cortisol_regulation"
            elif shift > 0:
                return "reward_seeking_increase"
            else:
                return "satisfaction_decrease"
                
        elif hormone_name == 'cortisol':
            if self.regulation_active.get(Hormone.CORTISOL):
                return "dopamine_regulation"
            elif shift > 0:
                return "chronic_stress_adaptation"
            else:
                return "stress_recovery"
                
        elif shift > 0:
            return f"{hormone_name}_adaptation_up"
        else:
            return f"{hormone_name}_adaptation_down"
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of baseline adaptation state"""
        summary = {
            'regulation_active': {
                'dopamine': self.regulation_active.get(Hormone.DOPAMINE, False),
                'cortisol': self.regulation_active.get(Hormone.CORTISOL, False)
            },
            'wave_counts': {},
            'recent_shifts': {},
            'time_until_next': max(0, self.adaptation_interval - 
                                  (datetime.now() - self.last_adaptation_time).total_seconds())
        }
        
        # Wave counts
        for hormone in Hormone:
            if hormone.value in self.wave_history:
                summary['wave_counts'][hormone.value] = len(self.wave_history[hormone])
        
        # Recent shifts
        for hormone in Hormone:
            if hormone.value in self.shift_history and self.shift_history[hormone.value]:
                recent = self.shift_history[hormone.value][-1]
                summary['recent_shifts'][hormone.value] = {
                    'shift': round(recent['shift'], 3),
                    'baseline': round(recent['baseline'], 1),
                    'reason': recent['reason']
                }
        
        return summary
