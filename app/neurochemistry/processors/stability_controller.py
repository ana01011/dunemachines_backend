"""
Stability controller to prevent pathological states
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import logging
from datetime import datetime

from ..core.constants import (
    HORMONE_MIN, HORMONE_MAX, 
    UPPER_SOFT_BOUND, LOWER_SOFT_BOUND,
    MAX_EIGENVALUE
)
from ..hormones import BaseHormone

logger = logging.getLogger(__name__)

class StabilityController:
    """
    Ensures neurochemical system stability and prevents pathological states
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize stability controller
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        
        # Stability parameters
        self.max_change_per_update = 10.0
        self.oscillation_damping = 0.7
        self.extreme_threshold = 85
        self.low_threshold = 15
        
        # State tracking
        self.oscillation_detector = {
            hormone: deque(maxlen=10)
            for hormone in self.state.hormones
        }
        
        self.extreme_state_counter = 0
        self.intervention_history = deque(maxlen=100)
        
        # System health metrics
        self.system_energy = 0.0
        self.system_entropy = 0.0
        self.stability_score = 1.0
        
    def check_and_correct(self) -> Dict[str, any]:
        """
        Check system stability and apply corrections if needed
        
        Returns:
            Dictionary of interventions applied
        """
        interventions = {}
        
        # Check each hormone
        for hormone_name, hormone in self.state.hormones.items():
            # Check bounds
            if self._check_bounds(hormone):
                interventions[f"{hormone_name}_bounds"] = "corrected"
                
            # Check oscillation
            if self._check_oscillation(hormone_name, hormone):
                interventions[f"{hormone_name}_oscillation"] = "damped"
                
            # Check extremes
            if self._check_extreme_state(hormone):
                interventions[f"{hormone_name}_extreme"] = "moderated"
        
        # Check system-wide stability
        system_intervention = self._check_system_stability()
        if system_intervention:
            interventions['system'] = system_intervention
            
        # Record interventions
        if interventions:
            self.intervention_history.append({
                'time': datetime.now(),
                'interventions': interventions,
                'stability_score': self.stability_score
            })
            logger.warning(f"Stability interventions applied: {interventions}")
            
        return interventions
    
    def _check_bounds(self, hormone: BaseHormone) -> bool:
        """
        Check and correct hormone bounds
        
        Args:
            hormone: Hormone instance
            
        Returns:
            True if correction was applied
        """
        corrected = False
        
        # Hard bounds check
        if hormone.current_level > HORMONE_MAX - 0.1:
            hormone.current_level = HORMONE_MAX - 0.1
            corrected = True
            logger.debug(f"{hormone.name} hit upper bound")
            
        elif hormone.current_level < HORMONE_MIN + 0.1:
            hormone.current_level = HORMONE_MIN + 0.1
            corrected = True
            logger.debug(f"{hormone.name} hit lower bound")
            
        # Soft bounds with compression
        elif hormone.current_level > UPPER_SOFT_BOUND:
            excess = hormone.current_level - UPPER_SOFT_BOUND
            compressed = UPPER_SOFT_BOUND + (20 * np.tanh(excess / 20))
            if compressed < hormone.current_level:
                hormone.current_level = compressed
                corrected = True
                
        elif hormone.current_level < LOWER_SOFT_BOUND:
            deficit = LOWER_SOFT_BOUND - hormone.current_level
            expanded = LOWER_SOFT_BOUND - (20 * np.tanh(deficit / 20))
            if expanded > hormone.current_level:
                hormone.current_level = expanded
                corrected = True
                
        return corrected
    
    def _check_oscillation(self, hormone_name: str, hormone: BaseHormone) -> bool:
        """
        Check and dampen oscillations
        
        Args:
            hormone_name: Name of hormone
            hormone: Hormone instance
            
        Returns:
            True if damping was applied
        """
        # Record current level
        self.oscillation_detector[hormone_name].append(hormone.current_level)
        
        if len(self.oscillation_detector[hormone_name]) < 4:
            return False
            
        # Check for oscillation pattern
        levels = list(self.oscillation_detector[hormone_name])
        differences = [levels[i] - levels[i-1] for i in range(1, len(levels))]
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(differences)):
            if differences[i] * differences[i-1] < 0:  # Different signs
                sign_changes += 1
                
        # If oscillating (3+ sign changes in recent history)
        if sign_changes >= 3:
            # Apply damping
            amplitude = abs(hormone.current_level - hormone.baseline)
            damped_amplitude = amplitude * self.oscillation_damping
            
            # Move toward baseline with damped amplitude
            if hormone.current_level > hormone.baseline:
                hormone.current_level = hormone.baseline + damped_amplitude
            else:
                hormone.current_level = hormone.baseline - damped_amplitude
                
            logger.debug(f"Damped oscillation in {hormone_name}")
            return True
            
        return False
    
    def _check_extreme_state(self, hormone: BaseHormone) -> bool:
        """
        Check and moderate extreme states
        
        Args:
            hormone: Hormone instance
            
        Returns:
            True if moderation was applied
        """
        moderated = False
        
        # Check for extreme high
        if hormone.current_level > self.extreme_threshold:
            # Apply pressure toward baseline
            pressure = (hormone.current_level - self.extreme_threshold) * 0.1
            hormone.current_level -= pressure
            self.extreme_state_counter += 1
            moderated = True
            logger.debug(f"{hormone.name} extreme high moderated")
            
        # Check for extreme low
        elif hormone.current_level < self.low_threshold:
            # Apply pressure toward baseline
            pressure = (self.low_threshold - hormone.current_level) * 0.1
            hormone.current_level += pressure
            self.extreme_state_counter += 1
            moderated = True
            logger.debug(f"{hormone.name} extreme low moderated")
        else:
            # Reset counter if not extreme
            self.extreme_state_counter = max(0, self.extreme_state_counter - 1)
            
        return moderated
    
    def _check_system_stability(self) -> Optional[str]:
        """
        Check overall system stability
        
        Returns:
            Intervention type if needed, None otherwise
        """
        # Calculate system energy
        total_energy = 0
        for hormone in self.state.hormones.values():
            total_energy += hormone.current_level
            total_energy += (hormone.baseline ** 2) / 100
            
        self.system_energy = total_energy
        
        # Calculate system entropy (disorder)
        levels = [h.current_level for h in self.state.hormones.values()]
        if levels:
            # Normalize levels
            total = sum(levels)
            if total > 0:
                probs = [l/total for l in levels]
                # Calculate entropy
                self.system_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        # Calculate stability score
        energy_factor = 1.0 / (1.0 + abs(self.system_energy - 500) / 500)
        entropy_factor = 1.0 / (1.0 + abs(self.system_entropy - 1.5))
        extreme_factor = 1.0 / (1.0 + self.extreme_state_counter / 10)
        
        self.stability_score = (energy_factor + entropy_factor + extreme_factor) / 3
        
        # Check if intervention needed
        if self.stability_score < 0.3:
            # System is unstable, apply global dampening
            self._apply_global_dampening(0.8)
            return "global_dampening"
            
        elif self.extreme_state_counter > 10:
            # Too many extremes, reset toward baseline
            self._reset_toward_baseline(0.2)
            return "baseline_reset"
            
        elif self.system_energy > 700:
            # System too energetic, apply cooling
            self._apply_global_dampening(0.9)
            return "energy_cooling"
            
        return None
    
    def _apply_global_dampening(self, factor: float):
        """
        Apply dampening to all hormones
        
        Args:
            factor: Dampening factor (0-1)
        """
        for hormone in self.state.hormones.values():
            amplitude = hormone.current_level - hormone.baseline
            hormone.current_level = hormone.baseline + (amplitude * factor)
            
        logger.info(f"Applied global dampening: factor={factor:.2f}")
    
    def _reset_toward_baseline(self, strength: float):
        """
        Reset hormones toward baseline
        
        Args:
            strength: Reset strength (0-1)
        """
        for hormone in self.state.hormones.values():
            diff = hormone.baseline - hormone.current_level
            hormone.current_level += diff * strength
            
        logger.info(f"Reset toward baseline: strength={strength:.2f}")
        self.extreme_state_counter = 0
    
    def predict_stability(self, time_horizon: float = 60.0) -> float:
        """
        Predict future stability
        
        Args:
            time_horizon: Seconds to look ahead
            
        Returns:
            Predicted stability score
        """
        # Simple prediction based on current trajectory
        current_score = self.stability_score
        
        # Factor in recent interventions
        recent_interventions = sum(1 for i in self.intervention_history 
                                 if (datetime.now() - i['time']).total_seconds() < 300)
        
        # More interventions = less stable future
        intervention_factor = 1.0 / (1.0 + recent_interventions * 0.1)
        
        # Factor in extreme states
        extreme_factor = 1.0 / (1.0 + self.extreme_state_counter * 0.05)
        
        predicted = current_score * intervention_factor * extreme_factor
        
        return max(0.1, min(1.0, predicted))
    
    def get_stability_report(self) -> Dict:
        """Get comprehensive stability report"""
        report = {
            'stability_score': round(self.stability_score, 3),
            'system_energy': round(self.system_energy, 1),
            'system_entropy': round(self.system_entropy, 3),
            'extreme_state_count': self.extreme_state_counter,
            'recent_interventions': len([i for i in self.intervention_history
                                       if (datetime.now() - i['time']).total_seconds() < 300]),
            'predicted_stability': round(self.predict_stability(), 3),
            'hormone_states': {}
        }
        
        # Add individual hormone stability
        for hormone_name, hormone in self.state.hormones.items():
            distance_from_baseline = abs(hormone.current_level - hormone.baseline)
            is_extreme = hormone.current_level > self.extreme_threshold or \
                        hormone.current_level < self.low_threshold
                        
            report['hormone_states'][hormone_name] = {
                'stable': distance_from_baseline < 20,
                'extreme': is_extreme,
                'distance_from_baseline': round(distance_from_baseline, 1)
            }
            
        return report
