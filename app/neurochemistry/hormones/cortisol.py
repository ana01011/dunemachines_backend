"""
Cortisol hormone implementation - Stress and attention
"""
from typing import Dict, Optional
import numpy as np
import time

from .base_hormone import BaseHormone
from ..core.constants import EventType

class Cortisol(BaseHormone):
    """
    Cortisol: The stress and attention hormone
    
    Characteristics:
    - Rises with uncertainty, failure, and prediction errors
    - Regulates dopamine when dopamine baseline > 50
    - Increases planning depth and attention to detail
    - Chronic elevation leads to burnout
    """
    
    def __init__(self, config: Dict):
        """Initialize cortisol with specific configuration"""
        super().__init__('cortisol', config)
        
        # Cortisol-specific parameters
        self.stress_threshold = float(config.get('stress_threshold', 50))
        self.brake_threshold = float(config.get('brake_threshold', 50))
        self.chronic_stress_level = 0.0  # Tracks long-term stress
        self.stress_accumulation_rate = 0.01
        self.stress_recovery_rate = 0.02
        
        # Regulation state
        self.is_regulating_dopamine = False
        self.regulation_strength = 0.0
        
        # Prediction error tracking
        self.recent_errors = []
        self.error_sensitivity = 0.3
        
        # Burnout protection
        self.burnout_risk = 0.0
        self.consecutive_high_stress = 0
        
    def calculate_response(self, event_type: str, magnitude: float,
                          context: Optional[Dict] = None) -> float:
        """
        Calculate cortisol response to events
        
        Cortisol responds to:
        - Failure and errors (stress response)
        - Uncertainty (vigilance)
        - High complexity (attention needed)
        - Time pressure (urgency)
        - Prediction errors (something unexpected)
        """
        response = 0.0
        
        if event_type == EventType.TASK_FAILURE:
            # Failure causes significant cortisol spike
            response = self.sensitivity * magnitude * 0.6
            
            # Larger spike if failure was unexpected
            if context and context.get('expected_success', 0) > 0.7:
                response *= 1.5
                
            # Track consecutive stress
            self.consecutive_high_stress += 1
            
        elif event_type == EventType.ERROR:
            # Errors cause immediate cortisol response
            response = self.sensitivity * magnitude * 0.7
            
            # Add to error tracking
            self.recent_errors.append({
                'time': time.time(),
                'magnitude': magnitude
            })
            
        elif event_type == EventType.UNCERTAINTY:
            # Uncertainty maintains elevated cortisol
            response = self.sensitivity * magnitude * 0.4
            
        elif event_type == EventType.HIGH_COMPLEXITY:
            # Complexity requires attention (moderate cortisol)
            response = self.sensitivity * magnitude * 0.35
            
        elif event_type == EventType.TIME_PRESSURE:
            # Urgency spikes cortisol
            urgency = context.get('urgency', 0.5) if context else 0.5
            response = self.sensitivity * (urgency or 0.5) * 0.5
            
        elif event_type == EventType.TASK_SUCCESS:
            # Success reduces cortisol (relief)
            response = -0.3 * magnitude
            
            # Reset stress counters
            self.consecutive_high_stress = max(0, self.consecutive_high_stress - 1)
            
        # Calculate prediction error component
        if context and 'expected' in context and 'actual' in context:
            error = abs(context['expected'] - context['actual'])
            response += self.error_sensitivity * error
            
        return response
    
    def calculate_interaction(self, other_hormones: Dict[str, BaseHormone]) -> float:
        """
        Calculate interactions with other hormones
        
        Cortisol:
        - Is triggered by high dopamine baseline (>50) for regulation
        - Increases with adrenaline (stress + urgency)
        - Is suppressed by serotonin (contentment reduces stress)
        - Is modulated by oxytocin (social bonding reduces stress)
        """
        interaction = 0.0
        
        # Dopamine regulation trigger
        if 'dopamine' in other_hormones:
            dopamine = other_hormones['dopamine']
            
            # If dopamine baseline is high, cortisol rises to regulate
            if dopamine.baseline > self.brake_threshold:
                # Proportional to how far above threshold
                excess = dopamine.baseline - self.brake_threshold
                regulation_boost = 0.02 * excess
                
                # Temporary baseline boost for regulation
                if not self.is_regulating_dopamine:
                    self.is_regulating_dopamine = True
                    self.regulation_strength = regulation_boost
                    # This will increase cortisol to suppress dopamine
                    interaction += regulation_boost * 50  # Strong response
                else:
                    # Maintain regulation
                    interaction += self.regulation_strength * 30
                    
            else:
                # No longer need to regulate
                if self.is_regulating_dopamine:
                    self.is_regulating_dopamine = False
                    self.regulation_strength = 0.0
                    # Cortisol can relax
                    interaction -= 10
        
        # Adrenaline amplification (stress + urgency = higher cortisol)
        if 'adrenaline' in other_hormones:
            adrenaline = other_hormones['adrenaline']
            if adrenaline.current_level > adrenaline.baseline:
                amplification = 0.015 * (adrenaline.current_level - adrenaline.baseline)
                interaction += amplification * self.current_level / 100
        
        # Serotonin suppression (contentment reduces stress)
        if 'serotonin' in other_hormones:
            serotonin = other_hormones['serotonin']
            if serotonin.current_level > 60:
                suppression = -0.01 * (serotonin.current_level - 60)
                interaction += suppression * self.current_level / 100
        
        # Oxytocin modulation (social support reduces stress)
        if 'oxytocin' in other_hormones:
            oxytocin = other_hormones['oxytocin']
            if oxytocin.current_level > oxytocin.baseline:
                relief = -0.008 * (oxytocin.current_level - oxytocin.baseline)
                interaction += relief * self.current_level / 100
        
        return interaction
    
    def update(self, dt: float, event: Optional[Dict] = None,
               other_hormones: Optional[Dict] = None) -> None:
        """
        Extended update with chronic stress tracking
        """
        # Regular update
        super().update(dt, event, other_hormones)
        
        # Update chronic stress level
        if self.current_level > self.stress_threshold:
            # Accumulate chronic stress
            self.chronic_stress_level += self.stress_accumulation_rate * dt
            self.chronic_stress_level = min(1.0, self.chronic_stress_level)
            
            # Track consecutive high stress
            if self.current_level > 70:
                self.consecutive_high_stress += dt
        else:
            # Recovery from chronic stress
            self.chronic_stress_level -= self.stress_recovery_rate * dt
            self.chronic_stress_level = max(0.0, self.chronic_stress_level)
            self.consecutive_high_stress = 0
        
        # Calculate burnout risk
        self.burnout_risk = self.chronic_stress_level * 0.7 + \
                           min(1.0, self.consecutive_high_stress / 100) * 0.3
        
        # Clean old errors
        current_time = time.time()
        self.recent_errors = [e for e in self.recent_errors 
                             if current_time - e['time'] < 300]  # Keep 5 minutes
    
    def update_baseline(self, wave_analysis_result: Optional[float] = None) -> float:
        """
        Cortisol baseline adapts to chronic stress patterns
        """
        # If chronically stressed, baseline creeps up
        if self.chronic_stress_level > 0.5:
            shift = 0.5 * self.chronic_stress_level  # Slow upward creep
        # If regulated dopamine successfully, baseline can drop
        elif self.is_regulating_dopamine and self.regulation_strength > 0:
            shift = -0.3  # Relax after regulation
        else:
            # Normal baseline adaptation
            shift = super()._calculate_baseline_shift()
        
        # Apply shift
        shift = np.clip(shift, -2.0, 2.0)
        self.baseline += shift
        self.baseline = np.clip(self.baseline, 10, 80)  # Cortisol baseline caps at 80
        
        return shift
    
    def get_attention_multiplier(self) -> float:
        """
        Get attention/planning depth multiplier based on cortisol
        
        Returns:
            Multiplier for planning depth (1.0 - 2.0)
        """
        # Cortisol increases attention and thoroughness
        if self.current_level > self.baseline:
            # More cortisol = more careful planning
            excess = min(40, self.current_level - self.baseline)
            return 1.0 + (excess / 40) * 1.0  # Up to 2x planning
        return 1.0
    
    def should_request_clarification(self) -> bool:
        """
        Determine if high uncertainty warrants requesting clarification
        
        Returns:
            True if should ask for clarification
        """
        # High cortisol + many recent errors = need clarification
        return (self.current_level > 60 and len(self.recent_errors) > 3) or \
               (self.burnout_risk > 0.7)
    
    def get_state_summary(self) -> Dict:
        """Extended state summary with cortisol-specific info"""
        summary = super().get_state_summary()
        summary.update({
            'chronic_stress': round(self.chronic_stress_level, 3),
            'burnout_risk': round(self.burnout_risk, 3),
            'is_regulating_dopamine': self.is_regulating_dopamine,
            'regulation_strength': round(self.regulation_strength, 3),
            'attention_multiplier': round(self.get_attention_multiplier(), 2),
            'recent_error_count': len(self.recent_errors),
            'should_clarify': self.should_request_clarification()
        })
        return summary
