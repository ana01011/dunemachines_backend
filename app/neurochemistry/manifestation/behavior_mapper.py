"""
Maps neurochemical states to behavioral parameters
"""
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from datetime import datetime
import logging

from ..core.constants import (
    PLANNING_DEPTH_MIN, PLANNING_DEPTH_MAX,
    RISK_TOLERANCE_MIN, RISK_TOLERANCE_MAX,
    CONFIDENCE_MIN, CONFIDENCE_MAX,
    PROCESSING_SPEED_MIN, PROCESSING_SPEED_MAX
)

logger = logging.getLogger(__name__)

class BehaviorMapper:
    """
    Maps neurochemical states to behavioral outputs
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize behavior mapper
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        self.config = neurochemical_state.config.get('behavior', {})
        
        # Behavioral history for smoothing
        self.behavior_history = []
        self.history_window = 5
        
        # Current behavioral state
        self.current_behavior = {
            'planning_depth': 3,
            'risk_tolerance': 0.5,
            'processing_speed': 1.0,
            'confidence': 0.5,
            'creativity': 0.5,
            'empathy': 0.5,
            'patience': 0.5,
            'thoroughness': 0.5
        }
        
        # Behavioral tendencies based on chronic states
        self.tendencies = {
            'analytical': 0.5,  # High cortisol, high planning
            'impulsive': 0.5,   # High dopamine, low planning
            'creative': 0.5,    # Balanced with slight dopamine elevation
            'cautious': 0.5,    # High cortisol, low risk
            'adventurous': 0.5  # High dopamine, high risk
        }
        
    def get_current_parameters(self) -> Dict[str, float]:
        """
        Get current behavioral parameters based on neurochemical state
        
        Returns:
            Dictionary of behavioral parameters
        """
        # Get hormone states
        hormones = self.state.hormones
        dopamine = hormones.get('dopamine')
        cortisol = hormones.get('cortisol')
        adrenaline = hormones.get('adrenaline')
        serotonin = hormones.get('serotonin')
        oxytocin = hormones.get('oxytocin')
        
        # Calculate each behavioral parameter
        params = {}
        
        # Planning Depth
        params['planning_depth'] = self._calculate_planning_depth(
            dopamine, cortisol, adrenaline, serotonin
        )
        
        # Risk Tolerance
        params['risk_tolerance'] = self._calculate_risk_tolerance(
            dopamine, cortisol, serotonin
        )
        
        # Processing Speed
        params['processing_speed'] = self._calculate_processing_speed(
            adrenaline, cortisol
        )
        
        # Confidence
        params['confidence'] = self._calculate_confidence(
            serotonin, dopamine, cortisol
        )
        
        # Creativity
        params['creativity'] = self._calculate_creativity(
            dopamine, serotonin, cortisol
        )
        
        # Empathy
        params['empathy'] = self._calculate_empathy(
            oxytocin, serotonin
        )
        
        # Patience
        params['patience'] = self._calculate_patience(
            serotonin, cortisol, adrenaline
        )
        
        # Thoroughness
        params['thoroughness'] = self._calculate_thoroughness(
            cortisol, serotonin
        )
        
        # Additional behavioral flags
        params['should_explore'] = self._should_explore(dopamine, serotonin)
        params['should_exploit'] = self._should_exploit(dopamine, serotonin)
        params['needs_break'] = self._needs_break(cortisol, adrenaline)
        params['in_flow_state'] = self._in_flow_state(dopamine, cortisol, adrenaline)
        
        # Smooth parameters with history
        smoothed = self._smooth_parameters(params)
        
        # Update history
        self.behavior_history.append({
            'timestamp': datetime.now(),
            'parameters': smoothed.copy()
        })
        
        # Trim history
        if len(self.behavior_history) > self.history_window:
            self.behavior_history.pop(0)
        
        # Update current behavior
        self.current_behavior.update(smoothed)
        
        # Update tendencies
        self._update_tendencies()
        
        return smoothed
    
    def _calculate_planning_depth(self, dopamine, cortisol, adrenaline, serotonin) -> float:
        """
        Calculate planning depth based on hormones
        
        High cortisol → more planning
        High adrenaline → less planning (urgency)
        High dopamine → less planning (impulsivity)
        High serotonin → moderate planning (balanced)
        """
        base = self.config.get('planning_depth', {}).get('base', 3)
        
        # Cortisol increases planning
        cortisol_factor = 0
        if cortisol and hasattr(cortisol, 'get_attention_multiplier'):
            cortisol_factor = cortisol.get_attention_multiplier() - 1.0
        elif cortisol:
            cortisol_amp = cortisol.get_amplitude()
            if cortisol_amp > 0:
                cortisol_factor = cortisol_amp / 50 * 0.5  # Up to 50% increase
        
        # Dopamine decreases planning (impulsivity)
        dopamine_factor = 0
        if dopamine:
            dopamine_amp = dopamine.get_amplitude()
            if dopamine_amp > 20:  # Excited state
                dopamine_factor = -min(0.3, dopamine_amp / 100)
        
        # Adrenaline decreases planning (urgency)
        adrenaline_factor = 0
        if adrenaline:
            adrenaline_amp = adrenaline.get_amplitude()
            if adrenaline_amp > 10:
                adrenaline_factor = -min(0.4, adrenaline_amp / 80)
        
        # Serotonin provides stability
        serotonin_factor = 0
        if serotonin and serotonin.current_level > 60:
            # High serotonin prevents extremes
            total_factors = cortisol_factor + dopamine_factor + adrenaline_factor
            if abs(total_factors) > 0.5:
                serotonin_factor = -total_factors * 0.3  # Dampen extremes
        
        # Calculate final planning depth
        planning = base * (1 + cortisol_factor + dopamine_factor + 
                          adrenaline_factor + serotonin_factor)
        
        return np.clip(planning, PLANNING_DEPTH_MIN, PLANNING_DEPTH_MAX)
    
    def _calculate_risk_tolerance(self, dopamine, cortisol, serotonin) -> float:
        """
        Calculate risk tolerance based on hormones
        
        High dopamine → higher risk tolerance
        High cortisol → lower risk tolerance
        High serotonin → moderate, confident risk
        """
        base = self.config.get('risk_tolerance', {}).get('base', 0.5)
        
        risk = base
        
        # Dopamine increases risk-taking
        if dopamine:
            dopamine_amp = dopamine.get_amplitude()
            # Positive amplitude = seeking = more risk
            if dopamine_amp > 0:
                risk += min(0.3, dopamine_amp / 100)
            # Negative amplitude (below baseline) = conservative
            else:
                risk += max(-0.2, dopamine_amp / 100)
        
        # Cortisol decreases risk-taking
        if cortisol:
            cortisol_level = cortisol.current_level
            if cortisol_level > 50:
                # High stress = risk averse
                risk -= min(0.3, (cortisol_level - 50) / 100)
        
        # Serotonin provides confident risk-taking
        if serotonin and hasattr(serotonin, 'confidence_level'):
            if serotonin.confidence_level > 0.6:
                # Confident = calculated risks
                risk += 0.1
        
        return np.clip(risk, RISK_TOLERANCE_MIN, RISK_TOLERANCE_MAX)
    
    def _calculate_processing_speed(self, adrenaline, cortisol) -> float:
        """
        Calculate processing speed based on hormones
        
        High adrenaline → faster processing
        Very high adrenaline → reduced accuracy
        High cortisol → slower, more careful
        """
        base = self.config.get('processing_speed', {}).get('base', 1.0)
        
        speed = base
        
        # Adrenaline increases speed
        if adrenaline and hasattr(adrenaline, 'get_performance_modifiers'):
            perf_mods = adrenaline.get_performance_modifiers()
            speed = base * perf_mods['processing_speed']
        elif adrenaline:
            adrenaline_amp = adrenaline.get_amplitude()
            if adrenaline_amp > 0:
                speed *= (1 + min(0.5, adrenaline_amp / 100))
        
        # Cortisol can slow processing (careful thinking)
        if cortisol and cortisol.current_level > 60:
            # High stress can impair speed
            speed *= (1 - min(0.2, (cortisol.current_level - 60) / 100))
        
        return np.clip(speed, PROCESSING_SPEED_MIN, PROCESSING_SPEED_MAX)
    
    def _calculate_confidence(self, serotonin, dopamine, cortisol) -> float:
        """
        Calculate confidence based on hormones
        
        High serotonin → high confidence
        Recent successes (dopamine spikes) → confidence
        High cortisol → reduced confidence
        """
        base = self.config.get('confidence', {}).get('base', 0.5)
        
        confidence = base
        
        # Serotonin is primary confidence driver
        if serotonin and hasattr(serotonin, 'get_confidence_modifier'):
            confidence = base * serotonin.get_confidence_modifier()
        elif serotonin:
            if serotonin.current_level > 50:
                confidence += (serotonin.current_level - 50) / 100
        
        # Recent dopamine success boosts confidence
        if dopamine and hasattr(dopamine, 'expected_reward'):
            if dopamine.expected_reward > 0.6:
                confidence += 0.1
        
        # High cortisol reduces confidence
        if cortisol and cortisol.current_level > 70:
            confidence *= 0.8
        
        return np.clip(confidence, CONFIDENCE_MIN, CONFIDENCE_MAX)
    
    def _calculate_creativity(self, dopamine, serotonin, cortisol) -> float:
        """
        Calculate creativity level
        
        Moderate dopamine elevation → creativity
        Too high dopamine → scattered
        High cortisol → rigid thinking
        Balanced serotonin → creative confidence
        """
        creativity = 0.5
        
        if dopamine:
            dopamine_level = dopamine.current_level
            # Sweet spot for creativity: slightly elevated dopamine
            if 50 < dopamine_level < 70:
                creativity = 0.7
            elif dopamine_level > 80:
                creativity = 0.4  # Too scattered
            else:
                creativity = 0.5
        
        # High cortisol reduces creativity
        if cortisol and cortisol.current_level > 60:
            creativity *= (1 - min(0.4, (cortisol.current_level - 60) / 100))
        
        # Good serotonin supports creative confidence
        if serotonin and serotonin.current_level > 50:
            creativity *= 1.1
        
        return np.clip(creativity, 0, 1)
    
    def _calculate_empathy(self, oxytocin, serotonin) -> float:
        """
        Calculate empathy level
        
        High oxytocin → high empathy
        Good serotonin → emotional availability
        """
        empathy = 0.5
        
        if oxytocin and hasattr(oxytocin, 'empathy_level'):
            empathy = oxytocin.empathy_level
        elif oxytocin:
            empathy = min(1.0, oxytocin.current_level / 60)
        
        # Serotonin enables emotional availability
        if serotonin and serotonin.current_level > 40:
            empathy *= (1 + (serotonin.current_level - 40) / 100)
        
        return np.clip(empathy, 0, 1)
    
    def _calculate_patience(self, serotonin, cortisol, adrenaline) -> float:
        """
        Calculate patience level
        
        High serotonin → patient
        High cortisol → impatient (stressed)
        High adrenaline → impatient (urgent)
        """
        patience = 0.5
        
        # Serotonin increases patience
        if serotonin:
            patience = min(0.8, serotonin.current_level / 80)
        
        # Cortisol decreases patience when elevated
        if cortisol and cortisol.current_level > 60:
            patience *= (1 - min(0.4, (cortisol.current_level - 60) / 80))
        
        # Adrenaline decreases patience
        if adrenaline and adrenaline.current_level > adrenaline.baseline + 20:
            patience *= 0.7
        
        return np.clip(patience, 0, 1)
    
    def _calculate_thoroughness(self, cortisol, serotonin) -> float:
        """
        Calculate thoroughness level
        
        High cortisol → thorough (anxious checking)
        Good serotonin → systematic thoroughness
        """
        thoroughness = 0.5
        
        # Cortisol increases thoroughness
        if cortisol:
            if 40 < cortisol.current_level < 70:
                # Optimal stress for thoroughness
                thoroughness = 0.7
            elif cortisol.current_level >= 70:
                # Too stressed, might miss things
                thoroughness = 0.5
        
        # Serotonin enables systematic approach
        if serotonin and serotonin.current_level > 50:
            thoroughness *= 1.2
        
        return np.clip(thoroughness, 0, 1)
    
    def _should_explore(self, dopamine, serotonin) -> bool:
        """Determine if should explore new approaches"""
        if not dopamine:
            return False
            
        # Low dopamine below baseline = need novelty
        if dopamine.current_level < dopamine.baseline - 10:
            return True
            
        # High confidence + moderate dopamine = explore
        if serotonin and serotonin.current_level > 60 and \
           dopamine.current_level > dopamine.baseline:
            return True
            
        return False
    
    def _should_exploit(self, dopamine, serotonin) -> bool:
        """Determine if should exploit known strategies"""
        if not dopamine or not serotonin:
            return True  # Default to safe
            
        # High serotonin + stable dopamine = exploit what works
        if serotonin.current_level > 60 and \
           abs(dopamine.get_amplitude()) < 15:
            return True
            
        return False
    
    def _needs_break(self, cortisol, adrenaline) -> bool:
        """Determine if needs a break"""
        needs_break = False
        
        # High chronic stress
        if cortisol and hasattr(cortisol, 'burnout_risk'):
            if cortisol.burnout_risk > 0.6:
                needs_break = True
        elif cortisol and cortisol.current_level > 80:
            needs_break = True
        
        # Adrenaline exhaustion
        if adrenaline and hasattr(adrenaline, 'is_exhausted'):
            if adrenaline.is_exhausted:
                needs_break = True
                
        return needs_break
    
    def _in_flow_state(self, dopamine, cortisol, adrenaline) -> bool:
        """Determine if in flow state"""
        if not all([dopamine, cortisol, adrenaline]):
            return False
            
        # Flow = moderate arousal, positive mood, focused
        dopamine_good = 50 < dopamine.current_level < 70
        cortisol_optimal = 30 < cortisol.current_level < 50
        adrenaline_moderate = adrenaline.baseline < adrenaline.current_level < adrenaline.baseline + 30
        
        return dopamine_good and cortisol_optimal and adrenaline_moderate
    
    def _smooth_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Smooth parameters with historical values
        
        Args:
            params: Current parameters
            
        Returns:
            Smoothed parameters
        """
        if not self.behavior_history:
            return params
            
        smoothed = params.copy()
        
        # Get recent history
        recent = self.behavior_history[-min(3, len(self.behavior_history)):]
        
        # Apply exponential moving average
        alpha = 0.3  # Weight for current value
        
        for key in ['planning_depth', 'risk_tolerance', 'processing_speed', 
                   'confidence', 'creativity', 'empathy', 'patience', 'thoroughness']:
            if key in params and key in recent[-1]['parameters']:
                historical_avg = np.mean([h['parameters'].get(key, params[key]) 
                                         for h in recent])
                smoothed[key] = alpha * params[key] + (1 - alpha) * historical_avg
                
        return smoothed
    
    def _update_tendencies(self):
        """Update behavioral tendencies based on chronic states"""
        hormones = self.state.hormones
        
        # Analytical tendency (high cortisol, high planning)
        if hormones.get('cortisol'):
            cortisol_chronic = hormones['cortisol'].baseline > 40
            if cortisol_chronic and self.current_behavior['planning_depth'] > 5:
                self.tendencies['analytical'] = min(1.0, self.tendencies['analytical'] + 0.01)
            else:
                self.tendencies['analytical'] *= 0.99
        
        # Impulsive tendency (high dopamine, low planning)
        if hormones.get('dopamine'):
            dopamine_seeking = hormones['dopamine'].current_level > hormones['dopamine'].baseline + 20
            if dopamine_seeking and self.current_behavior['planning_depth'] < 3:
                self.tendencies['impulsive'] = min(1.0, self.tendencies['impulsive'] + 0.01)
            else:
                self.tendencies['impulsive'] *= 0.99
        
        # Creative tendency (balanced with slight elevation)
        if self.current_behavior['creativity'] > 0.6:
            self.tendencies['creative'] = min(1.0, self.tendencies['creative'] + 0.01)
        else:
            self.tendencies['creative'] *= 0.99
        
        # Cautious tendency (high cortisol, low risk)
        if self.current_behavior['risk_tolerance'] < 0.3:
            self.tendencies['cautious'] = min(1.0, self.tendencies['cautious'] + 0.01)
        else:
            self.tendencies['cautious'] *= 0.99
        
        # Adventurous tendency (high risk tolerance)
        if self.current_behavior['risk_tolerance'] > 0.7:
            self.tendencies['adventurous'] = min(1.0, self.tendencies['adventurous'] + 0.01)
        else:
            self.tendencies['adventurous'] *= 0.99
    
    def get_behavioral_profile(self) -> Dict:
        """Get comprehensive behavioral profile"""
        return {
            'current_behavior': self.current_behavior.copy(),
            'tendencies': self.tendencies.copy(),
            'dominant_tendency': max(self.tendencies.items(), key=lambda x: x[1])[0],
            'behavioral_state': self._get_behavioral_state(),
            'recommendations': self._get_behavioral_recommendations()
        }
    
    def _get_behavioral_state(self) -> str:
        """Get current behavioral state description"""
        if self.current_behavior.get('needs_break'):
            return "exhausted"
        elif self.current_behavior.get('in_flow_state'):
            return "flow"
        elif self.current_behavior['planning_depth'] > 7:
            return "hyper-analytical"
        elif self.current_behavior['risk_tolerance'] > 0.8:
            return "risk-seeking"
        elif self.current_behavior['confidence'] > 0.8:
            return "confident"
        elif self.current_behavior['creativity'] > 0.7:
            return "creative"
        else:
            return "balanced"
    
    def _get_behavioral_recommendations(self) -> List[str]:
        """Get recommendations based on current state"""
        recommendations = []
        
        if self.current_behavior.get('needs_break'):
            recommendations.append("Take a break to recover")
            
        if self.tendencies['cautious'] > 0.7:
            recommendations.append("Consider taking calculated risks")
            
        if self.tendencies['impulsive'] > 0.7:
            recommendations.append("Slow down and plan more thoroughly")
            
        if self.current_behavior['creativity'] < 0.3:
            recommendations.append("Try exploring new approaches")
            
        if self.current_behavior['empathy'] < 0.3:
            recommendations.append("Consider user's emotional state")
            
        return recommendations
