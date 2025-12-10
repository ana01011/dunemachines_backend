"""
Minimization Principle Implementation
The brain minimizes cost function to achieve efficient neurochemical states
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .constants import *
from .state import NeurochemicalState

class NeurochemicalMinimization:
    """
    Implements the minimization principle:
    Brain adjusts baselines and behaviors to minimize total cost
    """
    
    def __init__(self, state: NeurochemicalState):
        self.state = state
        self.cost_history = []
        self.gradient_history = []
    
    def calculate_total_cost(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total cost function:
        L = C_deviation + C_change + C_metabolic + C_uncertainty + C_allostatic
        
        Returns: (total_cost, cost_components)
        """
        # Get current state
        X = self.state.hormones
        B = self.state.baselines
        R = self.state.receptors
        X_exp = self.state.expected
        
        # 1. Deviation cost - energy to maintain distance from baseline
        deviation = X - B
        # Weighted by receptor sensitivity (desensitized = higher cost)
        C_deviation = np.sum(ALPHA_DEVIATION * deviation**2 * np.exp(-R))
        
        # 2. Change cost - energy for rapid changes
        if len(self.state.short_history) > 0:
            X_prev = self.state.short_history[-1]
            change = X - X_prev
            C_change = np.sum(BETA_CHANGE * change**2)
        else:
            C_change = 0.0
        
        # 3. Metabolic cost - resource usage
        # Normalized hormone levels
        X_norm = X / HORMONE_MAX
        
        # Resource depletion factors (with bounds to prevent explosion)
        tyr_depletion = max(0, min(1, 1 - self.state.p_tyr / P_TYR_MAX))
        trp_depletion = max(0, min(1, 1 - self.state.p_trp / P_TRP_MAX))
        atp_depletion = max(0, min(1, 1 - self.state.e_atp / E_ATP_MAX))
        
        C_metabolic = (
            1.0 * np.sum(X_norm**2) +  # Reduced weight
            2.0 * tyr_depletion**2 +    # Reduced weight
            2.0 * trp_depletion**2 +    # Reduced weight
            3.0 * atp_depletion**2      # Reduced weight
        )
        
        # 4. Uncertainty cost - prediction errors are expensive (but bounded)
        PE = X_exp - X
        C_uncertainty = 1.0 * np.sum(PE**2 / (K_COST + PE**2))  # Reduced weight
        
        # 5. Allostatic cost - chronic stress burden
        L = self.state.allostatic_load
        C_allostatic = (L / L_MAX)**2 * 20.0
        
        # 6. Extremity cost - being at limits is expensive
        extremity_low = np.sum(np.exp(-X / 10))  # Cost of very low levels
        extremity_high = np.sum(np.exp((X - 90) / 10))  # Cost of very high levels
        C_extremity = extremity_low + extremity_high
        
        # Total cost
        total_cost = (
            C_deviation + 
            C_change + 
            C_metabolic + 
            C_uncertainty + 
            C_allostatic + 
            C_extremity
        )
        
        components = {
            'deviation': C_deviation,
            'change': C_change,
            'metabolic': C_metabolic,
            'uncertainty': C_uncertainty,
            'allostatic': C_allostatic,
            'extremity': C_extremity,
            'total': total_cost
        }
        
        # Store in history
        self.cost_history.append(components)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)
        
        return total_cost, components
    
    def calculate_cost_gradient(self) -> np.ndarray:
        """
        Calculate gradient of cost function with respect to hormones
        ∇L = ∂L/∂X
        """
        X = self.state.hormones
        B = self.state.baselines
        R = self.state.receptors
        X_exp = self.state.expected
        
        gradient = np.zeros(7)
        
        for i in range(7):
            # Deviation gradient
            grad_deviation = 2 * ALPHA_DEVIATION[i] * (X[i] - B[i]) * np.exp(-R[i])
            
            # Change gradient (if we have history)
            if len(self.state.short_history) > 0:
                X_prev = self.state.short_history[-1]
                grad_change = 2 * BETA_CHANGE[i] * (X[i] - X_prev[i])
            else:
                grad_change = 0
            
            # Metabolic gradient
            grad_metabolic = 4 * X[i] / (HORMONE_MAX**2)
            
            # Uncertainty gradient
            PE_i = X_exp[i] - X[i]
            grad_uncertainty = -6.0 * PE_i * K_COST[i] / ((K_COST[i] + PE_i**2)**2)
            
            # Extremity gradient
            grad_extremity_low = -np.exp(-X[i] / 10) / 10
            grad_extremity_high = np.exp((X[i] - 90) / 10) / 10
            
            gradient[i] = (
                grad_deviation + 
                grad_change + 
                grad_metabolic + 
                grad_uncertainty + 
                grad_extremity_low + 
                grad_extremity_high
            )
        
        # Store gradient
        self.gradient_history.append(gradient)
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        
        return gradient
    
    def calculate_optimal_baseline_shift(self) -> np.ndarray:
        """
        Calculate how baselines should shift to minimize future cost
        This implements the key insight: brain adjusts baselines to make
        frequent states cheaper
        """
        if len(self.state.history) < 20:
            return np.zeros(7)
        
        # Get recent hormone history
        recent_history = np.array(list(self.state.history)[-50:])
        
        # Calculate statistics
        mean_levels = np.mean(recent_history, axis=0)
        std_levels = np.std(recent_history, axis=0)
        
        # Current baselines
        B = self.state.baselines
        
        # Optimal baseline shift
        shift = np.zeros(7)
        
        for i in range(7):
            # Baseline should move toward frequently visited states
            # Weighted by how often we're there
            frequency_weight = 1 - np.exp(-std_levels[i] / 20)
            
            # But not too far from rest baseline
            rest_pull = 0.3 * (BASELINE_REST[i] - B[i])
            
            # Shift toward mean of recent activity
            activity_pull = frequency_weight * (mean_levels[i] - B[i])
            
            # If we're consistently above/below, adjust faster
            if len(self.state.history) > 30:
                recent_10 = recent_history[-10:, i]
                recent_30 = recent_history[-30:, i]
                
                if np.all(recent_10 > B[i]) and np.all(recent_30 > B[i] - 5):
                    # Consistently above - raise baseline
                    shift[i] = 0.1 * (mean_levels[i] - B[i])
                elif np.all(recent_10 < B[i]) and np.all(recent_30 < B[i] + 5):
                    # Consistently below - lower baseline
                    shift[i] = 0.1 * (mean_levels[i] - B[i])
                else:
                    # Mixed - slower adjustment
                    shift[i] = 0.02 * activity_pull + 0.01 * rest_pull
            else:
                shift[i] = 0.02 * activity_pull + 0.01 * rest_pull
            
            # Limit shift rate
            shift[i] = np.clip(shift[i], -2.0, 2.0)
        
        return shift
    
    def calculate_seeking_intensity(self) -> Dict[str, float]:
        """
        Calculate how intensely the system should seek different rewards
        Based on prediction errors and cost gradients
        """
        PE = self.state.get_prediction_error()
        gradient = self.calculate_cost_gradient()
        
        # Dopamine seeking (for reward)
        PE_D = PE[D_IDX]
        tolerance_D = 1 - self.state.receptors[D_IDX]
        dopamine_seeking = (
            min(1.0, abs(PE_D) / 30) * (1 + tolerance_D) * 
            (1 + min(1.0, abs(gradient[D_IDX]) / 50)) *
            (0.5 + 0.5 * self._sigmoid(PE_D / 50))  # Scaled properly
        ) * 0.5  # Scale final output
        
        # Serotonin seeking (for stability)
        PE_S = PE[S_IDX]
        instability = np.std(gradient) if len(self.gradient_history) > 5 else 1.0
        serotonin_seeking = (
            abs(PE_S) * instability * 
            (1 + abs(gradient[S_IDX]) / 10) *
            (0.5 + 0.5 * self._sigmoid(PE_S / 50))
        )
        
        # Oxytocin seeking (for connection)
        PE_O = PE[O_IDX]
        stress = self.state.cortisol / 100
        oxytocin_seeking = (
            abs(PE_O) * (1 + stress) *
            (1 + abs(gradient[O_IDX]) / 10) *
            (0.5 + 0.5 * self._sigmoid(PE_O / 50))
        )
        
        # Endorphin seeking (for pleasure/relief)
        PE_E = PE[E_IDX]
        pain_signal = max(0, gradient[E_IDX])  # Positive gradient = need relief
        endorphin_seeking = (
            abs(PE_E) * (1 + pain_signal / 10) *
            (0.5 + 0.5 * self._sigmoid(PE_E / 50))
        )
        
        return {
            'dopamine_seeking': np.clip(dopamine_seeking, 0, 1),
            'serotonin_seeking': np.clip(serotonin_seeking * 0.5, 0, 1),
            'oxytocin_seeking': np.clip(oxytocin_seeking * 0.5, 0, 1),
            'endorphin_seeking': np.clip(endorphin_seeking * 0.5, 0, 1),
            'total_seeking': np.clip(
                (dopamine_seeking + serotonin_seeking + 
                 oxytocin_seeking + endorphin_seeking) / 4, 0, 1
            )
        }
    
    def calculate_efficiency_score(self) -> float:
        """
        Calculate how efficiently the system is operating
        Lower cost per unit function = higher efficiency
        """
        total_cost, components = self.calculate_total_cost()
        
        # Functional output (mood, energy, focus, etc)
        behavioral = self.state.get_behavioral_parameters()
        total_function = sum(behavioral.values())
        
        # Efficiency = function / cost
        if total_cost > 0:
            efficiency = total_function / (total_cost + 1)
        else:
            efficiency = total_function
        
        return np.clip(efficiency, 0, 10)
    
    def suggest_optimal_action(self) -> Dict[str, float]:
        """
        Suggest actions that would minimize future cost
        """
        gradient = self.calculate_cost_gradient()
        seeking = self.calculate_seeking_intensity()
        
        suggestions = {}
        
        # High cortisol - suggest calming
        if gradient[C_IDX] > 5:
            suggestions['seek_calm'] = min(1.0, gradient[C_IDX] / 20)
            suggestions['social_connection'] = 0.7
            suggestions['exercise'] = 0.5
        
        # Low dopamine with high seeking - suggest reward
        if seeking['dopamine_seeking'] > 0.7:
            suggestions['seek_reward'] = seeking['dopamine_seeking']
            suggestions['novel_experience'] = 0.6
        
        # Low serotonin - suggest stability
        if seeking['serotonin_seeking'] > 0.7:
            suggestions['routine_activity'] = 0.8
            suggestions['social_positive'] = 0.7
            suggestions['achievement'] = 0.6
        
        # High metabolic cost - suggest rest
        _, components = self.calculate_total_cost()
        if components['metabolic'] > 20:
            suggestions['rest'] = min(1.0, components['metabolic'] / 30)
            suggestions['nutrition'] = 0.8
            suggestions['reduce_stress'] = 0.7
        
        # High allostatic load - suggest recovery
        if self.state.allostatic_load > L_THRESHOLD:
            suggestions['deep_rest'] = 0.9
            suggestions['meditation'] = 0.8
            suggestions['gentle_exercise'] = 0.6
            suggestions['social_support'] = 0.7
        
        return suggestions
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def get_cost_trajectory(self) -> Dict[str, list]:
        """
        Get cost history for analysis
        """
        if not self.cost_history:
            return {}
        
        trajectory = {
            'deviation': [c['deviation'] for c in self.cost_history],
            'change': [c['change'] for c in self.cost_history],
            'metabolic': [c['metabolic'] for c in self.cost_history],
            'uncertainty': [c['uncertainty'] for c in self.cost_history],
            'allostatic': [c['allostatic'] for c in self.cost_history],
            'total': [c['total'] for c in self.cost_history]
        }
        
        return trajectory
    
    def predict_future_cost(self, action: Dict[str, float], horizon: int = 10) -> float:
        """
        Predict future cost if certain actions are taken
        """
        # This would need the dynamics model to properly predict
        # For now, we'll use a simple heuristic
        
        current_cost, _ = self.calculate_total_cost()
        
        # Estimate impact of actions
        cost_reduction = 0
        
        if 'rest' in action:
            cost_reduction += action['rest'] * 5
        
        if 'seek_calm' in action:
            cost_reduction += action['seek_calm'] * 3
        
        if 'social_connection' in action:
            cost_reduction += action['social_connection'] * 2
        
        if 'nutrition' in action:
            cost_reduction += action['nutrition'] * 4
        
        # Predict future cost
        future_cost = current_cost * np.exp(-cost_reduction / 10)
        
        return future_cost
