"""
Minimization Principle Implementation
The brain minimizes cost function to achieve efficient neurochemical states
"""
 
import numpy as np
from typing import Dict, Tuple, Optional
from .constants import *
from .state import NeurochemicalState
 
class NeurochemicalMinimization:
    def __init__(self, state: NeurochemicalState):
        self.state = state
        self.cost_history = []
        self.gradient_history = []
 
    def calculate_total_cost(self) -> Tuple[float, Dict[str, float]]:
        X = self.state.hormones
        B = self.state.baselines
        R = self.state.receptors
        X_exp = self.state.expected
 
        deviation = X - B
        C_deviation = np.sum(ALPHA_DEVIATION * deviation**2 * np.exp(-R))
 
        if len(self.state.short_history) > 0:
            X_prev = self.state.short_history[-1]
            change = X - X_prev
            C_change = np.sum(BETA_CHANGE * change**2)
        else:
            C_change = 0.0
 
        X_norm = X / HORMONE_MAX
        tyr_depletion = max(0, min(1, 1 - self.state.p_tyr / P_TYR_MAX))
        trp_depletion = max(0, min(1, 1 - self.state.p_trp / P_TRP_MAX))
        atp_depletion = max(0, min(1, 1 - self.state.e_atp / E_ATP_MAX))
 
        C_metabolic = 1.0 * np.sum(X_norm**2) + 2.0 * tyr_depletion**2 + 2.0 * trp_depletion**2 + 3.0 * atp_depletion**2
 
        PE = X_exp - X
        C_uncertainty = 1.0 * np.sum(PE**2 / (K_COST_SCALAR + PE**2))
 
        L = self.state.allostatic_load
        C_allostatic = (L / L_MAX)**2 * 20.0
 
        extremity_low = np.sum(np.exp(-X / 0.1))
        extremity_high = np.sum(np.exp((X - 0.9) / 0.1))
        C_extremity = extremity_low + extremity_high
 
        total_cost = C_deviation + C_change + C_metabolic + C_uncertainty + C_allostatic + C_extremity
 
        components = {
            'deviation': float(C_deviation), 'change': float(C_change),
            'metabolic': float(C_metabolic), 'uncertainty': float(C_uncertainty),
            'allostatic': float(C_allostatic), 'extremity': float(C_extremity),
            'total': float(total_cost)
        }
 
        self.cost_history.append(components)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)
 
        return total_cost, components
 
    def calculate_cost_gradient(self) -> np.ndarray:
        X = self.state.hormones
        B = self.state.baselines
        R = self.state.receptors
        X_exp = self.state.expected
        gradient = np.zeros(7)
 
        for i in range(7):
            grad_deviation = 2 * ALPHA_DEVIATION[i] * (X[i] - B[i]) * np.exp(-R[i])
            grad_change = 0
            if len(self.state.short_history) > 0:
                X_prev = self.state.short_history[-1]
                grad_change = 2 * BETA_CHANGE[i] * (X[i] - X_prev[i])
 
            grad_metabolic = 4 * X[i] / (HORMONE_MAX[i]**2)
            PE_i = X_exp[i] - X[i]
            k_cost_i = K_COST_ARRAY[i]
            grad_uncertainty = -6.0 * PE_i * k_cost_i / ((k_cost_i + PE_i**2)**2)
            grad_extremity_low = -np.exp(-X[i] / 0.1) / 0.1
            grad_extremity_high = np.exp((X[i] - 0.9) / 0.1) / 0.1
 
            gradient[i] = float(grad_deviation + grad_change + grad_metabolic + grad_uncertainty + grad_extremity_low + grad_extremity_high)
 
        self.gradient_history.append(gradient.copy())
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        return gradient
 
    def calculate_optimal_baseline_shift(self) -> np.ndarray:
        if len(self.state.history) < 20:
            return np.zeros(7)
        recent_history = np.array(list(self.state.history)[-50:])
        mean_levels = np.mean(recent_history, axis=0)
        std_levels = np.std(recent_history, axis=0)
        B = self.state.baselines
        shift = np.zeros(7)
 
        for i in range(7):
            frequency_weight = 1 - np.exp(-std_levels[i] / 0.2)
            rest_baseline = BASELINE_REST[i] / 100.0
            rest_pull = 0.3 * (rest_baseline - B[i])
            activity_pull = frequency_weight * (mean_levels[i] - B[i])
 
            if len(self.state.history) > 30:
                recent_10 = recent_history[-10:, i]
                recent_30 = recent_history[-30:, i]
                if np.all(recent_10 > B[i]) and np.all(recent_30 > B[i] - 0.05):
                    shift[i] = 0.1 * (mean_levels[i] - B[i])
                elif np.all(recent_10 < B[i]) and np.all(recent_30 < B[i] + 0.05):
                    shift[i] = 0.1 * (mean_levels[i] - B[i])
                else:
                    shift[i] = 0.02 * activity_pull + 0.01 * rest_pull
            else:
                shift[i] = 0.02 * activity_pull + 0.01 * rest_pull
            shift[i] = np.clip(shift[i], -0.02, 0.02)
        return shift
 
    def calculate_seeking_intensity(self) -> Dict[str, float]:
        PE = self.state.get_prediction_error()
        gradient = self.calculate_cost_gradient()
 
        PE_D = PE[D_IDX]
        tolerance_D = 1 - self.state.receptors[D_IDX]
        dopamine_seeking = min(1.0, abs(PE_D) * 3) * (1 + tolerance_D) * (1 + min(1.0, abs(gradient[D_IDX]) / 5)) * (0.5 + 0.5 * self._sigmoid(PE_D * 20)) * 0.5
 
        PE_S = PE[S_IDX]
        instability = float(np.std(gradient)) if len(self.gradient_history) > 5 else 1.0
        serotonin_seeking = abs(PE_S) * 10 * instability * (1 + abs(gradient[S_IDX]) / 10) * (0.5 + 0.5 * self._sigmoid(PE_S * 20))
 
        PE_O = PE[O_IDX]
        stress = self.state.cortisol
        oxytocin_seeking = abs(PE_O) * 10 * (1 + stress) * (1 + abs(gradient[O_IDX]) / 10) * (0.5 + 0.5 * self._sigmoid(PE_O * 20))
 
        PE_E = PE[E_IDX]
        pain_signal = max(0, gradient[E_IDX])
        endorphin_seeking = abs(PE_E) * 10 * (1 + pain_signal / 10) * (0.5 + 0.5 * self._sigmoid(PE_E * 20))
 
        return {
            'dopamine_seeking': float(np.clip(dopamine_seeking, 0, 1)),
            'serotonin_seeking': float(np.clip(serotonin_seeking * 0.5, 0, 1)),
            'oxytocin_seeking': float(np.clip(oxytocin_seeking * 0.5, 0, 1)),
            'endorphin_seeking': float(np.clip(endorphin_seeking * 0.5, 0, 1)),
            'total_seeking': float(np.clip((dopamine_seeking + serotonin_seeking + oxytocin_seeking + endorphin_seeking) / 4, 0, 1))
        }
 
    def calculate_efficiency_score(self) -> float:
        total_cost, components = self.calculate_total_cost()
        behavioral = self.state.get_behavioral_parameters()
        total_function = sum(behavioral.values())
        if total_cost > 0:
            efficiency = total_function / (total_cost + 1)
        else:
            efficiency = total_function
        return float(np.clip(efficiency, 0, 10))
 
    def suggest_optimal_action(self) -> Dict[str, float]:
        gradient = self.calculate_cost_gradient()
        seeking = self.calculate_seeking_intensity()
        suggestions = {}
 
        if gradient[C_IDX] > 0.5:
            suggestions['seek_calm'] = min(1.0, gradient[C_IDX] / 2)
            suggestions['social_connection'] = 0.7
            suggestions['exercise'] = 0.5
 
        if seeking['dopamine_seeking'] > 0.7:
            suggestions['seek_reward'] = seeking['dopamine_seeking']
            suggestions['novel_experience'] = 0.6
 
        if seeking['serotonin_seeking'] > 0.7:
            suggestions['routine_activity'] = 0.8
            suggestions['social_positive'] = 0.7
            suggestions['achievement'] = 0.6
 
        _, components = self.calculate_total_cost()
        if components['metabolic'] > 2:
            suggestions['rest'] = min(1.0, components['metabolic'] / 3)
            suggestions['nutrition'] = 0.8
            suggestions['reduce_stress'] = 0.7
 
        if self.state.allostatic_load > L_THRESHOLD:
            suggestions['deep_rest'] = 0.9
            suggestions['meditation'] = 0.8
            suggestions['gentle_exercise'] = 0.6
            suggestions['social_support'] = 0.7
 
        return suggestions
 
    def _sigmoid(self, x: float) -> float:
        x = np.clip(x, -500, 500)
        return float(1.0 / (1.0 + np.exp(-x)))
 
    def get_cost_trajectory(self) -> Dict[str, list]:
        if not self.cost_history:
            return {}
        return {
            'deviation': [c['deviation'] for c in self.cost_history],
            'change': [c['change'] for c in self.cost_history],
            'metabolic': [c['metabolic'] for c in self.cost_history],
            'uncertainty': [c['uncertainty'] for c in self.cost_history],
            'allostatic': [c['allostatic'] for c in self.cost_history],
            'total': [c['total'] for c in self.cost_history]
        }
 
    def predict_future_cost(self, action: Dict[str, float], horizon: int = 10) -> float:
        current_cost, _ = self.calculate_total_cost()
        cost_reduction = 0
        if 'rest' in action:
            cost_reduction += action['rest'] * 5
        if 'seek_calm' in action:
            cost_reduction += action['seek_calm'] * 3
        if 'social_connection' in action:
            cost_reduction += action['social_connection'] * 2
        if 'nutrition' in action:
            cost_reduction += action['nutrition'] * 4
        future_cost = current_cost * np.exp(-cost_reduction / 10)
        return float(future_cost)
