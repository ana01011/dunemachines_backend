"""
Enhanced dynamics with stronger, more realistic responses
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
import math

from .constants import *
from .state import NeurochemicalState

class NeurochemicalDynamics:
    """
    Enhanced dynamics with tuned parameters for realistic responses
    """
    
    def __init__(self, state: NeurochemicalState):
        self.state = state
        self.t = 0.0
        
        # Response amplification factors (to make changes more visible)
        self.response_gain = 5.0  # Overall gain
        self.emotion_gain = 8.0   # Emotional response gain
        self.stress_gain = 6.0    # Stress response gain
        
    def calculate_dopamine_dynamics(self, X: np.ndarray, X_eff: np.ndarray, 
                                   PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced dopamine dynamics"""
        D = X[D_IDX]
        B_D = self.state.baselines[D_IDX]
        R_D = self.state.receptors[D_IDX]
        
        # Homeostasis (weakened to allow bigger changes)
        dD_dt = -LAMBDA_DECAY[D_IDX] * 1.0 * (D - B_D) * R_D  # Stronger homeostasis
        
        # STRONG reward response
        reward = inputs.get('reward', 0)
        if reward > 0:
            dD_dt += self.emotion_gain * reward * (1 - D/100) * R_D
        
        # Punishment decreases dopamine
        punishment = inputs.get('punishment', 0)
        if punishment > 0:
            dD_dt -= self.emotion_gain * 0.7 * punishment * (D/100)
        
        # Novelty spike
        novelty = inputs.get('novelty', 0)
        dD_dt += 3.0 * novelty * (1 - D/100) * R_D
        
        # Cortisol suppression (when stressed)
        C_eff = X_eff[C_IDX]
        if C_eff > 50:
            dD_dt -= 0.5 * (C_eff - 50) / 50 * D/100
        
        # Social boost
        social = inputs.get('social', 0)
        if social > 0.5:
            dD_dt += 2.0 * social * (1 - D/100)
        
        return dD_dt * self.response_gain
    
    def calculate_serotonin_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                     PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced serotonin dynamics"""
        S = X[S_IDX]
        B_S = self.state.baselines[S_IDX]
        R_S = self.state.receptors[S_IDX]
        
        # Homeostasis
        dS_dt = -LAMBDA_DECAY[S_IDX] * 0.8 * (S - B_S) * R_S  # Stronger homeostasis
        
        # Social success strongly increases serotonin
        social = inputs.get('social', 0)
        reward = inputs.get('reward', 0)
        dS_dt += 4.0 * social * reward * (1 - S/100) * R_S
        
        # Stability and routine increase serotonin
        uncertainty = inputs.get('uncertainty', 0)
        dS_dt -= 3.0 * uncertainty * S/100
        
        # Exercise boosts serotonin
        exercise = inputs.get('exercise', 0)
        dS_dt += 3.0 * exercise * (1 - S/100)
        
        # Depression: punishment and stress decrease serotonin
        punishment = inputs.get('punishment', 0)
        threat = inputs.get('threat', 0)
        dS_dt -= 4.0 * (punishment + threat * 0.5) * S/100
        
        # Oxytocin synergy
        O_eff = X_eff[O_IDX]
        if O_eff > 40:
            dS_dt += 0.3 * (O_eff - 40) / 60 * (1 - S/100)
        
        return dS_dt * self.response_gain
    
    def calculate_cortisol_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                   PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced cortisol dynamics"""
        C = X[C_IDX]
        B_C = self.state.baselines[C_IDX]
        R_C = self.state.receptors[C_IDX]
        
        # Faster return to baseline
        dC_dt = -LAMBDA_DECAY[C_IDX] * 1.5 * (C - B_C) * (1 + 0.5 * R_C)
        
        # STRONG threat/stress response
        threat = inputs.get('threat', 0)
        urgency = inputs.get('urgency', 0)
        uncertainty = inputs.get('uncertainty', 0)
        
        stress_total = threat + urgency * 0.7 + uncertainty * 0.5
        dC_dt += self.stress_gain * stress_total * (1 - C/100)
        
        # Punishment increases cortisol
        punishment = inputs.get('punishment', 0)
        dC_dt += 4.0 * punishment * (1 - C/100)
        
        # Relaxation and social bonding decrease cortisol
        social = inputs.get('social', 0)
        O_eff = X_eff[O_IDX]
        if O_eff > 40 and social > 0.5:
            dC_dt -= 3.0 * social * (O_eff/100) * C/100
        
        # Sleep/rest strongly reduces cortisol
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dC_dt -= 5.0 * sleep * C/100
        
        # Exercise temporarily increases then decreases
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dC_dt += 2.0 * exercise * (1 - C/80)  # Increase but cap at 80
        
        return dC_dt * self.response_gain
    
    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced adrenaline dynamics with STRONG response"""
        A = X[A_IDX]
        B_A = self.state.baselines[A_IDX]
        R_A = self.state.receptors[A_IDX]
        
        # Very fast decay
        dA_dt = -LAMBDA_DECAY[A_IDX] * 2.0 * (A - B_A) * (1 + R_A)
        
        # VERY STRONG urgency/threat response
        urgency = inputs.get('urgency', 0)
        threat = inputs.get('threat', 0)
        stress = inputs.get('punishment', 0) * 0.5  # Some stress from negative events
        
        # Calculate total emergency level
        emergency = urgency + threat + stress
        
        if emergency > 0:
            # MASSIVE response to any stress
            dA_dt += 25.0 * emergency * (1 - A/100) * R_A  # Increased from 15 to 25
        
        # Exercise increases adrenaline significantly
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dA_dt += 10.0 * exercise * (1 - A/100)
        
        # Relaxation strongly decreases adrenaline
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dA_dt -= 10.0 * sleep * A/100
        
        return dA_dt * self.response_gain
    
    def calculate_oxytocin_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                   PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced oxytocin dynamics"""
        O = X[O_IDX]
        B_O = self.state.baselines[O_IDX]
        R_O = self.state.receptors[O_IDX]
        
        # Homeostasis
        dO_dt = -LAMBDA_DECAY[O_IDX] * 0.5 * (O - B_O) * R_O
        
        # STRONG social bonding response
        social = inputs.get('social', 0)
        trust = inputs.get('trust', 0)
        touch = inputs.get('touch', 0)
        attachment = inputs.get('attachment', 0)
        
        bonding = social + trust * 0.7 + touch + attachment * 0.5
        if bonding > 0:
            dO_dt += self.emotion_gain * bonding * (1 - O/100) * R_O
        
        # Love and positive emotions boost oxytocin
        reward = inputs.get('reward', 0)
        if reward > 0 and social > 0:
            dO_dt += 3.0 * reward * social * (1 - O/100)
        
        # Stress and threat reduce oxytocin
        threat = inputs.get('threat', 0)
        C_eff = X_eff[C_IDX]
        if C_eff > 50:
            dO_dt -= 0.5 * (C_eff - 50) / 50 * O/100
        
        return dO_dt * self.response_gain
    
    def calculate_norepinephrine_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                          PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced norepinephrine dynamics"""
        N = X[N_IDX]
        B_N = self.state.baselines[N_IDX]
        R_N = self.state.receptors[N_IDX]
        
        # Homeostasis
        dN_dt = -LAMBDA_DECAY[N_IDX] * 0.7 * (N - B_N) * R_N
        
        # Attention and focus demands
        attention = inputs.get('attention', 0)
        urgency = inputs.get('urgency', 0)
        
        focus_demand = attention + urgency * 0.8
        dN_dt += 4.0 * focus_demand * (1 - N/100) * R_N
        
        # Stress increases norepinephrine
        threat = inputs.get('threat', 0)
        dN_dt += 3.0 * threat * (1 - N/100)
        
        # Exercise increases norepinephrine
        exercise = inputs.get('exercise', 0)
        dN_dt += 2.5 * exercise * (1 - N/100)
        
        # Sleep decreases norepinephrine
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dN_dt -= 3.0 * sleep * N/100
        
        return dN_dt * self.response_gain
    
    def calculate_endorphins_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced endorphins dynamics"""
        E = X[E_IDX]
        B_E = self.state.baselines[E_IDX]
        R_E = self.state.receptors[E_IDX]
        
        # Homeostasis
        dE_dt = -LAMBDA_DECAY[E_IDX] * 0.8 * (E - B_E) * R_E
        
        # Exercise STRONGLY increases endorphins
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            # Major endorphin release during exercise
            dE_dt += 15.0 * exercise * (1 - E/100)  # Very strong response!
        
        # Pleasure and reward increase endorphins
        pleasure = inputs.get('pleasure', 0)
        reward = inputs.get('reward', 0)
        dE_dt += 4.0 * (pleasure + reward * 0.5) * (1 - E/100)
        
        # Pain relief response
        pain = inputs.get('pain', 0)
        if pain > 0:
            dE_dt += 3.0 * pain * (1 - E/100)
        
        # Laughter and social joy
        social = inputs.get('social', 0)
        if social > 0.5 and reward > 0:
            dE_dt += 2.0 * social * reward * (1 - E/100)
        
        return dE_dt * self.response_gain
    
    def calculate_receptor_dynamics(self) -> np.ndarray:
        """Receptor dynamics (simplified for stability)"""
        dR_dt = np.zeros(7)
        X_eff = self.calculate_effective_hormones()
        
        for i in range(7):
            R = self.state.receptors[i]
            X = X_eff[i]
            R_bar = R_BASELINE[i]
            
            # Simplified: just track toward baseline with desensitization
            dR_dt[i] = 0.01 * (R_bar - R) - 0.001 * X * R
        
        return dR_dt
    
    def calculate_baseline_dynamics(self, reward_signal: float = 0) -> np.ndarray:
        """Baseline adaptation with drift correction"""
        dB_dt = np.zeros(7)
        X = self.state.hormones
        B = self.state.baselines
        
        for i in range(7):
            # Slow adaptation toward current levels
            adaptation = 0.001 * (X[i] - B[i])
            
            # Drift correction - pull back to rest baseline if too far
            rest_pull = 0.0005 * (BASELINE_REST[i] - B[i])
            
            # Circadian influence (stronger for cortisol)
            if i == C_IDX:  # Cortisol
                # Simple circadian: higher in morning, lower at night
                circadian = 5 * np.sin(2 * np.pi * self.t / 86400 - np.pi/2)
                dB_dt[i] = adaptation + rest_pull + 0.001 * circadian
            else:
                dB_dt[i] = adaptation + rest_pull
        
        return dB_dt
    
    def calculate_expectation_dynamics(self) -> np.ndarray:
        """Expected state dynamics"""
        dX_exp_dt = np.zeros(7)
        X = self.state.hormones
        X_exp = self.state.expected
        
        for i in range(7):
            # Track reality
            dX_exp_dt[i] = 0.05 * (X[i] - X_exp[i])
        
        return dX_exp_dt
    
    def calculate_allostatic_load_dynamics(self) -> float:
        """Allostatic load accumulation with proper scaling"""
        C = self.state.cortisol
        L = self.state.allostatic_load
        
        # Accumulate when cortisol is high (MUCH more sensitive)
        if C > 32:  # Even lower threshold
            accumulation = 0.01 * ((C - 32) / 10) ** 2  # 10x more accumulation
        else:
            accumulation = 0
        
        # Also accumulate from chronic low serotonin
        S = self.state.serotonin
        if S < 40:
            accumulation += 0.0005 * ((40 - S) / 40)
        
        # Recovery (slower, needs real rest)
        sleep = 0  # Would come from inputs
        recovery = 0.0001 * L * (1 + sleep)
        
        return accumulation - recovery
    
    def calculate_resource_dynamics(self, inputs: Dict) -> Tuple[float, float, float]:
        """Resource dynamics with proper depletion and recovery"""
        # Get current usage based on hormone production
        D = self.state.dopamine
        S = self.state.serotonin
        A = self.state.adrenaline
        
        # Tyrosine used by dopamine, norepinephrine, adrenaline
        tyrosine_usage = 0.001 * (D + self.state.norepinephrine + A) / 100
        
        # Tryptophan used by serotonin
        tryptophan_usage = 0.002 * S / 100
        
        # ATP used by all processes (MUCH MORE)
        total_activity = np.sum(np.abs(self.state.hormones - self.state.baselines)) / 100  # More sensitive
        atp_usage = 0.05 * (1 + total_activity * 3)  # 10x more usage
        
        # Recovery from nutrition and rest
        nutrition = inputs.get('nutrition', 0.5)
        sleep = inputs.get('sleep', 0)
        
        # Calculate changes
        dP_tyr_dt = nutrition * 0.01 - tyrosine_usage
        dP_trp_dt = nutrition * 0.01 - tryptophan_usage
        dE_ATP_dt = (nutrition * 0.02 + sleep * 0.03) - atp_usage
        
        return dP_tyr_dt, dP_trp_dt, dE_ATP_dt
    
    def calculate_interaction_effects(self) -> np.ndarray:
        """Hormone interactions (simplified)"""
        X_eff = self.calculate_effective_hormones()
        interactions = np.zeros(7)
        
        # Key interactions only
        # Cortisol suppresses dopamine and serotonin
        if X_eff[C_IDX] > 50:
            interactions[D_IDX] -= 2.0 * (X_eff[C_IDX] - 50) / 50
            interactions[S_IDX] -= 2.5 * (X_eff[C_IDX] - 50) / 50
        
        # Oxytocin boosts serotonin
        if X_eff[O_IDX] > 40:
            interactions[S_IDX] += 1.5 * (X_eff[O_IDX] - 40) / 60
        
        # Dopamine and endorphins synergy
        if X_eff[D_IDX] > 50:
            interactions[E_IDX] += 1.0 * (X_eff[D_IDX] - 50) / 50
        
        return interactions
    
    def calculate_noise(self) -> np.ndarray:
        """Simplified noise"""
        return np.random.normal(0, 0.5, 7)
    
    def calculate_effective_hormones(self) -> np.ndarray:
        """Calculate effective levels with Hill equations"""
        X_eff = np.zeros(7)
        for i in range(7):
            X = self.state.hormones[i]
            n = HILL_N[i]
            K = HILL_K[i]
            R = self.state.receptors[i]
            X_eff[i] = R * (X**n / (K**n + X**n))
        return X_eff
    
    def hill_equation(self, X: float, n: float, K: float, R: float) -> float:
        """Hill equation helper"""
        if X < 0:
            return 0.0
        return R * (X**n / (K**n + X**n))
    
    def sigmoid(self, x: float, threshold: float = 0, width: float = 1) -> float:
        """Sigmoid helper"""
        return 1.0 / (1.0 + np.exp(-(x - threshold) / width))
    
    def heaviside(self, x: float) -> float:
        """Step function helper"""
        return 1.0 if x > 0 else 0.0
    
    def step(self, dt: float, inputs: Dict) -> None:
        """Main integration step"""
        # Save to history
        self.state.save_to_history()
        
        # Get current state
        X = self.state.hormones
        X_eff = self.calculate_effective_hormones()
        PE = self.state.get_prediction_error()
        
        # Calculate all dynamics
        dX_dt = np.zeros(7)
        dX_dt[D_IDX] = self.calculate_dopamine_dynamics(X, X_eff, PE, inputs)
        dX_dt[S_IDX] = self.calculate_serotonin_dynamics(X, X_eff, PE, inputs)
        dX_dt[C_IDX] = self.calculate_cortisol_dynamics(X, X_eff, PE, inputs)
        dX_dt[A_IDX] = self.calculate_adrenaline_dynamics(X, X_eff, PE, inputs)
        dX_dt[O_IDX] = self.calculate_oxytocin_dynamics(X, X_eff, PE, inputs)
        dX_dt[N_IDX] = self.calculate_norepinephrine_dynamics(X, X_eff, PE, inputs)
        dX_dt[E_IDX] = self.calculate_endorphins_dynamics(X, X_eff, PE, inputs)
        
        # Add interactions
        dX_dt += self.calculate_interaction_effects()
        
        # Add noise
        dX_dt += self.calculate_noise()
        
        # Other dynamics
        dR_dt = self.calculate_receptor_dynamics()
        dB_dt = self.calculate_baseline_dynamics()
        dX_exp_dt = self.calculate_expectation_dynamics()
        dL_dt = self.calculate_allostatic_load_dynamics()
        dP_tyr_dt, dP_trp_dt, dE_ATP_dt = self.calculate_resource_dynamics(inputs)
        
        # Update all states
        self.state.hormones += dX_dt * dt
        self.state.receptors += dR_dt * dt
        self.state.baselines += dB_dt * dt
        self.state.expected += dX_exp_dt * dt
        self.state.allostatic_load += dL_dt * dt
        self.state.p_tyr += dP_tyr_dt * dt
        self.state.p_trp += dP_trp_dt * dt
        self.state.e_atp += dE_ATP_dt * dt
        
        # Apply bounds
        self.state.apply_bounds()
        
        # Update time
        self.t += dt
        self.state.last_update = time.time()
