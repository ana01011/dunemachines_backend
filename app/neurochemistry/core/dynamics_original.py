"""
Complete 7D Neurochemical Dynamics
Full differential equations with all biological terms
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
import math

from .constants import *
from .state import NeurochemicalState
from .state import NeurochemicalState

class NeurochemicalDynamics:
    """
    Implements complete differential equation system:
    dX/dt = F(X, R, B, PE, L, t) + noise
    """
    
    def __init__(self, state: NeurochemicalState):
        self.state = state
        self.t = 0.0  # Internal time for circadian
    
    def hill_equation(self, X: float, n: float, K: float, R: float) -> float:
        """
        Hill equation for non-linear activation
        X_eff = R * [X^n / (K^n + X^n)]
        """
        if X < 0:
            return 0.0
        return R * (X**n / (K**n + X**n))
    
    def calculate_effective_hormones(self) -> np.ndarray:
        """Calculate all effective hormone levels"""
        X_eff = np.zeros(7)
        for i in range(7):
            X_eff[i] = self.hill_equation(
                self.state.hormones[i],
                HILL_N[i],
                HILL_K[i],
                self.state.receptors[i]
            )
        return X_eff
    
    def sigmoid(self, x: float, threshold: float = 0, width: float = 1) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-(x - threshold) / width))
    
    def heaviside(self, x: float) -> float:
        """Heaviside step function"""
        return 1.0 if x > 0 else 0.0
    
    def calculate_dopamine_dynamics(self, X: np.ndarray, X_eff: np.ndarray, 
                                   PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete dopamine dynamics equation
        """
        D = X[D_IDX]
        B_D = self.state.baselines[D_IDX]
        R_D = self.state.receptors[D_IDX]
        PE_D = PE[D_IDX]
        
        # Effective hormones
        C_eff = X_eff[C_IDX]
        N_eff = X_eff[N_IDX]
        
        # Base homeostasis
        dD_dt = -LAMBDA_DECAY[D_IDX] * (D - B_D) * R_D
        
        # Prediction error response
        dD_dt += 0.5 * PE_D * self.sigmoid(PE_D / 10) * R_D
        
        # Reward integration
        reward = inputs.get('reward', 0)
        dD_dt += 0.3 * reward * np.exp(-self.t / 3600) * R_D
        
        # Cortisol suppression
        K_DC = 50.0
        dD_dt -= 0.3 * C_eff * D / (K_DC + D)
        
        # Norepinephrine synergy
        dD_dt += 0.2 * N_eff * np.sqrt(D) * (1 - D/100)
        
        # Saturation term
        K_sat = 80.0
        dD_dt -= 0.1 * D**3 / (K_sat**3 + D**3)
        
        # Seeking behavior (based on PE)
        seeking = self.calculate_seeking(PE_D, C_eff, N_eff)
        dD_dt += seeking * (1 - D/100)
        
        # Deviation cost (minimization principle)
        K_cost = K_COST[D_IDX]
        dD_dt -= 0.05 * (D - B_D)**3 / K_cost**3
        
        # Baseline tracking term
        if B_D > D:  # Baseline is above current
            dD_dt += 0.1 * (B_D - D) * self.heaviside(B_D - D)
        
        # Production and degradation
        production = self.calculate_production_D()
        degradation = self.calculate_degradation(D, K_DEG[D_IDX])
        dD_dt += production - degradation
        
        return dD_dt
    
    def calculate_serotonin_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                     PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete serotonin dynamics equation
        """
        S = X[S_IDX]
        B_S = self.state.baselines[S_IDX]
        R_S = self.state.receptors[S_IDX]
        
        C_eff = X_eff[C_IDX]
        A_eff = X_eff[A_IDX]
        O_eff = X_eff[O_IDX]
        E_eff = X_eff[E_IDX]
        
        # Homeostasis
        dS_dt = -LAMBDA_DECAY[S_IDX] * (S - B_S) * R_S
        
        # Social success integration
        social = inputs.get('social', 0)
        dS_dt += 0.4 * social * np.exp(-self.t / 7200) * R_S
        
        # Cortisol suppression
        K_SC = 60.0
        dS_dt -= 0.4 * C_eff * S / (K_SC + S)
        
        # Adrenaline suppression (when high)
        dS_dt -= 0.2 * A_eff * self.heaviside(A_eff - 50) * S / 100
        
        # Oxytocin synergy
        dS_dt += 0.3 * O_eff * np.sqrt(S) * (1 - S/100)
        
        # Endorphin boost
        dS_dt += 0.2 * E_eff * (1 - S/100)
        
        # Saturation
        K_sat = 85.0
        dS_dt -= 0.08 * S**3 / (K_sat**3 + S**3)
        
        # Deviation cost
        K_cost = K_COST[S_IDX]
        dS_dt -= 0.06 * (S - B_S)**3 / K_cost**3
        
        # Stability signal (rewards stable states)
        stability = self.calculate_stability_signal(X)
        dS_dt += 0.1 * stability * (1 - S/100)
        
        # Production and degradation
        production = self.calculate_production_S()
        degradation = self.calculate_degradation(S, K_DEG[S_IDX])
        dS_dt += production - degradation
        
        return dS_dt
    
    def calculate_cortisol_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                   PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete cortisol dynamics equation
        """
        C = X[C_IDX]
        B_C = self.state.baselines[C_IDX]
        R_C = self.state.receptors[C_IDX]
        
        O_eff = X_eff[O_IDX]
        S_eff = X_eff[S_IDX]
        A = X[A_IDX]
        
        # Faster homeostasis for cortisol
        dC_dt = -LAMBDA_DECAY[C_IDX] * (C - B_C) * (1 + 0.5 * R_C)
        
        # Total prediction error magnitude
        PE_total = np.linalg.norm(PE)
        uncertainty = inputs.get('uncertainty', 0)
        dC_dt += 0.4 * PE_total * (1 + uncertainty)
        
        # Threat response
        threat = inputs.get('threat', 0)
        K_threat = 30.0
        dC_dt += 0.5 * threat**3 / (K_threat**3 + threat**3)
        
        # Oxytocin suppression
        K_CO = 40.0
        dC_dt -= 0.3 * O_eff * C / (K_CO + C)
        
        # Serotonin suppression (when high)
        dC_dt -= 0.2 * S_eff * self.heaviside(S_eff - 60) * np.sqrt(C)
        
        # Adrenaline trigger (rate of change)
        if len(self.state.short_history) > 0:
            A_prev = self.state.short_history[-1][A_IDX]
            dA_dt = A - A_prev
            dC_dt += 0.3 * dA_dt * self.heaviside(dA_dt)
        
        # Allostatic load contribution
        L = self.state.allostatic_load
        dC_dt += 0.1 * L / L_MAX * (1 - C/100)
        
        # Deviation cost
        K_cost = K_COST[C_IDX]
        dC_dt -= 0.04 * (C - B_C)**3 / K_cost**3
        
        # Anticipated stress
        C_exp = self.state.expected[C_IDX]
        dC_dt += .15 * self.heaviside(C_exp - C) * (C_exp - C) / 20
        
        # Production (HPA axis)
        production = K_PROD[C_IDX] * self.state.e_atp / E_ATP_MAX
        degradation = self.calculate_degradation(C, K_DEG[C_IDX])
        dC_dt += production - degradation
        
        return dC_dt
    
    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete adrenaline dynamics equation
        """
        A = X[A_IDX]
        B_A = self.state.baselines[A_IDX]
        R_A = self.state.receptors[A_IDX]
        
        N_eff = X_eff[N_IDX]
        C = X[C_IDX]
        
        # Very fast decay
        dA_dt = -LAMBDA_DECAY[A_IDX] * (A - B_A) * (1 + R_A)
        
        # Urgency response (sharp)
        urgency = inputs.get('urgency', 0)
        K_urg = 25.0
        dA_dt += 0.6 * urgency**4 / (K_urg**4 + urgency**4)
        
        # Norepinephrine synergy
        dA_dt += 0.3 * N_eff * np.sqrt(A) * (1 - A/100)
        
        # Cortisol trigger (rate sensitive)
        if len(self.state.short_history) > 0:
            C_prev = self.state.short_history[-1][C_IDX]
            dC_dt = C - C_prev
            dA_dt += 0.4 * dC_dt * self.heaviside(dC_dt) * (1 - A/100)
        
        # Sharp saturation
        K_sat = 60.0
        dA_dt -= 0.15 * A**4 / (K_sat**4 + A**4)
        
        # High deviation cost (expensive to maintain)
        K_cost = K_COST[A_IDX]
        dA_dt -= 0.08 * (A - B_A)**4 / K_cost**4
        
        # Fight-flight signal
        fight_flight = inputs.get('fight_flight', 0)
        dA_dt += 0.5 * fight_flight * (1 - A/100)
        
        # Production from tyrosine pool
        production = self.calculate_production_A()
        
        # Rapid degradation
        degradation = self.calculate_degradation(A, K_DEG[A_IDX] * 1.5)
        
        # Pool depletion
        depletion = 0.1 * A**2 * (1 - self.state.p_tyr / P_TYR_MAX)
        
        dA_dt += production - degradation - depletion
        
        return dA_dt
    
    def calculate_oxytocin_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                   PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete oxytocin dynamics equation
        """
        O = X[O_IDX]
        B_O = self.state.baselines[O_IDX]
        R_O = self.state.receptors[O_IDX]
        
        C_eff = X_eff[C_IDX]
        S_eff = X_eff[S_IDX]
        
        # Homeostasis
        dO_dt = -LAMBDA_DECAY[O_IDX] * (O - B_O) * R_O
        
        # Social bonding
        social = inputs.get('social', 0)
        K_soc = 30.0
        dO_dt += 0.5 * social**2 / (K_soc**2 + social**2)
        
        # Trust building
        trust = inputs.get('trust', 0)
        dO_dt += 0.3 * trust * (1 - C_eff/100)
        
        # Cortisol suppression
        K_OC = 50.0
        dO_dt -= 0.3 * C_eff * O / (K_OC + O)
        
        # Serotonin synergy
        dO_dt += 0.2 * S_eff * np.sqrt(O) * (1 - O/100)
        
        # Physical contact
        touch = inputs.get('touch', 0)
        dO_dt += 0.4 * touch * R_O
        
        # Deviation cost
        K_cost = K_COST[O_IDX]
        dO_dt -= 0.05 * (O - B_O)**2 / K_cost**2
        
        # Attachment memory (slow integration)
        attachment = inputs.get('attachment', 0)
        dO_dt += 0.1 * attachment * np.exp(-self.t / 7200)
        
        # Production
        production = K_PROD[O_IDX] * self.state.e_atp / E_ATP_MAX
        degradation = self.calculate_degradation(O, K_DEG[O_IDX])
        dO_dt += production - degradation
        
        return dO_dt
    
    def calculate_norepinephrine_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                          PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete norepinephrine dynamics equation
        """
        N = X[N_IDX]
        B_N = self.state.baselines[N_IDX]
        R_N = self.state.receptors[N_IDX]
        
        D_eff = X_eff[D_IDX]
        A_eff = X_eff[A_IDX]
        
        # Homeostasis
        dN_dt = -LAMBDA_DECAY[N_IDX] * (N - B_N) * R_N
        
        # Attention demand
        attention = inputs.get('attention', 0)
        K_att = 35.0
        dN_dt += 0.5 * attention**3 / (K_att**3 + attention**3)
        
        # Dopamine coupling
        dN_dt += 0.2 * D_eff * (1 - N/100)
        
        # Adrenaline boost
        dN_dt += 0.3 * A_eff * np.sqrt(N) * (1 - N/80)
        
        # Saturation
        K_sat = 75.0
        dN_dt -= 0.1 * N**3 / (K_sat**3 + N**3)
        
        # Deviation cost
        K_cost = K_COST[N_IDX]
        dN_dt -= 0.06 * (N - B_N)**3 / K_cost**3
        
        # Vigilance from prediction errors
        vigilance = self.calculate_vigilance(PE)
        dN_dt += 0.2 * vigilance * (1 - N/100)
        
        # Production from tyrosine (shared with D and A)
        production = self.calculate_production_N()
        degradation = self.calculate_degradation(N, K_DEG[N_IDX])
        dN_dt += production - degradation
        
        return dN_dt
    
    def calculate_endorphins_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        """
        Complete endorphins dynamics equation
        """
        E = X[E_IDX]
        B_E = self.state.baselines[E_IDX]
        R_E = self.state.receptors[E_IDX]
        
        D_eff = X_eff[D_IDX]
        
        # Homeostasis
        dE_dt = -LAMBDA_DECAY[E_IDX] * (E - B_E) * R_E
        
        # Exercise release
        exercise = inputs.get('exercise', 0)
        K_ex = 30.0
        dE_dt += 0.5 * exercise**2 / (K_ex**2 + exercise**2)
        
        # Pain relief response
        pain = inputs.get('pain', 0)
        dE_dt += 0.4 * pain * (1 - E/100)
        
        # Dopamine synergy
        dE_dt += 0.2 * D_eff * np.sqrt(E) * (1 - E/100)
        
        # Pleasure response
        pleasure = inputs.get('pleasure', 0)
        K_pl = 25.0
        dE_dt += 0.4 * pleasure**3 / (K_pl**3 + pleasure**3)
        
        # Strong saturation
        K_sat = 70.0
        dE_dt -= 0.12 * E**4 / (K_sat**4 + E**4)
        
        # Deviation cost
        K_cost = K_COST[E_IDX]
        dE_dt -= 0.05 * (E - B_E)**2 / K_cost**2
        
        # Reward prediction
        PE_D = PE[D_IDX]
        dE_dt += 0.15 * self.heaviside(PE_D) * PE_D * np.exp(-E/50)
        
        # Production
        production = K_PROD[E_IDX] * self.state.e_atp / E_ATP_MAX
        # Fast enzymatic breakdown
        degradation = self.calculate_degradation(E, K_DEG[E_IDX] * 1.2)
        dE_dt += production - degradation
        
        return dE_dt
    
    def calculate_receptor_dynamics(self) -> np.ndarray:
        """
        Calculate receptor sensitivity changes
        dR/dt for all receptors
        """
        dR_dt = np.zeros(7)
        X_eff = self.calculate_effective_hormones()
        
        for i in range(7):
            R = self.state.receptors[i]
            X = X_eff[i]
            R_bar = R_BASELINE[i]
            
            # Recovery when hormone is low
            recovery = ALPHA_R[i] * (1 - R) * np.exp(-X / 20)
            
            # Desensitization when hormone is high
            K_R = 50.0
            desensitization = BETA_R[i] * R * (X / K_R)**2
            
            # Baseline return
            baseline_return = GAMMA_R[i] * (R_bar - R) * np.exp(-X / 30)
            
            # Chronic adaptation (from history)
            if len(self.state.history) > 10:
                recent_avg = np.mean([h[i] for h in list(self.state.history)[-20:]])
                chronic = 0.01 * recent_avg**2 / 1000
            else:
                chronic = 0
            
            # Withdrawal sensitization (when X << baseline)
            B = self.state.baselines[i]
            withdrawal = 0.05 * self.heaviside(B - X - 20) * (B - X) / 50
            
            dR_dt[i] = recovery - desensitization + baseline_return - chronic + withdrawal
        
        return dR_dt
    
    def calculate_baseline_dynamics(self, reward_signal: float = 0) -> np.ndarray:
        """
        Adaptive baseline dynamics (minimization principle)
        dB/dt
        """
        dB_dt = np.zeros(7)
        X = self.state.hormones
        B = self.state.baselines
        PE = self.state.expected - X
        L = self.state.allostatic_load
        
        # Calculate long-term average
        if len(self.state.history) > 50:
            X_long_avg = np.mean(list(self.state.history)[-100:], axis=0)
        else:
            X_long_avg = X
        
        for i in range(7):
            # Fast adaptation to current levels
            fast = ETA_FAST[i] * (X[i] - B[i]) * (1 + reward_signal)
            
            # Slow adaptation to long-term average
            slow = ETA_SLOW[i] * (X_long_avg[i] - B[i])
            
            # Return to rest baseline
            rest_return = 0.001 * (BASELINE_REST[i] - B[i]) * np.exp(-L / L_MAX)
            
            # Predictive adjustment (if expecting changes)
            if abs(PE[i]) > 10:
                predictive = 0.005 * PE[i] * self.heaviside(PE[i])
            else:
                predictive = 0
            
            # Circadian modulation
            circadian = self.calculate_circadian_modulation(i)
            
            dB_dt[i] = fast + slow - rest_return + predictive + circadian
        
        return dB_dt
    
    def calculate_expectation_dynamics(self) -> np.ndarray:
        """
        Update expected states
        dX_exp/dt
        """
        dX_exp_dt = np.zeros(7)
        X = self.state.hormones
        X_exp = self.state.expected
        B = self.state.baselines
        
        for i in range(7):
            # Track reality
            reality_tracking = 0.1 * (X[i] - X_exp[i])
            
            # Pattern recognition (simple AR model)
            if len(self.state.history) > 10:
                recent = [h[i] for h in list(self.state.history)[-10:]]
                trend = np.mean(np.diff(recent))
                pattern = 0.05 * trend
            else:
                pattern = 0
            
            # Context influence (move toward baseline)
            context = 0.02 * (B[i] - X_exp[i])
            
            # Confidence decay
            PE = abs(X_exp[i] - X[i])
            confidence_decay = 0.03 * (X_exp[i] - B[i]) * np.exp(-PE / 10)
            
            dX_exp_dt[i] = reality_tracking + pattern + context - confidence_decay
        
        return dX_exp_dt
    
    def calculate_allostatic_load_dynamics(self) -> float:
        """
        Chronic stress accumulation
        dL/dt
        """
        C = self.state.cortisol
        L = self.state.allostatic_load
        
        # Accumulation when cortisol is high
        if C > C_THRESHOLD:
            accumulation = KAPPA_L * (C - C_THRESHOLD) * np.exp(-self.t / 3600)
        else:
            accumulation = 0
        
        # Recovery factors
        S = self.state.serotonin
        O = self.state.oxytocin
        sleep = 0.1  # Would come from inputs in real implementation
        
        recovery_factors = sleep * (1 - C/100) * (S/100) * (O/100)
        recovery = LAMBDA_L * L * recovery_factors
        
        # Prediction failure stress
        PE = self.state.get_prediction_error()
        PE_stress = 0.0005 * np.sum(PE**2)
        
        dL_dt = accumulation - recovery + PE_stress
        
        return dL_dt
    
    def calculate_resource_dynamics(self, inputs: Dict) -> Tuple[float, float, float]:
        """
        Calculate changes in metabolic resources
        Returns: (dP_tyr/dt, dP_trp/dt, dE_ATP/dt)
        """
        # Tyrosine dynamics
        nutrition_tyr = inputs.get('nutrition', 0.5) * 2.0
        usage_tyr = (
            self.calculate_production_D() * 2 +
            self.calculate_production_N() * 1.5 +
            self.calculate_production_A() * 1
        )
        recycle_tyr = 0.3 * (
            self.calculate_degradation(self.state.dopamine, K_DEG[D_IDX]) +
            self.calculate_degradation(self.state.norepinephrine, K_DEG[N_IDX])
        )
        dP_tyr_dt = nutrition_tyr - usage_tyr + recycle_tyr
        
        # Tryptophan dynamics
        nutrition_trp = inputs.get('nutrition', 0.5) * 1.5
        usage_trp = self.calculate_production_S() * 2
        recycle_trp = 0.2 * self.calculate_degradation(self.state.serotonin, K_DEG[S_IDX])
        dP_trp_dt = nutrition_trp - usage_trp + recycle_trp
        
        # ATP dynamics
        glucose = inputs.get('glucose', 0.7)
        oxygen = inputs.get('oxygen', 0.9)
        K_O2 = 0.5
        
        # Aerobic metabolism
        atp_production = 5.0 * glucose * oxygen / (K_O2 + oxygen)
        
        # ATP usage (all synthesis costs)
        atp_usage = 0.1 * np.sum(np.abs(self.state.hormones - self.state.baselines))
        
        # Basal metabolism
        temperature = inputs.get('temperature', 1.0)
        basal = 1.0 * (1 + 0.5 * temperature)
        
        # Recovery during rest
        sleep = inputs.get('sleep', 0)
        recovery = 3.0 * sleep
        
        dE_ATP_dt = atp_production - atp_usage - basal + recovery
        
        return dP_tyr_dt, dP_trp_dt, dE_ATP_dt
    
    def calculate_interaction_effects(self) -> np.ndarray:
        """
        Calculate hormone-hormone interactions
        J(X,R,t) * X_eff term
        """
        X_eff = self.calculate_effective_hormones()
        interactions = np.zeros(7)
        
        for i in range(7):
            for j in range(7):
                if i != j:
                    # Phase-dependent interaction
                    J_ij = INTERACTION_MATRIX[i, j]
                    
                    # Activation of source hormone
                    activation_j = self.sigmoid((X_eff[j] - 30) / 10)
                    
                    # Receptivity of target hormone
                    receptivity_i = self.sigmoid((70 - X_eff[i]) / 10)
                    
                    # Metabolic modulation
                    metabolic = self.state.e_atp / E_ATP_MAX
                    
                    # Full interaction
                    interaction = J_ij * activation_j * receptivity_i * metabolic
                    interactions[i] += interaction * X_eff[j] * (1 - self.state.hormones[i]/100)
        
        return interactions
    
    def calculate_noise(self) -> np.ndarray:
        """
        State-dependent colored noise
        """
        noise = np.zeros(7)
        X = self.state.hormones
        
        # Update noise state (Ornstein-Uhlenbeck process)
        for i in range(7):
            # White noise
            white = np.random.normal(0, 1)
            
            # Update colored noise state
            tau = TAU_NOISE[i]
            self.state.noise_state[i] = (
                self.state.noise_state[i] * np.exp(-DT_INITIAL / tau) +
                np.sqrt(2 / tau) * white * np.sqrt(DT_INITIAL)
            )
            
            # Calculate noise amplitude
            base_noise = SIGMA_0[i]
            
            # Velocity-dependent (would need history)
            velocity_factor = 1.0
            
            # State-dependent
            state_factor = 1 + SIGMA_1[i] * (X[i] / 100)**2
            
            # Total noise
            noise[i] = base_noise * state_factor * velocity_factor * self.state.noise_state[i]
        
        return noise
    
    def calculate_circadian_modulation(self, hormone_idx: int) -> float:
        """
        Calculate circadian rhythm contribution
        """
        amplitude = CIRCADIAN_AMPLITUDE[hormone_idx]
        phase = CIRCADIAN_PHASE[hormone_idx]
        period = CIRCADIAN_PERIOD
        
        # Primary circadian
        primary = amplitude * np.sin(2 * np.pi * self.t / period + phase)
        
        # Secondary harmonic (ultradian)
        secondary = amplitude * np.sin(4 * np.pi * self.t / period + phase) / 3
        
        return (primary + secondary) * 0.01  # Scale down for baseline changes
    
    # Helper functions for production/degradation
    def calculate_production_D(self) -> float:
        """Dopamine production from tyrosine"""
        K_tyr = 20.0
        K_ATP = 30.0
        return K_PROD[D_IDX] * (
            self.state.p_tyr / (K_tyr + self.state.p_tyr) *
            self.state.e_atp / (K_ATP + self.state.e_atp)
        )
    
    def calculate_production_S(self) -> float:
        """Serotonin production from tryptophan"""
        K_trp = 15.0
        K_ATP = 30.0
        return K_PROD[S_IDX] * (
            self.state.p_trp / (K_trp + self.state.p_trp) *
            self.state.e_atp / (K_ATP + self.state.e_atp)
        )
    
    def calculate_production_A(self) -> float:
        """Adrenaline production from tyrosine"""
        K_tyr = 25.0
        K_ATP = 30.0
        # Competes with dopamine and norepinephrine for tyrosine
        competition_factor = 0.5
        return K_PROD[A_IDX] * competition_factor * (
            self.state.p_tyr / (K_tyr + self.state.p_tyr) *
            self.state.e_atp / (K_ATP + self.state.e_atp)
        )
    
    def calculate_production_N(self) -> float:
        """Norepinephrine production from dopamine/tyrosine"""
        K_tyr = 22.0
        K_ATP = 30.0
        # Can be synthesized from dopamine
        D_contribution = 0.3 * self.state.dopamine / 100
        return K_PROD[N_IDX] * (
            (self.state.p_tyr / (K_tyr + self.state.p_tyr) + D_contribution) *
            self.state.e_atp / (K_ATP + self.state.e_atp)
        )
    
    def calculate_degradation(self, hormone_level: float, k_deg: float) -> float:
        """Generic degradation calculation"""
        K_deg = 50.0
        return k_deg * hormone_level / (K_deg + hormone_level)
    
    def calculate_seeking(self, PE_D: float, C_eff: float, N_eff: float) -> float:
        """Seeking behavior intensity"""
        alpha_seek = 0.3
        return alpha_seek * PE_D * (1 + C_eff/50) * (1 + N_eff/50) * self.heaviside(PE_D)
    
    def calculate_stability_signal(self, X: np.ndarray) -> float:
        """Stability reward signal"""
        if len(self.state.short_history) > 5:
            recent = list(self.state.short_history)[-5:]
            volatility = np.mean([np.std([h[i] for h in recent]) for i in range(7)])
            stability = np.exp(-volatility / 10)
        else:
            stability = 0.5
        return stability
    
    def calculate_vigilance(self, PE: np.ndarray) -> float:
        """Vigilance required based on prediction errors"""
        K_vigilance = 30.0
        PE_magnitude = np.linalg.norm(PE)
        return PE_magnitude / (K_vigilance + PE_magnitude)
    
    def step(self, dt: float, inputs: Dict) -> None:
        """
        Main integration step
        Updates all state variables by dt
        """
        # Save current state to history
        self.state.save_to_history()
        
        # Get current state vectors
        X = self.state.hormones
        X_eff = self.calculate_effective_hormones()
        PE = self.state.get_prediction_error()
        
        # Calculate all derivatives
        dX_dt = np.zeros(7)
        
        # Hormone dynamics
        dX_dt[D_IDX] = self.calculate_dopamine_dynamics(X, X_eff, PE, inputs)
        dX_dt[S_IDX] = self.calculate_serotonin_dynamics(X, X_eff, PE, inputs)
        dX_dt[C_IDX] = self.calculate_cortisol_dynamics(X, X_eff, PE, inputs)
        dX_dt[A_IDX] = self.calculate_adrenaline_dynamics(X, X_eff, PE, inputs)
        dX_dt[O_IDX] = self.calculate_oxytocin_dynamics(X, X_eff, PE, inputs)
        dX_dt[N_IDX] = self.calculate_norepinephrine_dynamics(X, X_eff, PE, inputs)
        dX_dt[E_IDX] = self.calculate_endorphins_dynamics(X, X_eff, PE, inputs)
        
        # Add interaction effects
        dX_dt += self.calculate_interaction_effects()
        
        # Add noise
        dX_dt += self.calculate_noise()
        
        # Receptor dynamics
        dR_dt = self.calculate_receptor_dynamics()
        
        # Baseline dynamics
        reward_signal = inputs.get('reward', 0) - inputs.get('punishment', 0)
        dB_dt = self.calculate_baseline_dynamics(reward_signal)
        
        # Expectation dynamics
        dX_exp_dt = self.calculate_expectation_dynamics()
        
        # Allostatic load
        dL_dt = self.calculate_allostatic_load_dynamics()
        
        # Resource dynamics
        dP_tyr_dt, dP_trp_dt, dE_ATP_dt = self.calculate_resource_dynamics(inputs)
        
        # Update all states (Euler integration for now)
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
