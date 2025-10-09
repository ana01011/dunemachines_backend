"""
Core neurochemical state implementation
Implements the complete mathematical framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
from app.neurochemistry.core.constants import *

@dataclass
class NeurochemicalState:
    """
    Complete neurochemical state with dynamics
    X(t) = [D(t), C(t), A(t), S(t), O(t)]ᵀ ∈ ℝ⁵
    B(t) = [Dᵦ(t), Cᵦ(t), Aᵦ(t), Sᵦ(t), Oᵦ(t)]ᵀ ∈ ℝ⁵
    """
    
    # Current hormone levels X(t)
    dopamine: float = 50.0
    cortisol: float = 30.0
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Dynamic baselines B(t)
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    # Spike history for wave analysis
    dopamine_spikes: List[float] = field(default_factory=list)
    cortisol_spikes: List[float] = field(default_factory=list)
    
    # Learning parameters
    expected_reward: float = 0.5
    success_rate: float = 0.5
    recent_outcomes: List[float] = field(default_factory=list)
    
    # Adrenaline pool (depletion model)
    adrenaline_pool: float = 100.0
    
    # Time tracking
    last_update: float = 0.0
    time: float = 0.0
    
    # Event history
    event_history: List[Dict] = field(default_factory=list)
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector X(t)"""
        return np.array([
            self.dopamine,
            self.cortisol,
            self.adrenaline,
            self.serotonin,
            self.oxytocin
        ])
    
    def get_baseline_vector(self) -> np.ndarray:
        """Get baseline vector B(t)"""
        return np.array([
            self.dopamine_baseline,
            self.cortisol_baseline,
            self.adrenaline_baseline,
            self.serotonin_baseline,
            self.oxytocin_baseline
        ])
    
    def get_wave_amplitude(self) -> np.ndarray:
        """Get wave amplitude W(t) = X(t) - B(t)"""
        return self.get_state_vector() - self.get_baseline_vector()
    
    def calculate_interaction_matrix(self) -> np.ndarray:
        """
        Build interaction matrix J for hormone cross-talk
        J_ij represents influence of hormone j on hormone i
        """
        J = np.zeros((5, 5))
        
        # Dopamine row
        J[0, 1] = -INTERACTION_MATRIX['beta_DC']  # Cortisol suppresses
        J[0, 2] = INTERACTION_MATRIX['alpha_DA']   # Adrenaline boosts
        J[0, 3] = -INTERACTION_MATRIX['delta_DS']  # Low serotonin suppresses
        
        # Cortisol row  
        J[1, 0] = INTERACTION_MATRIX['gamma_CD']   # Dopamine baseline triggers
        J[1, 2] = INTERACTION_MATRIX['alpha_CA']   # Adrenaline boosts
        J[1, 3] = -INTERACTION_MATRIX['mu_CS']     # Serotonin suppresses
        
        # Adrenaline row
        J[2, 0] = INTERACTION_MATRIX['zeta_AD']    # Dopamine rate triggers
        J[2, 1] = INTERACTION_MATRIX['xi_AC']      # Cortisol rate triggers
        J[2, 3] = -INTERACTION_MATRIX['nu_AS']     # Serotonin suppresses
        
        # Serotonin row
        J[3, 0] = -INTERACTION_MATRIX['theta_SD']  # Low dopamine suppresses
        J[3, 1] = -INTERACTION_MATRIX['sigma_SC']  # Cortisol suppresses
        J[3, 2] = -INTERACTION_MATRIX['rho_SA']    # Adrenaline suppresses
        J[3, 4] = INTERACTION_MATRIX['epsilon_SO'] # Oxytocin boosts
        
        # Oxytocin row
        J[4, 0] = INTERACTION_MATRIX['kappa_OD']   # Dopamine boosts
        J[4, 3] = INTERACTION_MATRIX['lambda_OS']  # Serotonin boosts
        
        return J
    
    def apply_dynamics(self, dt: float, event: Optional['Event'] = None):
        """
        Apply full dynamics equations
        dX/dt = f(X, B, S, I) + η(t)
        """
        # Get current state
        X = self.get_state_vector()
        B = self.get_baseline_vector()
        W = X - B
        
        # Build return-to-baseline matrix
        Lambda = np.diag([
            LAMBDA_DOPAMINE,
            LAMBDA_CORTISOL,
            LAMBDA_ADRENALINE,
            LAMBDA_SEROTONIN,
            LAMBDA_OXYTOCIN
        ])
        
        # Calculate derivatives
        dX_dt = -Lambda @ W  # Return to baseline term
        
        # Add interaction effects
        J = self.calculate_interaction_matrix()
        dX_dt += J @ X
        
        # Add event-driven changes if event provided
        if event:
            dX_dt += self.calculate_event_response(event)
        
        # Add noise (stochastic term)
        noise = np.array([
            np.random.normal(0, NOISE_AMPLITUDE['dopamine']),
            np.random.normal(0, NOISE_AMPLITUDE['cortisol']),
            np.random.normal(0, NOISE_AMPLITUDE['adrenaline']),
            np.random.normal(0, NOISE_AMPLITUDE['serotonin']),
            np.random.normal(0, NOISE_AMPLITUDE['oxytocin'])
        ]) * np.sqrt(np.abs(W))  # Noise scales with distance from baseline
        
        dX_dt += noise
        
        # Update state (Euler integration)
        X_new = X + dX_dt * dt
        
        # Apply constraints
        X_new = np.clip(X_new, MIN_HORMONE, MAX_HORMONE)
        
        # Update adrenaline pool
        self.update_adrenaline_pool(dt)
        
        # Limit adrenaline by available pool
        X_new[2] = min(X_new[2], self.adrenaline_pool)
        
        # Update state
        self.dopamine = X_new[0]
        self.cortisol = X_new[1]
        self.adrenaline = X_new[2]
        self.serotonin = X_new[3]
        self.oxytocin = X_new[4]
        
        # Update spike history
        self.update_spike_history()
        
        # Adapt baselines
        self.adapt_baselines(dt)
        
        # Update time
        self.time += dt
        self.last_update = self.time
    
    def calculate_event_response(self, event: 'Event') -> np.ndarray:
        """Calculate hormone response to event"""
        response = np.zeros(5)
        
        # Dopamine: reward and novelty
        prediction_error = event.actual_reward - self.expected_reward
        response[0] = (
            REWARD_PREDICTION_GAIN * prediction_error +
            event.novelty * 10 +
            event.success_probability * 5
        )
        
        # Cortisol: stress and uncertainty
        response[1] = (
            PREDICTION_ERROR_SENSITIVITY * abs(prediction_error) +
            UNCERTAINTY_COEFFICIENT * event.uncertainty +
            TIME_PRESSURE_COEFFICIENT * event.time_pressure +
            event.complexity * 8
        )
        
        # Adrenaline: urgency and novelty
        response[2] = (
            NOVELTY_RESPONSE * event.novelty * 10 +
            event.urgency * 15 +
            event.intensity * 5
        )
        
        # Serotonin: success and consistency
        self.recent_outcomes.append(event.success_probability)
        if len(self.recent_outcomes) > PATTERN_WINDOW:
            self.recent_outcomes.pop(0)
        
        consistency = 1.0 / (1.0 + np.var(self.recent_outcomes))
        response[3] = (
            SUCCESS_INTEGRATION * event.success_probability * 10 +
            CONSISTENCY_BONUS * consistency * 5 -
            FAILURE_PENALTY * (1 - event.success_probability) * 5
        )
        
        # Oxytocin: social and emotional
        response[4] = (
            SOCIAL_BONDING_RATE * event.social_interaction * 10 +
            EMPATHY_COEFFICIENT * event.emotional_content * 8 +
            TRUST_BUILDING_RATE * event.trust_factor * 5
        )
        
        return response
    
    def update_spike_history(self):
        """Track spike amplitudes for wave analysis"""
        dopamine_wave = self.dopamine - self.dopamine_baseline
        cortisol_wave = self.cortisol - self.cortisol_baseline
        
        self.dopamine_spikes.append(dopamine_wave)
        self.cortisol_spikes.append(cortisol_wave)
        
        # Keep only recent spikes
        if len(self.dopamine_spikes) > SPIKE_WINDOW:
            self.dopamine_spikes.pop(0)
        if len(self.cortisol_spikes) > SPIKE_WINDOW:
            self.cortisol_spikes.pop(0)
    
    def adapt_baselines(self, dt: float):
        """
        Adapt baselines based on spike patterns
        Implements negative feedback for stability
        """
        # Dopamine baseline adaptation
        if len(self.dopamine_spikes) >= 3:
            # Calculate trend in spike amplitudes
            delta_spikes = np.diff(self.dopamine_spikes[-5:])
            mu_delta = np.mean(delta_spikes) if len(delta_spikes) > 0 else 0
            
            if self.dopamine_baseline < DOPAMINE_CEILING:
                # Below ceiling, can adapt up
                d_baseline = DOPAMINE_ADAPTATION_UP * mu_delta * (1 - self.dopamine_baseline/100)
            else:
                # At/above ceiling, trigger cortisol and force down
                self.cortisol += CORTISOL_TRIGGER_GAIN * (self.dopamine_baseline - DOPAMINE_CEILING)
                if mu_delta > 0:
                    d_baseline = -abs(DOPAMINE_ADAPTATION_DOWN * mu_delta) * (self.dopamine_baseline/100)
                else:
                    d_baseline = DOPAMINE_ADAPTATION_DOWN * mu_delta
            
            self.dopamine_baseline += d_baseline * dt
        
        # Cortisol baseline regulation
        d_cortisol_baseline = (
            CORTISOL_BASELINE_DOPAMINE_RISE * max(0, self.dopamine_baseline - DOPAMINE_CEILING) -
            CORTISOL_RELAXATION_RATE * (self.cortisol_baseline - 30)
        )
        
        # Add chronic stress adaptation
        if len(self.cortisol_spikes) > 0:
            avg_stress = np.mean([abs(s) for s in self.cortisol_spikes])
            d_cortisol_baseline += CHRONIC_STRESS_ADAPTATION * avg_stress
        
        self.cortisol_baseline += d_cortisol_baseline * dt
        
        # Clamp all baselines
        self.dopamine_baseline = np.clip(self.dopamine_baseline, MIN_BASELINE, MAX_BASELINE)
        self.cortisol_baseline = np.clip(self.cortisol_baseline, MIN_BASELINE, MAX_BASELINE)
        self.adrenaline_baseline = np.clip(self.adrenaline_baseline, MIN_BASELINE, MAX_BASELINE)
        self.serotonin_baseline = np.clip(self.serotonin_baseline, MIN_BASELINE, MAX_BASELINE)
        self.oxytocin_baseline = np.clip(self.oxytocin_baseline, MIN_BASELINE, MAX_BASELINE)
    
    def update_adrenaline_pool(self, dt: float):
        """Update available adrenaline (depletion model)"""
        # Regeneration
        self.adrenaline_pool += ADRENALINE_REGEN_RATE * dt
        
        # Usage
        self.adrenaline_pool -= self.adrenaline * ADRENALINE_USAGE_RATE * dt
        
        # Clamp
        self.adrenaline_pool = np.clip(self.adrenaline_pool, 0, MAX_HORMONE)
    
    def apply_opponent_process(self, hormone: str, spike_amplitude: float):
        """Apply opponent process after spike"""
        if hormone == 'dopamine':
            crash_level = self.dopamine_baseline - OPPONENT_PROCESS_STRENGTH * spike_amplitude
            self.dopamine = max(MIN_HORMONE, crash_level)
    
    def get_effective_hormones(self) -> Dict[str, float]:
        """Calculate effective hormone levels with interactions"""
        # Effective dopamine
        d_eff = self.dopamine * (1 - self.cortisol/200) * (1 + self.adrenaline/200) * (2 - self.serotonin/100)
        
        # Effective cortisol
        c_eff = self.cortisol * (1 + max(0, 50 - self.dopamine_baseline)/50) * (1 + self.adrenaline/100) * (1 - self.serotonin/200)
        
        return {
            'dopamine_effective': np.clip(d_eff, MIN_HORMONE, MAX_HORMONE),
            'cortisol_effective': np.clip(c_eff, MIN_HORMONE, MAX_HORMONE),
            'adrenaline': self.adrenaline,
            'serotonin': self.serotonin,
            'oxytocin': self.oxytocin
        }
    
    def calculate_lyapunov_function(self) -> float:
        """
        Calculate Lyapunov function V(X, B) for stability analysis
        V(X, B) = Σᵢ [½(Xᵢ - Bᵢ)² + ¼Bᵢ²]
        """
        X = self.get_state_vector()
        B = self.get_baseline_vector()
        W = X - B
        
        V = 0.5 * np.sum(W**2) + 0.25 * np.sum(B**2)
        return V
    
    def check_stability(self) -> bool:
        """Check if system is stable (eigenvalues of J < 1)"""
        J = self.calculate_interaction_matrix()
        eigenvalues = np.linalg.eigvals(J)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        return max_eigenvalue < 1.0
    
    def get_behavioral_parameters(self) -> Dict[str, float]:
        """
        Map neurochemistry to behavioral parameters
        """
        W = self.get_wave_amplitude()
        
        # Planning depth
        planning_depth_base = 5
        planning_depth = planning_depth_base * (
            (1 + W[1]/50) *  # Cortisol increases planning
            (1 - max(0, W[2])/100) *  # High adrenaline reduces planning
            (2 - self.serotonin/100)  # Low serotonin reduces planning
        )
        
        # Risk tolerance
        risk_tolerance = (
            0.5 +
            0.3 * np.tanh(W[0]/20) -  # Dopamine increases risk
            0.2 * np.tanh(W[1]/20)     # Cortisol reduces risk
        )
        
        # Processing speed
        processing_speed = (
            1 +
            0.5 * self._sigmoid(W[2]/10) -  # Adrenaline increases speed
            0.3 * self._sigmoid(W[1]/30)     # Cortisol reduces speed
        )
        
        # Confidence
        confidence = self._sigmoid(
            self.serotonin/30 +
            W[0]/40 -
            abs(W[1])/50
        )
        
        # Creativity
        creativity = (
            0.5 +
            0.3 * np.tanh(W[0]/25) +  # Dopamine enhances
            0.2 * np.tanh((self.serotonin - 50)/25) -  # Moderate serotonin helps
            0.4 * np.tanh(W[1]/25)  # Cortisol suppresses
        )
        
        # Empathy
        empathy = (
            0.5 +
            0.4 * np.tanh(self.oxytocin/25) +  # Oxytocin primary driver
            0.1 * np.tanh(self.serotonin/50)   # Serotonin helps
        )
        
        return {
            'planning_depth': max(1, planning_depth),
            'risk_tolerance': np.clip(risk_tolerance, 0, 1),
            'processing_speed': max(0.1, processing_speed),
            'confidence': np.clip(confidence, 0, 1),
            'creativity': np.clip(creativity, 0, 1),
            'empathy': np.clip(empathy, 0, 1),
            'patience': np.clip(1 - W[2]/50, 0.1, 1),  # Inverse of adrenaline
            'thoroughness': np.clip(1 + W[1]/50 - W[2]/50, 0.1, 2)  # Cortisol+ Adrenaline-
        }
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

@dataclass
class Event:
    """Event that affects neurochemical state"""
    type: str
    intensity: float = 0.5
    complexity: float = 0.5
    urgency: float = 0.5
    emotional_content: float = 0.5
    novelty: float = 0.5
    success_probability: float = 0.5
    social_interaction: float = 0.5
    actual_reward: float = 0.5
    uncertainty: float = 0.5
    time_pressure: float = 0.5
    trust_factor: float = 0.5
