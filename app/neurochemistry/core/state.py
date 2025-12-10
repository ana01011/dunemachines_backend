"""
7D Neurochemical State Implementation
Complete state management with all components
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
from collections import deque

from .constants import *

@dataclass
class NeurochemicalState:
    """
    Complete 7D neurochemical state
    X(t) = [D, S, C, A, O, N, E] - Hormone levels
    R(t) = [R_D, R_S, R_C, R_A, R_O, R_N, R_E] - Receptor sensitivities
    B(t) = [B_D, B_S, B_C, B_A, B_O, B_N, B_E] - Adaptive baselines
    X_exp(t) = Expected states for prediction
    """
    
    # User identification
    user_id: str = "default"
    
    # Core 7D vectors
    hormones: np.ndarray = field(default_factory=lambda: BASELINE_INITIAL.copy())
    receptors: np.ndarray = field(default_factory=lambda: R_BASELINE.copy())
    baselines: np.ndarray = field(default_factory=lambda: BASELINE_INITIAL.copy())
    expected: np.ndarray = field(default_factory=lambda: BASELINE_INITIAL.copy())
    
    # Metabolic resources
    p_tyr: float = P_TYR_MAX  # Tyrosine pool
    p_trp: float = P_TRP_MAX  # Tryptophan pool
    e_atp: float = E_ATP_MAX  # Energy
    
    # Allostatic load
    allostatic_load: float = 0.0
    
    # History tracking
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    short_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Time tracking
    time_created: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Noise state (colored noise)
    noise_state: np.ndarray = field(default_factory=lambda: np.zeros(7))
    
    def __post_init__(self):
        """Initialize arrays properly"""
        self.hormones = np.array(self.hormones, dtype=np.float64)
        self.receptors = np.array(self.receptors, dtype=np.float64)
        self.baselines = np.array(self.baselines, dtype=np.float64)
        self.expected = np.array(self.expected, dtype=np.float64)
        self.noise_state = np.array(self.noise_state, dtype=np.float64)
    
    @property
    def dopamine(self) -> float:
        return self.hormones[D_IDX]
    
    @property
    def serotonin(self) -> float:
        return self.hormones[S_IDX]
    
    @property
    def cortisol(self) -> float:
        return self.hormones[C_IDX]
    
    @property
    def adrenaline(self) -> float:
        return self.hormones[A_IDX]
    
    @property
    def oxytocin(self) -> float:
        return self.hormones[O_IDX]
    
    @property
    def norepinephrine(self) -> float:
        return self.hormones[N_IDX]
    
    @property
    def endorphins(self) -> float:
        return self.hormones[E_IDX]
    
    def get_effective_hormones(self) -> np.ndarray:
        """Calculate effective hormone levels with Hill equations"""
        X_eff = np.zeros(7)
        for i in range(7):
            # Hill equation: R * [X^n / (K^n + X^n)]
            X = self.hormones[i]
            n = HILL_N[i]
            K = HILL_K[i]
            R = self.receptors[i]
            
            X_eff[i] = R * (X**n / (K**n + X**n))
        
        return X_eff
    
    def get_prediction_error(self) -> np.ndarray:
        """Calculate prediction error PE = X_exp - X"""
        return self.expected - self.hormones
    
    def calculate_cost(self) -> float:
        """Calculate total cost for minimization principle"""
        # Deviation cost
        deviation = self.hormones - self.baselines
        deviation_cost = np.sum(ALPHA_DEVIATION * deviation**2 * np.exp(-self.receptors))
        
        # Change cost (approximate with history)
        if len(self.short_history) > 0:
            last_state = self.short_history[-1]
            change = self.hormones - last_state
            change_cost = np.sum(BETA_CHANGE * change**2)
        else:
            change_cost = 0.0
        
        # Metabolic cost
        metabolic_cost = 2.0 * (1 - self.e_atp/E_ATP_MAX)**2
        
        # Uncertainty cost
        PE = self.get_prediction_error()
        uncertainty_cost = 3.0 * np.sum(PE**2)
        
        return deviation_cost + change_cost + metabolic_cost + uncertainty_cost
    
    def apply_bounds(self):
        """Enforce biological constraints"""
        # Hormone bounds
        self.hormones = np.clip(self.hormones, HORMONE_MIN, HORMONE_MAX)
        
        # Receptor bounds
        self.receptors = np.clip(self.receptors, RECEPTOR_MIN, RECEPTOR_MAX)
        
        # Baseline bounds
        self.baselines = np.clip(self.baselines, HORMONE_MIN, HORMONE_MAX)
        
        # Resource bounds
        self.p_tyr = np.clip(self.p_tyr, 0, P_TYR_MAX)
        self.p_trp = np.clip(self.p_trp, 0, P_TRP_MAX)
        self.e_atp = np.clip(self.e_atp, 0, E_ATP_MAX)
        
        # Allostatic load bound
        self.allostatic_load = np.clip(self.allostatic_load, 0, L_MAX)
    
    def save_to_history(self):
        """Save current state to history"""
        state_snapshot = self.hormones.copy()
        self.history.append(state_snapshot)
        self.short_history.append(state_snapshot)
    
    def get_mood_state(self) -> str:
        """Enhanced mood state detection based on hormone patterns"""
        D, S, C, A, O, N, E = self.hormones
        
        # Calculate key ratios
        stress_ratio = C / (S + 1)  # High = stressed
        energy_ratio = (A + N) / 2
        pleasure_ratio = (D + E) / 2
        social_ratio = O
        
        # Priority-based mood detection
        if E > 65 and D > 60:
            return "euphoric"
        elif E > 55 and A > 30:
            return "energized"
        elif O > 70 and S > 45:
            return "loved"
        elif D > 65 and N > 60 and C < 40:
            return "motivated"
        elif N > 60 and C < 40 and A < 30:
            return "focused"
        elif stress_ratio > 1.5 and C > 45:
            return "stressed"
        elif C > 50 and A > 50:
            return "anxious"
        elif D < 40 and S < 40:
            return "sad"
        elif S < 35 and C > 40:
            return "depressed"
        elif O > 60 and C < 35:
            return "content"
        elif pleasure_ratio > 55:
            return "joyful"
        elif energy_ratio < 25 and C < 35:
            return "relaxed"
        elif energy_ratio < 20:
            return "tired"
        elif S > 55 and C < 35:
            return "calm"
        elif D > 55 and E > 40:
            return "happy"
        elif A > 20 and N > 50:
            return "alert"
        elif C > 35 and A < 15:
            return "worried"
        elif S > 40 and D > 50:
            return "balanced"
        else:
            return "neutral"
    
    def get_behavioral_parameters(self) -> Dict[str, float]:
        """Map neurochemistry to behavioral traits"""
        X_eff = self.get_effective_hormones()
        
        # Calculate behavioral parameters
        energy = (X_eff[A_IDX] + X_eff[N_IDX]) / 2
        mood = (X_eff[D_IDX] + X_eff[S_IDX] - X_eff[C_IDX] + X_eff[E_IDX]) / 4
        focus = X_eff[N_IDX] * (1 - X_eff[C_IDX]/100)
        creativity = X_eff[D_IDX] * (1 - X_eff[C_IDX]/100)
        empathy = X_eff[O_IDX] * (1 + X_eff[S_IDX]/100)
        confidence = X_eff[S_IDX] * (1 - X_eff[C_IDX]/100)
        
        return {
            'energy': np.clip(energy/100, 0, 1),
            'mood': np.clip((mood + 50)/100, 0, 1),
            'focus': np.clip(focus/100, 0, 1),
            'creativity': np.clip(creativity/100, 0, 1),
            'empathy': np.clip(empathy/100, 0, 1),
            'confidence': np.clip(confidence/100, 0, 1),
            'stress': np.clip(X_eff[C_IDX]/100, 0, 1),
            'motivation': np.clip(X_eff[D_IDX]/100, 0, 1)
        }
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'hormones': self.hormones.tolist(),
            'receptors': self.receptors.tolist(),
            'baselines': self.baselines.tolist(),
            'expected': self.expected.tolist(),
            'resources': {
                'p_tyr': self.p_tyr,
                'p_trp': self.p_trp,
                'e_atp': self.e_atp
            },
            'allostatic_load': self.allostatic_load,
            'mood': self.get_mood_state(),
            'behavioral': self.get_behavioral_parameters(),
            'time': time.time() - self.time_created
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NeurochemicalState':
        """Create state from dictionary"""
        state = cls(user_id=data.get('user_id', 'default'))
        state.hormones = np.array(data.get('hormones', BASELINE_INITIAL))
        state.receptors = np.array(data.get('receptors', R_BASELINE))
        state.baselines = np.array(data.get('baselines', BASELINE_INITIAL))
        state.expected = np.array(data.get('expected', BASELINE_INITIAL))
        
        resources = data.get('resources', {})
        state.p_tyr = resources.get('p_tyr', P_TYR_MAX)
        state.p_trp = resources.get('p_trp', P_TRP_MAX)
        state.e_atp = resources.get('e_atp', E_ATP_MAX)
        
        state.allostatic_load = data.get('allostatic_load', 0.0)
        
        return state
