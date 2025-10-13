"""
Neurochemistry V3: Complete Parameter Configuration
All biological constants and tunable parameters for the 7D system
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# =============================================================================
# HORMONE INDICES
# =============================================================================
HORMONES = ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 'oxytocin', 'norepinephrine', 'endorphins']
HORMONE_ABBREV = ['D', 'S', 'C', 'A', 'O', 'N', 'E']

D_IDX = 0  # Dopamine
S_IDX = 1  # Serotonin  
C_IDX = 2  # Cortisol
A_IDX = 3  # Adrenaline
O_IDX = 4  # Oxytocin
N_IDX = 5  # Norepinephrine
E_IDX = 6  # Endorphins


# =============================================================================
# HILL EQUATION PARAMETERS
# =============================================================================
@dataclass
class HillParameters:
    """Hill equation parameters for non-linear hormone activation"""
    n: np.ndarray = field(default_factory=lambda: np.array([
        2.5,  # Dopamine - moderate cooperativity
        2.0,  # Serotonin - gentle activation
        3.0,  # Cortisol - sharp threshold
        4.0,  # Adrenaline - very sharp activation  
        2.0,  # Oxytocin - low threshold, gentle
        3.0,  # Norepinephrine - sharp focus activation
        2.5,  # Endorphins - low threshold pleasure
    ]))
    
    K: np.ndarray = field(default_factory=lambda: np.array([
        40.0,  # Dopamine half-saturation
        50.0,  # Serotonin half-saturation
        45.0,  # Cortisol half-saturation
        35.0,  # Adrenaline half-saturation
        30.0,  # Oxytocin half-saturation
        40.0,  # Norepinephrine half-saturation
        25.0,  # Endorphins half-saturation
    ]))


# =============================================================================
# HOMEOSTASIS PARAMETERS
# =============================================================================
@dataclass
class HomeostasisParameters:
    """Homeostasis and decay parameters"""
    # Decay rates (return to baseline speeds)
    lambda_decay: np.ndarray = field(default_factory=lambda: np.array([
        0.05,   # Dopamine - moderate decay
        0.02,   # Serotonin - slow decay
        0.08,   # Cortisol - fast decay
        0.20,   # Adrenaline - very fast decay
        0.03,   # Oxytocin - slow decay
        0.10,   # Norepinephrine - fast decay
        0.12,   # Endorphins - fast decay
    ]))
    
    # Resting baselines (default levels)
    baseline_rest: np.ndarray = field(default_factory=lambda: np.array([
        40.0,  # Dopamine
        50.0,  # Serotonin
        25.0,  # Cortisol
        15.0,  # Adrenaline
        35.0,  # Oxytocin
        40.0,  # Norepinephrine
        30.0,  # Endorphins
    ]))
    
    # Initial baselines (can adapt over time)
    baseline_initial: np.ndarray = field(default_factory=lambda: np.array([
        50.0,  # Dopamine
        60.0,  # Serotonin
        30.0,  # Cortisol
        20.0,  # Adrenaline
        40.0,  # Oxytocin
        45.0,  # Norepinephrine
        35.0,  # Endorphins
    ]))


# =============================================================================
# RECEPTOR PARAMETERS
# =============================================================================
@dataclass
class ReceptorParameters:
    """Receptor sensitivity and adaptation parameters"""
    # Recovery rates (when hormone is low)
    alpha_R: np.ndarray = field(default_factory=lambda: np.array([
        0.05,  # D receptor recovery
        0.02,  # S receptor recovery
        0.08,  # C receptor recovery
        0.15,  # A receptor recovery
        0.03,  # O receptor recovery
        0.10,  # N receptor recovery
        0.08,  # E receptor recovery
    ]))
    
    # Desensitization rates (when hormone is high)
    beta_R: np.ndarray = field(default_factory=lambda: np.array([
        0.15,  # D receptor desensitization
        0.05,  # S receptor desensitization
        0.10,  # C receptor desensitization
        0.30,  # A receptor desensitization
        0.08,  # O receptor desensitization
        0.20,  # N receptor desensitization
        0.25,  # E receptor desensitization
    ]))
    
    # Baseline return rates
    gamma_R: np.ndarray = field(default_factory=lambda: np.array([
        0.02,  # D receptor baseline return
        0.01,  # S receptor baseline return
        0.03,  # C receptor baseline return
        0.05,  # A receptor baseline return
        0.02,  # O receptor baseline return
        0.04,  # N receptor baseline return
        0.03,  # E receptor baseline return
    ]))
    
    # Default receptor sensitivities
    R_baseline: np.ndarray = field(default_factory=lambda: np.array([
        0.80,  # D receptor baseline
        0.90,  # S receptor baseline
        0.70,  # C receptor baseline
        0.60,  # A receptor baseline
        0.85,  # O receptor baseline
        0.75,  # N receptor baseline
        0.70,  # E receptor baseline
    ]))


# =============================================================================
# BASELINE ADAPTATION PARAMETERS
# =============================================================================
@dataclass  
class BaselineAdaptationParameters:
    """Adaptive baseline parameters (minimization principle)"""
    # Fast adaptation rates
    eta_fast: np.ndarray = field(default_factory=lambda: np.array([
        0.010,  # Dopamine fast adaptation
        0.005,  # Serotonin fast adaptation
        0.020,  # Cortisol fast adaptation
        0.050,  # Adrenaline fast adaptation
        0.008,  # Oxytocin fast adaptation
        0.015,  # Norepinephrine fast adaptation
        0.012,  # Endorphins fast adaptation
    ]))
    
    # Slow adaptation rates
    eta_slow: np.ndarray = field(default_factory=lambda: np.array([
        0.001,   # Dopamine slow adaptation
        0.0005,  # Serotonin slow adaptation
        0.002,   # Cortisol slow adaptation
        0.005,   # Adrenaline slow adaptation
        0.0008,  # Oxytocin slow adaptation
        0.0015,  # Norepinephrine slow adaptation
        0.0012,  # Endorphins slow adaptation
    ]))
    
    # Time constants for adaptation
    tau_fast: float = 60.0    # 1 minute
    tau_slow: float = 3600.0  # 1 hour


# =============================================================================
# INTERACTION MATRIX
# =============================================================================
class InteractionMatrix:
    """Inter-hormone interaction strengths"""
    # Base interaction matrix (7x7)
    # Rows = source hormone, Columns = target hormone
    base = np.array([
        # D     S     C     A     O     N     E
        [ 0.0,  0.3, -0.4,  0.2,  0.3,  0.5,  0.4],  # D affects others
        [ 0.3,  0.0, -0.6, -0.3,  0.5,  0.1,  0.4],  # S affects others
        [-0.4, -0.6,  0.0,  0.6, -0.5,  0.3, -0.3],  # C affects others
        [ 0.2, -0.3,  0.6,  0.0, -0.2,  0.7,  0.1],  # A affects others
        [ 0.3,  0.5, -0.5, -0.1,  0.0,  0.0,  0.3],  # O affects others
        [ 0.5,  0.1,  0.3,  0.6,  0.0,  0.0,  0.2],  # N affects others
        [ 0.4,  0.4, -0.3,  0.1,  0.3,  0.2,  0.0],  # E affects others
    ])
    
    @classmethod
    def get_interaction(cls, source_idx: int, target_idx: int) -> float:
        """Get interaction strength from source to target hormone"""
        return cls.base[source_idx, target_idx]


# =============================================================================
# CIRCADIAN RHYTHM PARAMETERS
# =============================================================================
@dataclass
class CircadianParameters:
    """Circadian rhythm parameters for each hormone"""
    # Amplitudes of circadian oscillation
    amplitude: np.ndarray = field(default_factory=lambda: np.array([
        10.0,  # Dopamine amplitude
        8.0,   # Serotonin amplitude
        15.0,  # Cortisol amplitude (strongest circadian)
        5.0,   # Adrenaline amplitude
        5.0,   # Oxytocin amplitude
        8.0,   # Norepinephrine amplitude
        10.0,  # Endorphins amplitude
    ]))
    
    # Phase offsets (in radians)
    phase: np.ndarray = field(default_factory=lambda: np.array([
        np.pi/3,      # Dopamine peaks at 10 AM
        np.pi/2,      # Serotonin peaks at 12 PM
        0.0,          # Cortisol peaks at 6 AM
        np.pi/4,      # Adrenaline peaks at 9 AM
        3*np.pi/4,    # Oxytocin peaks at 6 PM
        np.pi/3,      # Norepinephrine peaks at 10 AM
        2*np.pi/3,    # Endorphins peaks at 2 PM
    ]))
    
    period: float = 86400.0  # 24 hours in seconds


# =============================================================================
# METABOLIC PARAMETERS
# =============================================================================
@dataclass
class MetabolicParameters:
    """Resource and energy constraint parameters"""
    # Maximum pools
    omega_max: float = 500.0       # Maximum total neurotransmitter units
    P_tyr_max: float = 100.0       # Maximum tyrosine pool
    P_trp_max: float = 100.0       # Maximum tryptophan pool
    E_ATP_max: float = 100.0       # Maximum ATP energy
    
    # Production rate constants
    k_prod: np.ndarray = field(default_factory=lambda: np.array([
        0.5,  # Dopamine production rate
        0.4,  # Serotonin production rate
        0.6,  # Cortisol production rate
        0.8,  # Adrenaline production rate
        0.3,  # Oxytocin production rate
        0.6,  # Norepinephrine production rate
        0.5,  # Endorphins production rate
    ]))
    
    # Degradation rate constants
    k_deg: np.ndarray = field(default_factory=lambda: np.array([
        0.3,  # Dopamine degradation
        0.2,  # Serotonin degradation
        0.4,  # Cortisol degradation
        0.7,  # Adrenaline degradation
        0.25, # Oxytocin degradation
        0.5,  # Norepinephrine degradation
        0.6,  # Endorphins degradation
    ]))
    
    # Michaelis-Menten constants for production
    K_tyr: float = 20.0  # Tyrosine half-saturation
    K_trp: float = 15.0  # Tryptophan half-saturation
    K_ATP: float = 30.0  # ATP half-saturation


# =============================================================================
# COST FUNCTION PARAMETERS
# =============================================================================
@dataclass
class CostParameters:
    """Parameters for the minimization cost function"""
    # Deviation cost weights
    alpha_deviation: np.ndarray = field(default_factory=lambda: np.array([
        1.0,  # Dopamine deviation cost
        1.2,  # Serotonin deviation cost (more expensive)
        0.8,  # Cortisol deviation cost (cheaper)
        0.6,  # Adrenaline deviation cost (cheapest)
        1.1,  # Oxytocin deviation cost
        0.9,  # Norepinephrine deviation cost
        1.0,  # Endorphins deviation cost
    ]))
    
    # Change cost weights (metabolic cost of rapid changes)
    beta_change: np.ndarray = field(default_factory=lambda: np.array([
        0.5,  # Dopamine change cost
        0.6,  # Serotonin change cost
        0.4,  # Cortisol change cost
        0.3,  # Adrenaline change cost
        0.55, # Oxytocin change cost
        0.45, # Norepinephrine change cost
        0.5,  # Endorphins change cost
    ]))
    
    # Metabolic cost weight
    gamma_metabolic: float = 2.0
    
    # Uncertainty cost weight
    epsilon_uncertainty: float = 3.0
    
    # Saturation thresholds for cost calculation
    K_cost: np.ndarray = field(default_factory=lambda: np.array([
        60.0,  # Dopamine cost threshold
        70.0,  # Serotonin cost threshold
        50.0,  # Cortisol cost threshold
        40.0,  # Adrenaline cost threshold
        65.0,  # Oxytocin cost threshold
        55.0,  # Norepinephrine cost threshold
        60.0,  # Endorphins cost threshold
    ]))


# =============================================================================
# ALLOSTATIC LOAD PARAMETERS
# =============================================================================
@dataclass
class AllostaticParameters:
    """Chronic stress and allostatic load parameters"""
    L_max: float = 100.0           # Maximum allostatic load
    L_threshold: float = 30.0      # Threshold for negative effects
    C_threshold: float = 50.0      # Cortisol threshold for load accumulation
    
    kappa_L: float = 0.001         # Load accumulation rate
    lambda_L: float = 0.0001       # Load recovery rate
    tau_L: float = 3600.0          # Load memory time constant (1 hour)
    
    # Effects of high load (reduction factors)
    homeostasis_impairment: float = 0.5
    production_impairment: float = 0.3
    baseline_shift: float = 0.2
    receptor_dysfunction: float = 0.2


# =============================================================================
# NOISE PARAMETERS
# =============================================================================
@dataclass
class NoiseParameters:
    """Stochastic noise parameters"""
    # Base noise levels
    sigma_0: np.ndarray = field(default_factory=lambda: np.array([
        0.5,  # Dopamine base noise
        0.3,  # Serotonin base noise
        0.6,  # Cortisol base noise
        0.8,  # Adrenaline base noise
        0.4,  # Oxytocin base noise
        0.6,  # Norepinephrine base noise
        0.5,  # Endorphins base noise
    ]))
    
    # Velocity-dependent noise scaling
    sigma_1: np.ndarray = field(default_factory=lambda: np.array([
        0.3,  # Dopamine velocity noise
        0.2,  # Serotonin velocity noise
        0.4,  # Cortisol velocity noise
        0.5,  # Adrenaline velocity noise
        0.2,  # Oxytocin velocity noise
        0.3,  # Norepinephrine velocity noise
        0.3,  # Endorphins velocity noise
    ]))
    
    # State-dependent noise scaling
    sigma_2: np.ndarray = field(default_factory=lambda: np.array([
        0.2,  # Dopamine state noise
        0.1,  # Serotonin state noise
        0.3,  # Cortisol state noise
        0.4,  # Adrenaline state noise
        0.2,  # Oxytocin state noise
        0.3,  # Norepinephrine state noise
        0.2,  # Endorphins state noise
    ]))
    
    # Noise correlation time constants
    tau_noise: np.ndarray = field(default_factory=lambda: np.array([
        10.0,  # Dopamine noise correlation
        20.0,  # Serotonin noise correlation
        15.0,  # Cortisol noise correlation
        5.0,   # Adrenaline noise correlation
        25.0,  # Oxytocin noise correlation
        12.0,  # Norepinephrine noise correlation
        18.0,  # Endorphins noise correlation
    ]))


# =============================================================================
# NUMERICAL INTEGRATION PARAMETERS
# =============================================================================
@dataclass
class NumericalParameters:
    """Parameters for numerical integration"""
    dt_initial: float = 0.1        # Initial timestep (seconds)
    dt_min: float = 0.001          # Minimum timestep
    dt_max: float = 1.0            # Maximum timestep
    
    rtol: float = 1e-6             # Relative tolerance
    atol: float = 1e-8             # Absolute tolerance
    
    max_iterations: int = 10000    # Maximum solver iterations
    
    # Bounds
    hormone_min: float = 0.0       # Minimum hormone level
    hormone_max: float = 100.0     # Maximum hormone level
    receptor_min: float = 0.0      # Minimum receptor sensitivity
    receptor_max: float = 1.0      # Maximum receptor sensitivity


# =============================================================================
# STATE TRANSITION PARAMETERS
# =============================================================================
@dataclass
class StateTransitionParameters:
    """Parameters for mood state transitions"""
    # Stable state definitions (7D vectors)
    states: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'euthymic': np.array([50, 60, 30, 20, 40, 45, 35]),
        'depressed': np.array([20, 25, 70, 15, 20, 30, 15]),
        'manic': np.array([85, 70, 20, 40, 50, 65, 60]),
        'anxious': np.array([40, 35, 65, 50, 25, 70, 20]),
        'content': np.array([60, 70, 25, 15, 60, 40, 50]),
        'stressed': np.array([35, 40, 75, 60, 20, 80, 25]),
    })
    
    # Temperature for transition probability
    T_base: float = 10.0
    
    # Potential well depths
    well_depth: float = 50.0
    
    # Transition barrier heights
    barrier_height: float = 30.0


# =============================================================================
# INPUT SENSITIVITY PARAMETERS
# =============================================================================
@dataclass
class InputSensitivityParameters:
    """How external inputs affect the system"""
    # Reward sensitivity
    reward_weight: np.ndarray = field(default_factory=lambda: np.array([
        0.8,   # Dopamine response to reward
        0.4,   # Serotonin response to reward
        -0.2,  # Cortisol response to reward
        0.1,   # Adrenaline response to reward
        0.3,   # Oxytocin response to reward
        0.2,   # Norepinephrine response to reward
        0.5,   # Endorphins response to reward
    ]))
    
    # Threat sensitivity
    threat_weight: np.ndarray = field(default_factory=lambda: np.array([
        -0.3,  # Dopamine response to threat
        -0.4,  # Serotonin response to threat
        0.9,   # Cortisol response to threat
        0.8,   # Adrenaline response to threat
        -0.2,  # Oxytocin response to threat
        0.7,   # Norepinephrine response to threat
        -0.1,  # Endorphins response to threat
    ]))
    
    # Social sensitivity
    social_weight: np.ndarray = field(default_factory=lambda: np.array([
        0.3,   # Dopamine response to social
        0.5,   # Serotonin response to social
        -0.3,  # Cortisol response to social
        0.0,   # Adrenaline response to social
        0.9,   # Oxytocin response to social
        0.1,   # Norepinephrine response to social
        0.4,   # Endorphins response to social
    ]))
    
    # Exercise sensitivity
    exercise_weight: np.ndarray = field(default_factory=lambda: np.array([
        0.4,   # Dopamine response to exercise
        0.3,   # Serotonin response to exercise
        -0.1,  # Cortisol response to exercise
        0.3,   # Adrenaline response to exercise
        0.2,   # Oxytocin response to exercise
        0.4,   # Norepinephrine response to exercise
        0.8,   # Endorphins response to exercise
    ]))


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================
@dataclass
class NeurochemistryConfig:
    """Master configuration containing all parameters"""
    hill: HillParameters = field(default_factory=HillParameters)
    homeostasis: HomeostasisParameters = field(default_factory=HomeostasisParameters)
    receptors: ReceptorParameters = field(default_factory=ReceptorParameters)
    baseline_adaptation: BaselineAdaptationParameters = field(default_factory=BaselineAdaptationParameters)
    interactions: InteractionMatrix = InteractionMatrix
    circadian: CircadianParameters = field(default_factory=CircadianParameters)
    metabolic: MetabolicParameters = field(default_factory=MetabolicParameters)
    cost: CostParameters = field(default_factory=CostParameters)
    allostatic: AllostaticParameters = field(default_factory=AllostaticParameters)
    noise: NoiseParameters = field(default_factory=NoiseParameters)
    numerical: NumericalParameters = field(default_factory=NumericalParameters)
    state_transitions: StateTransitionParameters = field(default_factory=StateTransitionParameters)
    input_sensitivity: InputSensitivityParameters = field(default_factory=InputSensitivityParameters)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure all arrays have correct dimensions
        assert len(self.hill.n) == 7, "Hill n must have 7 elements"
        assert len(self.hill.K) == 7, "Hill K must have 7 elements"
        assert self.interactions.base.shape == (7, 7), "Interaction matrix must be 7x7"
        
        # Ensure parameters are within valid ranges
        assert np.all(self.hill.n > 0), "Hill coefficients must be positive"
        assert np.all(self.hill.K > 0), "Half-saturation constants must be positive"
        assert np.all(self.homeostasis.lambda_decay > 0), "Decay rates must be positive"
        assert np.all(self.receptors.R_baseline >= 0) and np.all(self.receptors.R_baseline <= 1), \
            "Receptor baselines must be in [0, 1]"


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================
DEFAULT_CONFIG = NeurochemistryConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_hormone_name(idx: int) -> str:
    """Get hormone name from index"""
    return HORMONES[idx]


def get_hormone_index(name: str) -> int:
    """Get hormone index from name"""
    return HORMONES.index(name.lower())


def get_initial_state() -> np.ndarray:
    """Get initial 7D hormone state"""
    return DEFAULT_CONFIG.homeostasis.baseline_initial.copy()


def get_initial_receptors() -> np.ndarray:
    """Get initial receptor sensitivities"""
    return DEFAULT_CONFIG.receptors.R_baseline.copy()


def get_initial_baselines() -> np.ndarray:
    """Get initial adaptive baselines"""
    return DEFAULT_CONFIG.homeostasis.baseline_initial.copy()