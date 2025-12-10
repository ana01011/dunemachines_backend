"""
Constants for Enhanced Neurochemistry System with Opponent Processes
"""

import numpy as np

# Hill Equation Parameters
HILL_COEFFICIENTS = {
    'dopamine': 2.5,
    'serotonin': 2.0,
    'cortisol': 3.0,
    'adrenaline': 4.0,
    'oxytocin': 2.0,
    'norepinephrine': 3.0,
    'endorphins': 2.5
}

HALF_SATURATION = {
    'dopamine': 40.0,
    'serotonin': 50.0,
    'cortisol': 45.0,
    'adrenaline': 35.0,
    'oxytocin': 30.0,
    'norepinephrine': 40.0,
    'endorphins': 25.0
}

# Homeostasis Rates (base decay rates)
HOMEOSTASIS_RATES = {
    'dopamine': 0.05,      # Moderate decay
    'serotonin': 0.03,     # Slow decay (stable mood)
    'cortisol': 0.08,      # Faster decay (stress response)
    'adrenaline': 0.15,    # Very fast decay (fight or flight)
    'oxytocin': 0.04,      # Slow decay (bonding)
    'norepinephrine': 0.10, # Fast decay (attention)
    'endorphins': 0.12     # Fast decay (pain relief)
}

# Receptor Parameters
RECEPTOR_PARAMS = {
    'dopamine': {'alpha': 0.05, 'beta': 0.15, 'gamma': 0.02, 'baseline': 0.8},
    'serotonin': {'alpha': 0.02, 'beta': 0.05, 'gamma': 0.01, 'baseline': 0.9},
    'cortisol': {'alpha': 0.08, 'beta': 0.10, 'gamma': 0.03, 'baseline': 0.7},
    'adrenaline': {'alpha': 0.15, 'beta': 0.30, 'gamma': 0.05, 'baseline': 0.6},
    'oxytocin': {'alpha': 0.03, 'beta': 0.08, 'gamma': 0.02, 'baseline': 0.85},
    'norepinephrine': {'alpha': 0.10, 'beta': 0.20, 'gamma': 0.04, 'baseline': 0.75},
    'endorphins': {'alpha': 0.08, 'beta': 0.25, 'gamma': 0.03, 'baseline': 0.7}
}

# Baseline Adaptation Parameters
BASELINE_ADAPTATION = {
    'tau_fast': 100.0,   # Fast adaptation time constant (seconds)
    'tau_slow': 1000.0,  # Slow adaptation time constant (seconds)
    'spike_threshold': 1.5,  # Spike detection threshold (x baseline)
    'spike_window': 30.0     # Window for spike averaging (seconds)
}

# Circadian Parameters
CIRCADIAN = {
    'dopamine': {'mean': 50, 'amplitude': 10, 'phase': np.pi/3},
    'serotonin': {'mean': 60, 'amplitude': 8, 'phase': np.pi/2},
    'cortisol': {'mean': 30, 'amplitude': 15, 'phase': 0},
    'adrenaline': {'mean': 20, 'amplitude': 5, 'phase': np.pi/4},
    'oxytocin': {'mean': 40, 'amplitude': 5, 'phase': 3*np.pi/4},
    'norepinephrine': {'mean': 45, 'amplitude': 8, 'phase': np.pi/3},
    'endorphins': {'mean': 35, 'amplitude': 10, 'phase': 2*np.pi/3}
}

# Metabolic Parameters
METABOLIC = {
    'tyrosine_max': 1.0,
    'tryptophan_max': 1.0,
    'atp_max': 1.0,
    'tyrosine_recovery': 0.001,
    'tryptophan_recovery': 0.001,
    'atp_recovery': 0.002
}

# Cost Function Weights (for minimization principle)
COST_WEIGHTS = {
    'production': 10.0,    # Main focus - minimize production
    'maintenance': 2.0,    # Cost to maintain levels
    'change': 1.0,        # Smoothness penalty
    'metabolic': 0.5,     # Resource depletion cost
    'uncertainty': 0.1,    # Prediction error cost
    'allostatic': 1.0,    # Chronic stress cost
    'extremity': 0.2      # Penalty for extreme values
}

# Allostatic Load Parameters
ALLOSTATIC = {
    'cortisol_threshold': 40.0,  # Threshold for load accumulation
    'accumulation_rate': 0.01,   # Rate of load buildup
    'recovery_rate': 0.001,       # Rate of load recovery
    'max_load': 1.0              # Maximum allostatic load
}

# Interaction Matrix Base Values
INTERACTION_MATRIX = {
    ('dopamine', 'serotonin'): 0.2,
    ('dopamine', 'cortisol'): -0.3,
    ('dopamine', 'adrenaline'): 0.1,
    ('dopamine', 'oxytocin'): 0.2,
    ('dopamine', 'norepinephrine'): 0.4,
    ('dopamine', 'endorphins'): 0.3,
    ('serotonin', 'dopamine'): 0.3,
    ('serotonin', 'cortisol'): -0.5,
    ('serotonin', 'adrenaline'): -0.2,
    ('serotonin', 'oxytocin'): 0.4,
    ('serotonin', 'norepinephrine'): 0.1,
    ('serotonin', 'endorphins'): 0.3,
    ('cortisol', 'dopamine'): -0.4,
    ('cortisol', 'serotonin'): -0.6,
    ('cortisol', 'adrenaline'): 0.5,
    ('cortisol', 'oxytocin'): -0.4,
    ('cortisol', 'norepinephrine'): 0.3,
    ('cortisol', 'endorphins'): -0.2,
    ('adrenaline', 'dopamine'): 0.2,
    ('adrenaline', 'serotonin'): -0.3,
    ('adrenaline', 'cortisol'): 0.6,
    ('adrenaline', 'oxytocin'): -0.2,
    ('adrenaline', 'norepinephrine'): 0.7,
    ('adrenaline', 'endorphins'): 0.1,
    ('oxytocin', 'dopamine'): 0.3,
    ('oxytocin', 'serotonin'): 0.5,
    ('oxytocin', 'cortisol'): -0.5,
    ('oxytocin', 'adrenaline'): -0.1,
    ('oxytocin', 'norepinephrine'): 0.0,
    ('oxytocin', 'endorphins'): 0.2,
    ('norepinephrine', 'dopamine'): 0.5,
    ('norepinephrine', 'serotonin'): 0.1,
    ('norepinephrine', 'cortisol'): 0.3,
    ('norepinephrine', 'adrenaline'): 0.6,
    ('norepinephrine', 'oxytocin'): 0.0,
    ('norepinephrine', 'endorphins'): 0.2,
    ('endorphins', 'dopamine'): 0.4,
    ('endorphins', 'serotonin'): 0.4,
    ('endorphins', 'cortisol'): -0.3,
    ('endorphins', 'adrenaline'): 0.1,
    ('endorphins', 'oxytocin'): 0.3,
    ('endorphins', 'norepinephrine'): 0.2
}

# Noise Parameters (for stochastic dynamics)
NOISE_PARAMS = {
    'base_sigma': 0.1,      # Base noise level
    'rate_scaling': 0.5,    # Noise scales with rate of change
    'level_scaling': 0.2    # Noise scales with hormone level
}

# Production Gain Parameters (for input responses)
PRODUCTION_GAINS = {
    'dopamine': {
        'reward': 30.0,
        'novelty': 20.0,
        'success': 15.0
    },
    'serotonin': {
        'success': 25.0,
        'social': 15.0,
        'contentment': 20.0
    },
    'cortisol': {
        'threat': 40.0,
        'uncertainty': 20.0,
        'failure': 25.0
    },
    'adrenaline': {
        'urgency': 50.0,
        'threat': 35.0,
        'excitement': 30.0
    },
    'oxytocin': {
        'social': 35.0,
        'trust': 25.0,
        'bonding': 30.0
    },
    'norepinephrine': {
        'attention': 30.0,
        'urgency': 20.0,
        'focus': 25.0
    },
    'endorphins': {
        'exercise': 45.0,
        'pleasure': 30.0,
        'relaxation': 20.0,
        'pain_relief': 35.0
    }
}

# Opponent Process Pairs (for recovery dynamics)
OPPONENT_PAIRS = {
    'dopamine': {
        'opponents': ['serotonin', 'oxytocin'],
        'supporters': ['norepinephrine'],
        'weights': {'serotonin': 1.0, 'oxytocin': 0.5, 'norepinephrine': 0.5}
    },
    'serotonin': {
        'opponents': ['dopamine', 'norepinephrine'],
        'supporters': ['oxytocin'],
        'weights': {'dopamine': 1.0, 'norepinephrine': 0.5, 'oxytocin': 0.3}
    },
    'cortisol': {
        'opponents': ['oxytocin', 'serotonin'],
        'supporters': ['adrenaline'],
        'weights': {'oxytocin': 1.0, 'serotonin': 0.5, 'adrenaline': 0.3}
    },
    'adrenaline': {
        'opponents': ['endorphins', 'oxytocin'],
        'supporters': ['norepinephrine', 'cortisol'],
        'weights': {'endorphins': 1.0, 'oxytocin': 0.5, 'norepinephrine': 0.5, 'cortisol': 0.3}
    },
    'oxytocin': {
        'opponents': ['cortisol', 'adrenaline'],
        'supporters': ['serotonin'],
        'weights': {'cortisol': 1.0, 'adrenaline': 0.5, 'serotonin': 0.3}
    },
    'norepinephrine': {
        'opponents': ['serotonin', 'endorphins'],
        'supporters': ['dopamine', 'adrenaline'],
        'weights': {'serotonin': 0.7, 'endorphins': 0.5, 'dopamine': 0.5, 'adrenaline': 0.3}
    },
    'endorphins': {
        'opponents': ['adrenaline', 'cortisol'],
        'supporters': ['oxytocin'],
        'weights': {'adrenaline': 1.0, 'cortisol': 0.5, 'oxytocin': 0.3}
    }
}
