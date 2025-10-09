"""
Constants and parameters for neurochemical dynamics
Based on mathematical framework with proven stability
"""

# Return-to-baseline rates (lambda values)
LAMBDA_DOPAMINE = 0.1
LAMBDA_CORTISOL = 0.15  # Faster than dopamine
LAMBDA_ADRENALINE = 0.3  # Fastest decay
LAMBDA_SEROTONIN = 0.02  # Very slow decay
LAMBDA_OXYTOCIN = 0.05

# Dopamine parameters
REWARD_PREDICTION_GAIN = 0.5  # rho
CORTISOL_SUPPRESSION = 0.02   # beta
DOPAMINE_ADAPTATION_UP = 0.1   # kappa_1
DOPAMINE_ADAPTATION_DOWN = 0.15  # kappa_2
OPPONENT_PROCESS_STRENGTH = 0.4  # theta
CORTISOL_TRIGGER_GAIN = 0.5  # gamma

# Cortisol parameters
PREDICTION_ERROR_SENSITIVITY = 0.3  # phi
UNCERTAINTY_COEFFICIENT = 0.2  # psi
TIME_PRESSURE_COEFFICIENT = 0.1  # omega
SEROTONIN_SUPPRESSION = 0.01  # mu
CORTISOL_BASELINE_DOPAMINE_RISE = 0.02  # epsilon_1
CORTISOL_RELAXATION_RATE = 0.05  # epsilon_2
CHRONIC_STRESS_ADAPTATION = 0.01  # epsilon_3

# Adrenaline parameters
NOVELTY_RESPONSE = 0.4  # nu
DOPAMINE_RATE_COUPLING = 0.2  # zeta
CORTISOL_RATE_COUPLING = 0.15  # xi
ADRENALINE_REGEN_RATE = 5.0  # rho_regen
ADRENALINE_USAGE_RATE = 0.1  # delta_use

# Serotonin parameters
SUCCESS_INTEGRATION = 0.2  # tau_1
CONSISTENCY_BONUS = 0.15  # tau_2
FAILURE_PENALTY = 0.1  # tau_3

# Oxytocin parameters
SOCIAL_BONDING_RATE = 0.3
EMPATHY_COEFFICIENT = 0.25
TRUST_BUILDING_RATE = 0.1

# Interaction matrix coefficients
INTERACTION_MATRIX = {
    'beta_DC': 0.02,  # Dopamine suppressed by Cortisol
    'alpha_DA': 0.01,  # Dopamine boosted by Adrenaline
    'delta_DS': 0.015,  # Dopamine suppressed by low Serotonin
    'gamma_CD': 0.025,  # Cortisol triggered by Dopamine baseline
    'alpha_CA': 0.02,  # Cortisol boosted by Adrenaline
    'mu_CS': 0.01,  # Cortisol suppressed by Serotonin
    'zeta_AD': 0.03,  # Adrenaline triggered by Dopamine rate
    'xi_AC': 0.025,  # Adrenaline triggered by Cortisol rate
    'nu_AS': 0.02,  # Adrenaline suppressed by Serotonin
    'theta_SD': 0.01,  # Serotonin suppressed by low Dopamine
    'sigma_SC': 0.02,  # Serotonin suppressed by Cortisol
    'rho_SA': 0.015,  # Serotonin suppressed by Adrenaline
    'epsilon_SO': 0.02,  # Serotonin boosted by Oxytocin
    'kappa_OD': 0.015,  # Oxytocin boosted by Dopamine
    'lambda_OS': 0.025,  # Oxytocin boosted by Serotonin
}

# Baseline bounds
MIN_BASELINE = 10.0
MAX_BASELINE = 90.0
DOPAMINE_CEILING = 50.0

# Hormone bounds
MIN_HORMONE = 0.0
MAX_HORMONE = 100.0

# Noise parameters (for stochastic terms)
NOISE_AMPLITUDE = {
    'dopamine': 0.5,
    'cortisol': 0.3,
    'adrenaline': 0.8,
    'serotonin': 0.2,
    'oxytocin': 0.3
}

# Learning parameters
LEARNING_RATE = 0.01
EXPECTATION_ADAPTATION = 0.1
TREND_INCORPORATION = 0.05

# Homeostasis parameters
HOMEOSTASIS_STRENGTH = 0.05
FAST_HOMEOSTASIS = 0.1  # For adrenaline

# Window sizes for analysis
SPIKE_WINDOW = 10  # Number of recent spikes to analyze
PATTERN_WINDOW = 50  # Events for pattern recognition
BASELINE_ADAPTATION_WINDOW = 20
