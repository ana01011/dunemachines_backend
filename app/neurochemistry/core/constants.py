"""
Neurochemistry Constants
"""
import numpy as np

D_IDX = 0
S_IDX = 1
C_IDX = 2
A_IDX = 3
O_IDX = 4
N_IDX = 5
E_IDX = 6
NUM_HORMONES = 7

HORMONE_NAMES = ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 'oxytocin', 'norepinephrine', 'endorphins']

BASELINE_INITIAL = np.array([0.5, 0.5, 0.3, 0.3, 0.4, 0.4, 0.4])
LAMBDA_DECAY = np.array([0.1, 0.05, 0.08, 0.15, 0.06, 0.12, 0.07])
K_PROD = np.array([0.2, 0.15, 0.25, 0.3, 0.1, 0.25, 0.15])
K_DEG = np.array([0.1, 0.08, 0.12, 0.15, 0.05, 0.1, 0.08])

HILL_N = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
HILL_K = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

HORMONE_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HORMONE_MAX = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
RECEPTOR_MIN = 0.1
RECEPTOR_MAX = 2.0

P_TYR_MAX = 1.0
P_TRP_MAX = 1.0
E_ATP_MAX = 1.0

L_MAX = 1.0
L_THRESHOLD = 0.7

ALPHA_DEVIATION = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
BETA_CHANGE = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
GAMMA_METABOLIC = 0.3

DT_INITIAL = 0.1
RECEPTOR_DECAY = 0.05
RECEPTOR_SENSITIVITY = 0.1

BASELINE_ADAPTATION_RATE = 0.01
BASELINE_MIN = 0.2
BASELINE_MAX = 0.8

NOISE_SCALE = 0.02

K_COST = {'deviation': 1.0, 'change': 0.5, 'metabolic': 0.3, 'uncertainty': 0.2, 'allostatic': 0.4, 'extremity': 0.3}

MOOD_THRESHOLDS = {'high': 0.7, 'low': 0.3, 'neutral_low': 0.4, 'neutral_high': 0.6}

INTERACTION_MATRIX = np.array([
    [0.0, -0.1, 0.1, 0.2, -0.1, 0.2, 0.1],
    [-0.1, 0.0, -0.2, -0.1, 0.2, -0.1, 0.1],
    [0.1, -0.2, 0.0, 0.3, -0.2, 0.2, -0.1],
    [0.2, -0.1, 0.2, 0.0, -0.1, 0.3, -0.1],
    [-0.1, 0.2, -0.2, -0.1, 0.0, -0.1, 0.2],
    [0.2, -0.1, 0.1, 0.2, -0.1, 0.0, 0.0],
    [0.1, 0.1, -0.1, -0.1, 0.2, 0.0, 0.0],
])
