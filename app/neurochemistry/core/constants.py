"""
Core constants for the neurochemical system
"""
from enum import Enum
from typing import Final

# Hormone names
class Hormone(str, Enum):
    DOPAMINE = "dopamine"
    CORTISOL = "cortisol"
    ADRENALINE = "adrenaline"
    SEROTONIN = "serotonin"
    OXYTOCIN = "oxytocin"

# Event types that trigger neurochemical responses
class EventType(str, Enum):
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    HIGH_COMPLEXITY = "high_complexity"
    ANTICIPATION = "anticipation"
    SOCIAL_POSITIVE = "social_positive"
    SOCIAL_NEGATIVE = "social_negative"
    UNCERTAINTY = "uncertainty"
    TIME_PRESSURE = "time_pressure"
    NOVELTY = "novelty"
    ROUTINE = "routine"
    ERROR = "error"
    LEARNING = "learning"

# System bounds
HORMONE_MIN: Final = 0.0
HORMONE_MAX: Final = 100.0
BASELINE_MIN: Final = 10.0
BASELINE_MAX: Final = 90.0

# Soft bound thresholds
UPPER_SOFT_BOUND: Final = 80.0
LOWER_SOFT_BOUND: Final = 20.0

# Time constants (in seconds)
MIN_UPDATE_INTERVAL: Final = 0.1
MAX_UPDATE_INTERVAL: Final = 1.0

# Learning parameters
LEARNING_RATE: Final = 0.01
MEMORY_WINDOW: Final = 100  # Number of recent events to consider
ADAPTATION_THRESHOLD: Final = 10  # Minimum events before adapting

# Stability parameters
MAX_EIGENVALUE: Final = 0.95  # For system stability
CONVERGENCE_TOLERANCE: Final = 0.01
MAX_ITERATIONS: Final = 100

# Wave analysis
MIN_WAVES_FOR_BASELINE_SHIFT: Final = 3
MAX_WAVE_HISTORY: Final = 20
WAVE_AMPLITUDE_THRESHOLD: Final = 5.0  # Minimum amplitude to count as wave

# Behavioral bounds
PLANNING_DEPTH_MIN: Final = 1
PLANNING_DEPTH_MAX: Final = 10
RISK_TOLERANCE_MIN: Final = 0.1
RISK_TOLERANCE_MAX: Final = 0.9
CONFIDENCE_MIN: Final = 0.2
CONFIDENCE_MAX: Final = 0.95
PROCESSING_SPEED_MIN: Final = 0.5
PROCESSING_SPEED_MAX: Final = 2.0
