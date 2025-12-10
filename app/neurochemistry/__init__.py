"""
Neurochemistry V3: Advanced 7D Neurochemical System
With minimization principle and biological realism
"""

from .core.state import NeurochemicalState
from .core.dynamics import NeurochemicalDynamics
from .interface import NeurochemicalSystem

__version__ = "3.0.0"
__all__ = ["NeurochemicalState", "NeurochemicalDynamics", "NeurochemicalSystem"]
