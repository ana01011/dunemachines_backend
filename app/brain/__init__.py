"""
Brain module - Neural routing and processing
"""
from app.brain.thalamus import Thalamus, BrainArea, ThalamusOutput, create_thalamus
from app.brain.brain_area import BrainArea as BrainAreaNN, AreaOutput, CodeArea, MathArea, MemoryArea
from app.brain.learning import HebbianLearner, RewardSignal

__all__ = [
    "Thalamus",
    "BrainArea", 
    "ThalamusOutput",
    "create_thalamus",
    "BrainAreaNN",
    "AreaOutput",
    "CodeArea",
    "MathArea", 
    "MemoryArea",
    "HebbianLearner",
    "RewardSignal"
]
