"""
OMNIUS v2 - Multi-Brain Neural Router System
"""

from app.brain.neuron import (
    LIFNeuron,
    NeuronType,
    Synapse,
    create_neuron_layer
)

from app.brain.neural_router import (
    NeuralRouter,
    RouterConfig,
    RouterOutput,
    BrainRegionType,
    create_neural_router
)

from app.brain.learning import (
    HebbianLearner,
    RewardSignal
)

from app.brain.hippocampus import (
    Hippocampus,
    MemoryRecord,
    RetrievalResult,
    create_hippocampus
)

__version__ = "2.0.0"

__all__ = [
    "LIFNeuron",
    "NeuronType", 
    "Synapse",
    "create_neuron_layer",
    "NeuralRouter",
    "RouterConfig",
    "RouterOutput",
    "BrainRegionType",
    "create_neural_router",
    "HebbianLearner",
    "RewardSignal",
    "Hippocampus",
    "MemoryRecord",
    "RetrievalResult",
    "create_hippocampus",
]
