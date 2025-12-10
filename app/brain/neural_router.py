"""
Neural Router - Routes queries to specialist brain regions
"""
import numpy as np
import hashlib
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from app.brain.neuron import LIFNeuron, NeuronType


class BrainRegionType(str, Enum):
    LANGUAGE = "language"
    MATH = "math"
    MEMORY = "memory"


@dataclass
class RouterConfig:
    input_size: int = 768
    hidden_size: int = 128
    output_size: int = 3
    activation_threshold: float = 0.5
    learning_rate: float = 0.01
    weight_init_scale: float = 0.1
    use_bias: bool = True


@dataclass
class RouterOutput:
    activations: Dict[str, float]
    active_regions: List[BrainRegionType]
    confidence: float
    hidden_activations: np.ndarray
    output_raw: np.ndarray
    routing_time_ms: float
    query_hash: str
    decision_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activations": self.activations,
            "active_regions": [r.value for r in self.active_regions],
            "confidence": self.confidence,
            "routing_time_ms": self.routing_time_ms,
            "query_hash": self.query_hash,
            "decision_id": self.decision_id
        }


class NeuralRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.is_initialized = False
        self.total_decisions = 0
        self.total_learning_events = 0
        self.region_indices = {
            BrainRegionType.LANGUAGE: 0,
            BrainRegionType.MATH: 1,
            BrainRegionType.MEMORY: 2
        }
        self.index_to_region = {v: k for k, v in self.region_indices.items()}
        self._initialize_network()

    def _initialize_network(self):
        cfg = self.config
        scale = np.sqrt(2.0 / (cfg.input_size + cfg.hidden_size))
        self.W_input_hidden = np.random.randn(cfg.input_size, cfg.hidden_size) * scale
        scale = np.sqrt(2.0 / (cfg.hidden_size + cfg.output_size))
        self.W_hidden_output = np.random.randn(cfg.hidden_size, cfg.output_size) * scale
        if cfg.use_bias:
            self.b_hidden = np.zeros(cfg.hidden_size)
            self.b_output = np.zeros(cfg.output_size)
        else:
            self.b_hidden = None
            self.b_output = None
        self.hidden_neurons = [
            LIFNeuron(f"hidden_{i}", NeuronType.HIDDEN) for i in range(cfg.hidden_size)
        ]
        self.output_neurons = [
            LIFNeuron(f"output_{i}", NeuronType.OUTPUT) for i in range(cfg.output_size)
        ]
        self.eligibility_input_hidden = np.zeros_like(self.W_input_hidden)
        self.eligibility_hidden_output = np.zeros_like(self.W_hidden_output)
        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        self.is_initialized = True

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def forward(self, embedding: np.ndarray, query: str = "", use_neurons: bool = False) -> RouterOutput:
        start_time = time.time()
        embedding = np.asarray(embedding).flatten()
        if len(embedding) != self.config.input_size:
            raise ValueError(f"Expected embedding size {self.config.input_size}, got {len(embedding)}")
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.last_input = embedding.copy()
        hidden_pre = embedding @ self.W_input_hidden
        if self.b_hidden is not None:
            hidden_pre = hidden_pre + self.b_hidden
        if use_neurons:
            hidden_activations = np.zeros(self.config.hidden_size)
            for i, neuron in enumerate(self.hidden_neurons):
                neuron.receive_input(hidden_pre[i])
                neuron.step(dt=1.0)
                hidden_activations[i] = neuron.get_firing_rate()
        else:
            hidden_activations = self._relu(hidden_pre)
        self.last_hidden = hidden_activations.copy()
        output_pre = hidden_activations @ self.W_hidden_output
        if self.b_output is not None:
            output_pre = output_pre + self.b_output
        if use_neurons:
            output_raw = np.zeros(self.config.output_size)
            for i, neuron in enumerate(self.output_neurons):
                neuron.receive_input(output_pre[i])
                neuron.step(dt=1.0)
                output_raw[i] = neuron.get_firing_rate()
        else:
            output_raw = self._sigmoid(output_pre)
        self.last_output = output_raw.copy()
        activations = {
            "language": float(output_raw[0]),
            "math": float(output_raw[1]),
            "memory": float(output_raw[2])
        }
        active_regions = []
        threshold = self.config.activation_threshold
        if activations["language"] >= threshold:
            active_regions.append(BrainRegionType.LANGUAGE)
        if activations["math"] >= threshold:
            active_regions.append(BrainRegionType.MATH)
        if activations["memory"] >= threshold:
            active_regions.append(BrainRegionType.MEMORY)
        if not active_regions:
            active_regions.append(BrainRegionType.LANGUAGE)
        confidence = float(np.max(output_raw))
        self.total_decisions += 1
        routing_time_ms = (time.time() - start_time) * 1000
        outer_ih = np.outer(embedding, hidden_activations)
        self.eligibility_input_hidden = 0.9 * self.eligibility_input_hidden + 0.1 * outer_ih
        outer_ho = np.outer(hidden_activations, output_raw)
        self.eligibility_hidden_output = 0.9 * self.eligibility_hidden_output + 0.1 * outer_ho
        return RouterOutput(
            activations=activations,
            active_regions=active_regions,
            confidence=confidence,
            hidden_activations=hidden_activations,
            output_raw=output_raw,
            routing_time_ms=routing_time_ms,
            query_hash=self._hash_query(query) if query else ""
        )

    def learn(self, reward: float, learning_rate_modifier: float = 1.0, use_eligibility: bool = True) -> Dict[str, float]:
        if self.last_input is None or self.last_hidden is None:
            return {"error": "No forward pass to learn from"}
        lr = self.config.learning_rate * learning_rate_modifier
        if use_eligibility:
            dW_ih = lr * reward * self.eligibility_input_hidden
            dW_ho = lr * reward * self.eligibility_hidden_output
        else:
            dW_ih = lr * reward * np.outer(self.last_input, self.last_hidden)
            dW_ho = lr * reward * np.outer(self.last_hidden, self.last_output)
        self.W_input_hidden += dW_ih
        self.W_hidden_output += dW_ho
        weight_decay = 0.0001
        self.W_input_hidden *= (1 - weight_decay)
        self.W_hidden_output *= (1 - weight_decay)
        if self.b_hidden is not None:
            self.b_hidden += lr * reward * self.last_hidden * 0.1
        if self.b_output is not None:
            self.b_output += lr * reward * self.last_output * 0.1
        self.total_learning_events += 1
        change_magnitude = float(np.mean(np.abs(dW_ih)) + np.mean(np.abs(dW_ho)))
        return {
            "reward": reward,
            "learning_rate": lr,
            "weight_change": change_magnitude,
            "total_learning_events": self.total_learning_events
        }

    def modulate_from_neurochemistry(self, neuro_state: Dict[str, float]):
        dopamine = neuro_state.get("dopamine", 0.5)
        norepinephrine = neuro_state.get("norepinephrine", 0.5)
        cortisol = neuro_state.get("cortisol", 0.5)
        self.config.learning_rate = 0.01 * (0.5 + dopamine)
        self.config.activation_threshold = 0.5 - 0.2 * (norepinephrine - 0.5)
        self.config.activation_threshold = np.clip(self.config.activation_threshold, 0.3, 0.7)
        if cortisol > 0.7:
            self.config.learning_rate *= 0.5

    def get_weights(self) -> Dict[str, Any]:
        return {
            "W_input_hidden": self.W_input_hidden.tolist(),
            "W_hidden_output": self.W_hidden_output.tolist(),
            "b_hidden": self.b_hidden.tolist() if self.b_hidden is not None else None,
            "b_output": self.b_output.tolist() if self.b_output is not None else None,
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "output_size": self.config.output_size,
                "activation_threshold": self.config.activation_threshold,
                "learning_rate": self.config.learning_rate
            }
        }

    def set_weights(self, weights: Dict[str, Any]):
        self.W_input_hidden = np.array(weights["W_input_hidden"])
        self.W_hidden_output = np.array(weights["W_hidden_output"])
        if weights.get("b_hidden") is not None:
            self.b_hidden = np.array(weights["b_hidden"])
        if weights.get("b_output") is not None:
            self.b_output = np.array(weights["b_output"])
        if "config" in weights:
            cfg = weights["config"]
            self.config.input_size = cfg.get("input_size", 768)
            self.config.hidden_size = cfg.get("hidden_size", 128)
            self.config.output_size = cfg.get("output_size", 3)
            self.config.activation_threshold = cfg.get("activation_threshold", 0.5)
            self.config.learning_rate = cfg.get("learning_rate", 0.01)

    def reset_neurons(self):
        for neuron in self.hidden_neurons:
            neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()

    def get_stats(self) -> Dict[str, Any]:
        total_params = (
            self.W_input_hidden.size +
            self.W_hidden_output.size +
            (self.b_hidden.size if self.b_hidden is not None else 0) +
            (self.b_output.size if self.b_output is not None else 0)
        )
        return {
            "is_initialized": self.is_initialized,
            "architecture": f"{self.config.input_size}->{self.config.hidden_size}->{self.config.output_size}",
            "total_parameters": total_params,
            "total_decisions": self.total_decisions,
            "total_learning_events": self.total_learning_events,
            "activation_threshold": self.config.activation_threshold,
            "learning_rate": self.config.learning_rate,
            "hidden_neurons": len(self.hidden_neurons),
            "output_neurons": len(self.output_neurons)
        }

    def __repr__(self) -> str:
        return f"NeuralRouter(arch={self.config.input_size}->{self.config.hidden_size}->{self.config.output_size}, decisions={self.total_decisions})"


def create_neural_router(input_size: int = 768, hidden_size: int = 128, output_size: int = 3, learning_rate: float = 0.01) -> NeuralRouter:
    config = RouterConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate
    )
    return NeuralRouter(config)
