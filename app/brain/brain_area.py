"""
Brain Area - Pure neural network for tool activation signals
No hints, no hardcoding - just learned signals
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Tool:
    name: str
    description: str = ""


@dataclass
class AreaOutput:
    area_name: str
    signal_received: float
    tool_activations: Dict[str, float]
    active_tools: List[str]


class BrainArea:
    """Neural network that learns tool activations from signals"""
    
    def __init__(self, name: str, tools: List[str], input_size: int = 128, hidden_size: int = 64):
        self.name = name
        self.tool_names = tools
        self.num_tools = len(tools)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Neural network weights (Xavier init)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, self.num_tools) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(self.num_tools)
        
        # Learning state
        self.learning_rate = 0.01
        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        
        # Inter-area connections
        self.connections: Dict[str, float] = {}
        
        # Neuromodulation state
        self.neuro_state = {"dopamine": 0.5, "norepinephrine": 0.5}
        
        # Stats
        self.total_activations = 0
        self.tool_usage_count = {t: 0 for t in self.tool_names}

    def connect_to(self, area_name: str, strength: float):
        self.connections[area_name] = np.clip(strength, 0, 1)

    def set_neuro_state(self, state: Dict[str, float]):
        self.neuro_state.update(state)

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _apply_neuro_modulation(self, activations: np.ndarray) -> np.ndarray:
        dopamine = self.neuro_state.get("dopamine", 0.5)
        
        # Dopamine modulates activation strength
        if dopamine > 0.5:
            activations = activations * (1 + (dopamine - 0.5) * 0.5)
        
        return np.clip(activations, 0, 1)

    def process(self, input_signal: np.ndarray, area_activation: float) -> AreaOutput:
        """Process signal through neural network, output tool activations"""
        
        # Scale input by signal strength from thalamus
        scaled_input = input_signal * area_activation
        
        # Pad or truncate to input size
        if len(scaled_input) < self.input_size:
            scaled_input = np.pad(scaled_input, (0, self.input_size - len(scaled_input)))
        else:
            scaled_input = scaled_input[:self.input_size]
        
        # Forward pass
        self.last_input = scaled_input
        hidden = self._relu(scaled_input @ self.W1 + self.b1)
        self.last_hidden = hidden
        
        raw_output = self._sigmoid(hidden @ self.W2 + self.b2)
        tool_activations = self._apply_neuro_modulation(raw_output)
        self.last_output = tool_activations
        
        # Build output
        tool_activation_dict = {}
        active_tools = []
        threshold = 0.4
        
        for i, tool_name in enumerate(self.tool_names):
            activation = float(tool_activations[i])
            tool_activation_dict[tool_name] = activation
            
            if activation > threshold:
                active_tools.append(tool_name)
                self.tool_usage_count[tool_name] += 1
        
        self.total_activations += 1
        
        return AreaOutput(
            area_name=self.name,
            signal_received=area_activation,
            tool_activations=tool_activation_dict,
            active_tools=active_tools
        )

    def learn(self, reward: float) -> Dict[str, Any]:
        """Hebbian learning from reward signal"""
        if self.last_input is None or self.last_output is None:
            return {"status": "no_data"}
        
        effective_lr = self.learning_rate * (0.5 + reward)
        dW2 = effective_lr * reward * np.outer(self.last_hidden, self.last_output)
        self.W2 += dW2
        
        return {
            "reward": reward,
            "weight_update": float(np.abs(dW2).mean())
        }

    def get_stats(self) -> Dict:
        return {
            "name": self.name,
            "tools": self.tool_names,
            "total_activations": self.total_activations,
            "tool_usage": self.tool_usage_count
        }


# Specific brain areas with their tools
class CodeArea(BrainArea):
    def __init__(self):
        super().__init__("code", ["sandbox", "linter", "debugger", "formatter", "git"])


class MathArea(BrainArea):
    def __init__(self):
        super().__init__("math", ["solver", "symbolic", "plotter", "calculator"])


class MemoryArea(BrainArea):
    def __init__(self):
        super().__init__("memory", ["vector_search", "keyword_search", "graph_query"])
