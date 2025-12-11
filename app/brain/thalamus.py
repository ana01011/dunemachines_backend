"""
Thalamus - Central Router for the Multi-Brain System
"""
import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BrainArea(str, Enum):
    CODE = "code"
    MATH = "math"
    MEMORY = "memory"
    PHYSICS = "physics"
    LANGUAGE = "language"


@dataclass
class ActivationPattern:
    id: str
    areas: Dict[BrainArea, float]
    strength: float = 0.5
    success_count: int = 0
    fail_count: int = 0
    
    def get_success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class ThalamusOutput:
    activations: Dict[BrainArea, float]
    active_areas: List[BrainArea]
    pattern_used: Optional[str] = None
    confidence: float = 0.5
    routing_time_ms: float = 0.0


class Thalamus:
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 512,
        num_areas: int = 5,
        learning_rate: float = 0.01,
        activation_threshold: float = 0.5
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_areas = num_areas
        self.learning_rate = learning_rate
        self.activation_threshold = activation_threshold
        
        self.area_list = list(BrainArea)[:num_areas]
        self.area_to_idx = {area: i for i, area in enumerate(self.area_list)}
        
        scale1 = np.sqrt(2.0 / (input_size + hidden_size))
        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        
        scale2 = np.sqrt(2.0 / (hidden_size + num_areas))
        self.W2 = np.random.randn(hidden_size, num_areas) * scale2
        self.b2 = np.zeros(num_areas)
        
        self.inter_area_weights = np.eye(num_areas) * 0.5
        self._init_inter_area_connections()
        
        self.patterns: Dict[str, ActivationPattern] = {}
        self.last_input: Optional[np.ndarray] = None
        self.last_hidden: Optional[np.ndarray] = None
        self.last_output: Optional[np.ndarray] = None
        self.last_pattern_id: Optional[str] = None
        
        self.total_routings = 0
        self.total_learnings = 0
        
        self.neuro_state = {
            "dopamine": 0.5,
            "norepinephrine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.5
        }
    
    def _init_inter_area_connections(self):
        if BrainArea.CODE in self.area_to_idx and BrainArea.MATH in self.area_to_idx:
            i, j = self.area_to_idx[BrainArea.CODE], self.area_to_idx[BrainArea.MATH]
            self.inter_area_weights[i, j] = 0.3
            self.inter_area_weights[j, i] = 0.3
        
        if BrainArea.MATH in self.area_to_idx and BrainArea.PHYSICS in self.area_to_idx:
            i, j = self.area_to_idx[BrainArea.MATH], self.area_to_idx[BrainArea.PHYSICS]
            self.inter_area_weights[i, j] = 0.4
            self.inter_area_weights[j, i] = 0.4
        
        if BrainArea.MEMORY in self.area_to_idx:
            mem_idx = self.area_to_idx[BrainArea.MEMORY]
            for i in range(self.num_areas):
                if i != mem_idx:
                    self.inter_area_weights[mem_idx, i] = 0.2
                    self.inter_area_weights[i, mem_idx] = 0.2
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _apply_neuro_modulation(self, activations: np.ndarray) -> np.ndarray:
        dopamine = self.neuro_state.get("dopamine", 0.5)
        norepinephrine = self.neuro_state.get("norepinephrine", 0.5)
        cortisol = self.neuro_state.get("cortisol", 0.5)
        
        da_factor = 1.0 + (dopamine - 0.5) * 0.5
        activations = activations * da_factor
        
        if norepinephrine > 0.5:
            mean_act = np.mean(activations)
            activations = np.where(
                activations > mean_act,
                activations * (1 + (norepinephrine - 0.5) * 0.3),
                activations * (1 - (norepinephrine - 0.5) * 0.3)
            )
        
        if cortisol > 0.6:
            threshold = 0.3 + cortisol * 0.2
            activations = np.where(activations < threshold, activations * 0.5, activations)
        
        return np.clip(activations, 0, 1)
    
    def _apply_inter_area_influence(self, activations: np.ndarray, cycles: int = 3) -> np.ndarray:
        for _ in range(cycles):
            lateral_input = self.inter_area_weights @ activations
            activations = 0.7 * activations + 0.3 * self._sigmoid(lateral_input)
        return activations
    
    def route(self, input_signal: np.ndarray, pfc_hints: Optional[Dict[BrainArea, float]] = None) -> ThalamusOutput:
        start_time = time.time()
        
        input_signal = np.asarray(input_signal).flatten()
        if len(input_signal) != self.input_size:
            if len(input_signal) < self.input_size:
                input_signal = np.pad(input_signal, (0, self.input_size - len(input_signal)))
            else:
                input_signal = input_signal[:self.input_size]
        
        input_signal = input_signal / (np.linalg.norm(input_signal) + 1e-8)
        self.last_input = input_signal.copy()
        
        hidden = self._relu(input_signal @ self.W1 + self.b1)
        self.last_hidden = hidden.copy()
        
        raw_activations = self._sigmoid(hidden @ self.W2 + self.b2)
        
        if pfc_hints:
            for area, hint in pfc_hints.items():
                if area in self.area_to_idx:
                    idx = self.area_to_idx[area]
                    raw_activations[idx] = 0.6 * raw_activations[idx] + 0.4 * hint
        
        activations = self._apply_neuro_modulation(raw_activations)
        activations = self._apply_inter_area_influence(activations)
        self.last_output = activations.copy()
        
        pattern_id = self._find_matching_pattern(activations)
        if pattern_id:
            pattern = self.patterns[pattern_id]
            for area, strength in pattern.areas.items():
                if area in self.area_to_idx:
                    idx = self.area_to_idx[area]
                    activations[idx] = max(activations[idx], strength * pattern.strength)
            self.last_pattern_id = pattern_id
        else:
            self.last_pattern_id = None
        
        active_areas = []
        activation_dict = {}
        for area in self.area_list:
            idx = self.area_to_idx[area]
            activation_dict[area] = float(activations[idx])
            if activations[idx] >= self.activation_threshold:
                active_areas.append(area)
        
        if not active_areas:
            max_idx = np.argmax(activations)
            active_areas.append(self.area_list[max_idx])
        
        confidence = float(np.max(activations))
        self.total_routings += 1
        routing_time = (time.time() - start_time) * 1000
        
        return ThalamusOutput(
            activations=activation_dict,
            active_areas=active_areas,
            pattern_used=pattern_id,
            confidence=confidence,
            routing_time_ms=routing_time
        )
    
    def _find_matching_pattern(self, activations: np.ndarray, threshold: float = 0.7) -> Optional[str]:
        best_match = None
        best_score = threshold
        
        for pattern_id, pattern in self.patterns.items():
            pattern_vec = np.zeros(self.num_areas)
            for area, strength in pattern.areas.items():
                if area in self.area_to_idx:
                    pattern_vec[self.area_to_idx[area]] = strength
            
            sim = np.dot(activations, pattern_vec) / (np.linalg.norm(activations) * np.linalg.norm(pattern_vec) + 1e-8)
            score = sim * pattern.strength * pattern.get_success_rate()
            
            if score > best_score:
                best_score = score
                best_match = pattern_id
        
        return best_match
    
    def learn(self, reward: float, areas_used: List[BrainArea]) -> Dict[str, Any]:
        if self.last_input is None or self.last_hidden is None:
            return {"error": "No forward pass to learn from"}
        
        dopamine = self.neuro_state.get("dopamine", 0.5)
        cortisol = self.neuro_state.get("cortisol", 0.5)
        effective_lr = self.learning_rate * (0.5 + dopamine) * (1.5 - cortisol)
        
        target = np.zeros(self.num_areas)
        for area in areas_used:
            if area in self.area_to_idx:
                target[self.area_to_idx[area]] = 1.0
        
        error = reward * (target - self.last_output)
        
        dW2 = effective_lr * np.outer(self.last_hidden, error)
        self.W2 += dW2
        self.b2 += effective_lr * error
        
        hidden_error = error @ self.W2.T * (self.last_hidden > 0).astype(float)
        dW1 = effective_lr * np.outer(self.last_input, hidden_error)
        self.W1 += dW1
        self.b1 += effective_lr * hidden_error
        
        if reward > 0:
            for i, area_i in enumerate(areas_used):
                for j, area_j in enumerate(areas_used):
                    if i != j and area_i in self.area_to_idx and area_j in self.area_to_idx:
                        idx_i, idx_j = self.area_to_idx[area_i], self.area_to_idx[area_j]
                        self.inter_area_weights[idx_i, idx_j] += effective_lr * reward * 0.1
                        self.inter_area_weights[idx_i, idx_j] = np.clip(self.inter_area_weights[idx_i, idx_j], 0, 1)
        
        self._update_pattern(areas_used, reward)
        self.total_learnings += 1
        
        return {"reward": reward, "learning_rate": effective_lr, "areas_updated": [a.value for a in areas_used]}
    
    def _update_pattern(self, areas_used: List[BrainArea], reward: float):
        pattern_key = "_".join(sorted([a.value for a in areas_used]))
        
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            if reward > 0:
                pattern.success_count += 1
                pattern.strength = min(1.0, pattern.strength + 0.05)
            else:
                pattern.fail_count += 1
                pattern.strength = max(0.1, pattern.strength - 0.03)
        else:
            areas_dict = {area: 0.8 for area in areas_used}
            self.patterns[pattern_key] = ActivationPattern(
                id=pattern_key, areas=areas_dict, strength=0.5 if reward > 0 else 0.3,
                success_count=1 if reward > 0 else 0, fail_count=0 if reward > 0 else 1
            )
    
    def set_neuro_state(self, neuro_state: Dict[str, float]):
        self.neuro_state.update(neuro_state)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size, "hidden_size": self.hidden_size,
            "num_areas": self.num_areas, "total_routings": self.total_routings,
            "total_learnings": self.total_learnings, "num_patterns": len(self.patterns)
        }
    
    def get_weights(self) -> Dict[str, Any]:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "inter_area_weights": self.inter_area_weights.tolist()
        }
    
    def set_weights(self, weights: Dict[str, Any]):
        self.W1 = np.array(weights["W1"])
        self.b1 = np.array(weights["b1"])
        self.W2 = np.array(weights["W2"])
        self.b2 = np.array(weights["b2"])
        self.inter_area_weights = np.array(weights["inter_area_weights"])


def create_thalamus(input_size: int = 256, hidden_size: int = 512, num_areas: int = 5) -> Thalamus:
    return Thalamus(input_size=input_size, hidden_size=hidden_size, num_areas=num_areas)
