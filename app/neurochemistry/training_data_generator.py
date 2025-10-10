"""
Generate comprehensive training data for multi-mood neural network
Covers all possible human emotional states in 3D VAD space
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
import json
from dataclasses import dataclass


@dataclass
class EmotionalAnchor:
    """Represents a known emotional state for training"""
    name: str
    vad: List[float]  # [valence, arousal, dominance]
    mood_vector: List[float]  # 20 component intensities


class TrainingDataGenerator:
    """
    Generates comprehensive training data covering all emotional states
    """
    
    def __init__(self):
        self.num_components = 20
        
    def generate_complete_dataset(self, samples_per_emotion: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete training dataset
        Returns: (inputs, targets) as torch tensors
        """
        all_inputs = []
        all_targets = []
        
        # 1. Core emotional anchors (must-have states)
        anchors = self._create_emotional_anchors()
        for anchor in anchors:
            # Add exact anchor
            all_inputs.append(anchor.vad)
            all_targets.append(anchor.mood_vector)
            
            # Add variations around anchor
            for _ in range(samples_per_emotion):
                # Small random perturbation
                noise = np.random.normal(0, 0.1, 3)
                varied_vad = np.clip(np.array(anchor.vad) + noise, [-1, 0, 0], [1, 1, 1])
                
                # Slightly varied mood vector
                mood_noise = np.random.normal(0, 0.05, self.num_components)
                varied_mood = np.clip(np.array(anchor.mood_vector) + mood_noise, 0, 1)
                
                all_inputs.append(varied_vad.tolist())
                all_targets.append(varied_mood.tolist())
        
        # 2. Grid sampling for smooth coverage
        grid_inputs, grid_targets = self._generate_grid_samples()
        all_inputs.extend(grid_inputs)
        all_targets.extend(grid_targets)
        
        # 3. Random sampling for diversity
        random_inputs, random_targets = self._generate_random_samples(5000)
        all_inputs.extend(random_inputs)
        all_targets.extend(random_targets)
        
        # Convert to tensors
        inputs = torch.tensor(all_inputs, dtype=torch.float32)
        targets = torch.tensor(all_targets, dtype=torch.float32)
        
        return inputs, targets
    
    def _create_emotional_anchors(self) -> List[EmotionalAnchor]:
        """
        Define core emotional states as training anchors
        """
        anchors = []
        
        # ANGER: High negative valence, high arousal, high dominance
        anchors.append(EmotionalAnchor(
            name="anger",
            vad=[-0.8, 0.85, 0.8],
            mood_vector=[
                0.95,  # attentive (laser-focused)
                0.90,  # energized
                0.85,  # tense
                0.75,  # assured
                0.10,  # warm (very low)
                0.80,  # restless
                0.70,  # protective
                0.15,  # curious (low)
                0.85,  # determined
                0.20,  # cautious (low)
                0.10,  # reflective (low)
                0.60,  # spontaneous
                0.30,  # methodical
                0.15,  # receptive (low)
                0.90,  # assertive
                0.40,  # conflicted
                0.30,  # transitioning
                0.20,  # anticipating
                0.25,  # processing
                0.10,  # resonating (low)
            ]
        ))
        
        # FEAR: High negative valence, high arousal, low dominance
        anchors.append(EmotionalAnchor(
            name="fear",
            vad=[-0.85, 0.9, 0.15],
            mood_vector=[
                0.98,  # attentive (hypervigilant)
                0.85,  # energized
                0.95,  # tense
                0.10,  # assured (very low)
                0.15,  # warm (low)
                0.90,  # restless
                0.95,  # protective
                0.20,  # curious (low)
                0.30,  # determined (low)
                0.98,  # cautious (very high)
                0.20,  # reflective
                0.70,  # spontaneous
                0.15,  # methodical (low)
                0.25,  # receptive
                0.10,  # assertive (very low)
                0.60,  # conflicted
                0.80,  # transitioning
                0.90,  # anticipating
                0.70,  # processing
                0.20,  # resonating
            ]
        ))
        
        # JOY: High positive valence, high arousal, high dominance
        anchors.append(EmotionalAnchor(
            name="joy",
            vad=[0.9, 0.75, 0.75],
            mood_vector=[
                0.70,  # attentive
                0.85,  # energized
                0.05,  # tense (very low)
                0.85,  # assured
                0.95,  # warm (very high)
                0.30,  # restless (some excitement)
                0.10,  # protective (low)
                0.90,  # curious (high)
                0.75,  # determined
                0.10,  # cautious (low)
                0.40,  # reflective
                0.80,  # spontaneous
                0.30,  # methodical
                0.90,  # receptive (very open)
                0.70,  # assertive
                0.10,  # conflicted (low)
                0.20,  # transitioning
                0.60,  # anticipating
                0.30,  # processing
                0.85,  # resonating (high empathy)
            ]
        ))
        
        # SADNESS: High negative valence, low arousal, low dominance
        anchors.append(EmotionalAnchor(
            name="sadness",
            vad=[-0.75, 0.2, 0.2],
            mood_vector=[
                0.30,  # attentive (low)
                0.15,  # energized (very low)
                0.40,  # tense (some)
                0.15,  # assured (low)
                0.25,  # warm (low)
                0.20,  # restless (low)
                0.30,  # protective
                0.15,  # curious (very low)
                0.20,  # determined (low)
                0.40,  # cautious
                0.80,  # reflective (high)
                0.10,  # spontaneous (very low)
                0.40,  # methodical
                0.50,  # receptive
                0.15,  # assertive (very low)
                0.50,  # conflicted
                0.30,  # transitioning
                0.20,  # anticipating (low)
                0.60,  # processing
                0.60,  # resonating
            ]
        ))
        
        # CALM: Neutral valence, low arousal, moderate dominance
        anchors.append(EmotionalAnchor(
            name="calm",
            vad=[0.1, 0.2, 0.5],
            mood_vector=[
                0.50,  # attentive (moderate)
                0.20,  # energized (low)
                0.10,  # tense (very low)
                0.60,  # assured
                0.60,  # warm
                0.05,  # restless (very low)
                0.15,  # protective (low)
                0.50,  # curious (moderate)
                0.40,  # determined
                0.30,  # cautious
                0.70,  # reflective
                0.20,  # spontaneous (low)
                0.65,  # methodical
                0.70,  # receptive
                0.40,  # assertive
                0.10,  # conflicted (low)
                0.10,  # transitioning (stable)
                0.30,  # anticipating
                0.40,  # processing
                0.65,  # resonating
            ]
        ))
        
        # EXCITEMENT: High positive valence, high arousal, moderate dominance
        anchors.append(EmotionalAnchor(
            name="excitement",
            vad=[0.7, 0.9, 0.6],
            mood_vector=[
                0.80,  # attentive
                0.95,  # energized (very high)
                0.20,  # tense (some tension from excitement)
                0.70,  # assured
                0.80,  # warm
                0.70,  # restless (excited energy)
                0.15,  # protective (low)
                0.95,  # curious (very high)
                0.80,  # determined
                0.15,  # cautious (low)
                0.30,  # reflective (low)
                0.90,  # spontaneous (very high)
                0.20,  # methodical (low)
                0.85,  # receptive
                0.75,  # assertive
                0.20,  # conflicted
                0.50,  # transitioning
                0.95,  # anticipating (very high)
                0.40,  # processing
                0.70,  # resonating
            ]
        ))
        
        # FRUSTRATION: Moderate negative valence, moderate arousal, low dominance
        anchors.append(EmotionalAnchor(
            name="frustration",
            vad=[-0.5, 0.6, 0.3],
            mood_vector=[
                0.70,  # attentive
                0.60,  # energized
                0.70,  # tense
                0.30,  # assured (low)
                0.20,  # warm (low)
                0.75,  # restless (high)
                0.50,  # protective
                0.30,  # curious (low)
                0.60,  # determined (still trying)
                0.50,  # cautious
                0.40,  # reflective
                0.50,  # spontaneous
                0.40,  # methodical
                0.30,  # receptive (low)
                0.50,  # assertive
                0.80,  # conflicted (high)
                0.60,  # transitioning
                0.40,  # anticipating
                0.70,  # processing
                0.30,  # resonating
            ]
        ))
        
        # Add more nuanced states...
        # CONTEMPLATIVE, ANXIOUS, CONTENT, BORED, SURPRISED, DISGUSTED, etc.
        
        return anchors
    
    def _generate_grid_samples(self, grid_size: int = 11) -> Tuple[List, List]:
        """
        Generate regular grid samples across the 3D space
        """
        inputs = []
        targets = []
        
        for v in np.linspace(-1, 1, grid_size):
            for a in np.linspace(0, 1, grid_size):
                for d in np.linspace(0, 1, grid_size):
                    inputs.append([v, a, d])
                    
                    # Calculate mood vector based on position
                    mood_vector = self._calculate_mood_from_vad(v, a, d)
                    targets.append(mood_vector)
        
        return inputs, targets
    
    def _generate_random_samples(self, n_samples: int) -> Tuple[List, List]:
        """
        Generate random samples for diverse coverage
        """
        inputs = []
        targets = []
        
        for _ in range(n_samples):
            # Random VAD
            v = np.random.uniform(-1, 1)
            a = np.random.uniform(0, 1)
            d = np.random.uniform(0, 1)
            
            inputs.append([v, a, d])
            targets.append(self._calculate_mood_from_vad(v, a, d))
        
        return inputs, targets
    
    def _calculate_mood_from_vad(self, v: float, a: float, d: float) -> List[float]:
        """
        Calculate expected mood components from VAD coordinates
        This defines the ground truth the network will learn
        """
        mood = np.zeros(self.num_components)
        
        # Core components based on VAD
        mood[0] = a  # attentive scales with arousal
        mood[1] = a  # energized is pure arousal
        mood[2] = max(0, -v) * a  # tense: negative valence + arousal
        mood[3] = max(0, v * 0.5 + d * 0.5)  # assured: positive + dominance
        mood[4] = max(0, v) * (1 - a * 0.3)  # warm: positive, dampened by arousal
        
        # Behavioral components
        mood[5] = a * max(0, -v * 0.5)  # restless: arousal + slight negative
        mood[6] = max(0, -v) * (1 - d)  # protective: negative + low dominance
        mood[7] = max(0, v) * (1 - abs(a - 0.5) * 2)  # curious: positive + moderate arousal
        mood[8] = d * (1 - abs(v) * 0.3)  # determined: dominance, reduced by extreme valence
        mood[9] = (1 - d) * (1 - v) * 0.5  # cautious: low dominance + not positive
        
        # Processing styles
        mood[10] = (1 - a) * (1 - abs(v))  # reflective: low arousal + neutral valence
        mood[11] = a * (1 - abs(d - 0.5) * 2)  # spontaneous: arousal + moderate dominance
        mood[12] = (1 - a) * d  # methodical: low arousal + high dominance
        mood[13] = max(0, v) * (1 - abs(d - 0.5))  # receptive: positive + moderate dominance
        mood[14] = d * (1 + v * 0.3)  # assertive: dominance, boosted by positive
        
        # Complex states
        mood[15] = (1 - abs(v)) * a * 0.7  # conflicted: neutral valence + arousal
        mood[16] = abs(np.sin(v * np.pi)) * a * 0.5  # transitioning: oscillating
        mood[17] = a * max(0, v) * 0.8  # anticipating: arousal + positive
        mood[18] = (1 - abs(v - 0.5)) * (1 - abs(a - 0.5))  # processing: moderate everything
        mood[19] = max(0, v) * (1 - d * 0.3) * 0.7  # resonating: positive + not too dominant
        
        # Normalize and clip
        mood = np.clip(mood, 0, 1)
        
        # Add some natural variance
        noise = np.random.normal(0, 0.02, self.num_components)
        mood = np.clip(mood + noise, 0, 1)
        
        return mood.tolist()
    
    def save_dataset(self, inputs: torch.Tensor, targets: torch.Tensor, path: str):
        """Save dataset to file"""
        torch.save({
            'inputs': inputs,
            'targets': targets,
            'num_samples': len(inputs),
            'input_dim': inputs.shape[1],
            'output_dim': targets.shape[1]
        }, path)
        print(f"Saved dataset with {len(inputs)} samples to {path}")