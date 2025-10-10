"""
Multi-Mood Emergence Neural Network
Maps 3D emotional space to multiple simultaneous mood components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import json


class MultiMoodEmergenceNetwork(nn.Module):
    """
    Deep neural network for nuanced emotion emergence
    Input: 3D vector [V, A, D]
    Output: 20 mood component intensities
    ~15K parameters for rich representation while staying fast
    """
    
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 128, 64, 32], output_dim=20):
        super().__init__()
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        self.dropouts.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(0.15))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        """
        Forward pass through the network
        """
        # Handle both single samples and batches
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Process through layers
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            
            # Skip batch norm if single sample
            if x.shape[0] > 1:
                x = bn(x)
            
            # Activation
            if i < 2:
                x = self.relu(x)
            else:
                x = self.leaky_relu(x)
            
            # Dropout only during training
            if self.training:
                x = dropout(x)
        
        # Output layer with sigmoid for 0-1 range
        output = torch.sigmoid(self.output_layer(x))
        
        return output


class MoodComponents:
    """
    Defines the 20 mood components the network learns
    """
    
    COMPONENTS = [
        # Core attention/energy (0-4)
        'attentive',      # Focus and concentration
        'energized',      # Activation level
        'tense',          # Stress and pressure
        'assured',        # Confidence
        'warm',           # Social connection
        
        # Behavioral tendencies (5-9)
        'restless',       # Agitation
        'protective',     # Defensive stance
        'curious',        # Exploratory drive
        'determined',     # Goal-focused
        'cautious',       # Risk-aware
        
        # Processing styles (10-14)
        'reflective',     # Introspective
        'spontaneous',    # Impulsive
        'methodical',     # Systematic
        'receptive',      # Open to input
        'assertive',      # Pushing forward
        
        # Complex states (15-19)
        'conflicted',     # Mixed signals
        'transitioning',  # State change
        'anticipating',   # Future-focused
        'processing',     # Analyzing
        'resonating',     # Empathetic
    ]
    
    @classmethod
    def get_component_name(cls, index: int) -> str:
        """Get component name by index"""
        return cls.COMPONENTS[index] if index < len(cls.COMPONENTS) else f"component_{index}"
    
    @classmethod
    def get_component_index(cls, name: str) -> int:
        """Get component index by name"""
        try:
            return cls.COMPONENTS.index(name)
        except ValueError:
            return -1


class MoodInterpreter:
    """
    Interprets neural network output into readable mood states
    """
    
    def __init__(self, model_path: str = None, threshold: float = 0.25):
        self.threshold = threshold
        self.network = MultiMoodEmergenceNetwork()
        
        if model_path:
            self.load_model(model_path)
        
        self.network.eval()
    
    def load_model(self, path: str):
        """Load trained model weights"""
        self.network.load_state_dict(torch.load(path, map_location='cpu'))
    
    def get_mood_state(self, v: float, a: float, d: float) -> Dict:
        """
        Get mood state from VAD coordinates
        Returns top active moods with intensities
        """
        with torch.no_grad():
            # Prepare input
            vad = torch.tensor([v, a, d], dtype=torch.float32)
            
            # Get mood components
            mood_vector = self.network(vad)
            
            # If batch, take first item
            if len(mood_vector.shape) > 1:
                mood_vector = mood_vector[0]
        
        # Extract significant moods
        moods = []
        for i, intensity in enumerate(mood_vector):
            if intensity > self.threshold:
                moods.append({
                    'component': MoodComponents.get_component_name(i),
                    'index': i,
                    'intensity': float(intensity)
                })
        
        # Sort by intensity
        moods.sort(key=lambda x: x['intensity'], reverse=True)
        
        # Ensure at least one mood
        if not moods:
            moods = [{'component': 'neutral', 'index': -1, 'intensity': 0.5}]
        
        return self.format_mood_state(moods)
    
    def format_mood_state(self, moods: List[Dict]) -> Dict:
        """
        Format mood state into readable description
        """
        descriptions = []
        
        for mood in moods[:5]:  # Top 5 maximum
            intensity = mood['intensity']
            component = mood['component']
            
            # Natural intensity descriptors
            if intensity > 0.85:
                prefix = "intensely"
            elif intensity > 0.70:
                prefix = "very"
            elif intensity > 0.55:
                prefix = "notably"
            elif intensity > 0.40:
                prefix = "moderately"
            else:
                prefix = "slightly"
            
            descriptions.append(f"{prefix} {component}")
        
        # Create composite description
        if len(descriptions) == 1:
            mood_string = descriptions[0]
        elif len(descriptions) == 2:
            mood_string = f"{descriptions[0]} + {descriptions[1]}"
        else:
            # Top 3 with "and undertones of" for the rest
            main_moods = " + ".join(descriptions[:3])
            if len(descriptions) > 3:
                undertones = ", ".join(descriptions[3:])
                mood_string = f"{main_moods} (with undertones of {undertones})"
            else:
                mood_string = main_moods
        
        return {
            'description': mood_string,
            'components': moods,
            'primary': moods[0]['component'],
            'primary_intensity': moods[0]['intensity'],
            'num_active': len(moods)
        }
    
    def get_behavioral_hints(self, mood_state: Dict) -> List[str]:
        """
        Convert mood state to behavioral hints for the LLM
        """
        hints = []
        
        for component in mood_state['components'][:3]:  # Top 3
            mood = component['component']
            intensity = component['intensity']
            
            if mood == 'attentive' and intensity > 0.5:
                hints.append("maintaining sharp focus")
            elif mood == 'tense' and intensity > 0.5:
                hints.append("experiencing elevated pressure")
            elif mood == 'warm' and intensity > 0.5:
                hints.append("expressing social connection")
            elif mood == 'energized' and intensity > 0.6:
                hints.append("operating with high activation")
            elif mood == 'cautious' and intensity > 0.5:
                hints.append("proceeding with care")
            elif mood == 'determined' and intensity > 0.6:
                hints.append("strongly goal-oriented")
            elif mood == 'reflective' and intensity > 0.5:
                hints.append("considering deeply")
            elif mood == 'protective' and intensity > 0.6:
                hints.append("maintaining boundaries")
        
        return hints