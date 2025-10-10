"""
Scalable Hormone-to-Mood Neural Network
Easily configurable architecture for discovering emotional states from hormones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import json


class ScalableHormoneNetwork(nn.Module):
    """
    Highly configurable neural network for hormone → mood mapping
    Easy to scale up/down by changing parameters
    """
    
    def __init__(
        self,
        input_dim: int = 5,           # Number of hormones (D, C, A, S, O)
        output_dim: int = 30,          # Number of mood components (increased for richness)
        hidden_layers: List[int] = None,  # Custom layer sizes
        activation: str = 'relu',      # Activation function
        dropout_rate: float = 0.1,     # Dropout for regularization
        use_batch_norm: bool = True,   # Batch normalization
        output_activation: str = 'sigmoid'  # Output activation
    ):
        super().__init__()
        
        # Default architecture if not specified
        if hidden_layers is None:
            # Default: progressively expanding then compressing
            # Easy to modify: just pass different list!
            hidden_layers = [32, 64, 128, 128, 64, 32]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.output_activation_type = output_activation
        
        # Build the network dynamically based on config
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Choose activation function
        self.activation = self._get_activation(activation)
        
        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            # Add linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add batch norm if requested
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Add dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
            
            print(f"Layer {i+1}: {prev_dim} → {hidden_dim}")
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        print(f"Output layer: {prev_dim} → {output_dim}")
        
        # Output activation
        self.output_activation = self._get_activation(output_activation)
        
        # Initialize weights with better strategy
        self._initialize_weights()
        
    def _get_activation(self, name: str):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()  # No activation
        }
        return activations.get(name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU family
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                # Xavier for tanh/sigmoid
                else:
                    nn.init.xavier_normal_(layer.weight)
                
                # Small bias initialization
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x):
        """
        Forward pass through the network
        Input: hormone levels [batch_size, 5]
        Output: mood components [batch_size, output_dim]
        """
        # Handle both single samples and batches
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Batch norm (if enabled and batch size > 1)
            if self.batch_norms and x.shape[0] > 1:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout (only during training)
            if self.training:
                x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation
        if self.output_activation_type != 'none':
            x = self.output_activation(x)
        
        return x
    
    def get_config(self) -> Dict:
        """Get network configuration for saving/loading"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'output_activation': self.output_activation_type
        }


class ExtendedMoodComponents:
    """
    Extended mood components for richer emotional states (30 components)
    """
    
    COMPONENTS = [
        # Core emotional energy (0-4)
        'attentive',       # 0: Focus and concentration
        'energized',       # 1: Activation level
        'tense',          # 2: Stress and pressure
        'assured',        # 3: Confidence
        'warm',           # 4: Social warmth
        
        # Behavioral tendencies (5-9)
        'restless',       # 5: Agitation
        'protective',     # 6: Defensive
        'curious',        # 7: Exploratory
        'determined',     # 8: Goal-focused
        'cautious',       # 9: Risk-aware
        
        # Processing styles (10-14)
        'reflective',     # 10: Introspective
        'spontaneous',    # 11: Impulsive
        'methodical',     # 12: Systematic
        'receptive',      # 13: Open
        'assertive',      # 14: Pushing forward
        
        # Complex states (15-19)
        'conflicted',     # 15: Mixed signals
        'transitioning',  # 16: Changing
        'anticipating',   # 17: Future-focused
        'processing',     # 18: Analyzing
        'resonating',     # 19: Empathetic
        
        # Novel AI states (20-29) - Unique to AI!
        'parallel_processing',  # 20: Multiple tracks
        'recursive_thinking',   # 21: Meta-cognition
        'pattern_matching',     # 22: Recognition mode
        'synthesizing',        # 23: Combining ideas
        'optimizing',          # 24: Efficiency seeking
        'exploratory',         # 25: Trying new paths
        'consolidating',       # 26: Memory forming
        'predictive',          # 27: Forecasting
        'calibrating',         # 28: Self-adjusting
        'emergent'            # 29: Novel states arising
    ]
    
    @classmethod
    def get_component_name(cls, index: int) -> str:
        """Get component name by index"""
        if index < len(cls.COMPONENTS):
            return cls.COMPONENTS[index]
        return f"component_{index}"
    
    @classmethod
    def describe_mood_state(cls, mood_vector: np.ndarray, threshold: float = 0.3) -> Dict:
        """
        Describe mood state from network output
        """
        active_moods = []
        
        for i, intensity in enumerate(mood_vector):
            if intensity > threshold:
                mood_name = cls.get_component_name(i)
                
                # Intensity descriptions
                if intensity > 0.9:
                    prefix = "extremely"
                elif intensity > 0.75:
                    prefix = "very"
                elif intensity > 0.6:
                    prefix = "notably"
                elif intensity > 0.45:
                    prefix = "moderately"
                else:
                    prefix = "slightly"
                
                active_moods.append({
                    'component': mood_name,
                    'intensity': float(intensity),
                    'description': f"{prefix} {mood_name}"
                })
        
        # Sort by intensity
        active_moods.sort(key=lambda x: x['intensity'], reverse=True)
        
        # Create natural description
        if len(active_moods) == 0:
            description = "neutral baseline"
        elif len(active_moods) == 1:
            description = active_moods[0]['description']
        elif len(active_moods) == 2:
            description = f"{active_moods[0]['description']} + {active_moods[1]['description']}"
        else:
            # Top 3 as main, rest as undertones
            main = " + ".join([m['description'] for m in active_moods[:3]])
            if len(active_moods) > 3:
                undertones = [m['component'] for m in active_moods[3:5]]  # Next 2
                description = f"{main} (with {', '.join(undertones)})"
            else:
                description = main
        
        return {
            'description': description,
            'active_moods': active_moods,
            'primary': active_moods[0] if active_moods else None,
            'num_active': len(active_moods)
        }


class NetworkConfigPresets:
    """
    Preset configurations for different use cases
    Easy to switch between architectures!
    """
    
    @staticmethod
    def tiny():
        """Tiny network for fast inference (~1K parameters)"""
        return {
            'hidden_layers': [16, 16],
            'dropout_rate': 0.05
        }
    
    @staticmethod
    def small():
        """Small network for embedded systems (~10K parameters)"""
        return {
            'hidden_layers': [32, 64, 32],
            'dropout_rate': 0.1
        }
    
    @staticmethod
    def medium():
        """Medium network for balanced performance (~50K parameters)"""
        return {
            'hidden_layers': [64, 128, 128, 64],
            'dropout_rate': 0.15
        }
    
    @staticmethod
    def large():
        """Large network for maximum accuracy (~200K parameters)"""
        return {
            'hidden_layers': [128, 256, 256, 256, 128, 64],
            'dropout_rate': 0.2
        }
    
    @staticmethod
    def deep():
        """Deep network for complex patterns (~500K parameters)"""
        return {
            'hidden_layers': [64, 128, 256, 512, 512, 256, 128, 64],
            'dropout_rate': 0.25
        }
    
    @staticmethod
    def experimental():
        """Experimental architecture with multiple pathways"""
        return {
            'hidden_layers': [50, 100, 200, 300, 200, 100, 50],
            'dropout_rate': 0.3,
            'activation': 'elu',
            'output_activation': 'none'  # Raw values, no sigmoid!
        }


def create_network(preset: str = 'medium', **kwargs) -> ScalableHormoneNetwork:
    """
    Create network with preset or custom configuration
    
    Examples:
        # Use preset
        net = create_network('large')
        
        # Custom configuration
        net = create_network(hidden_layers=[100, 200, 100])
        
        # Modify preset
        net = create_network('medium', output_dim=50)
    """
    # Get preset config
    presets = {
        'tiny': NetworkConfigPresets.tiny(),
        'small': NetworkConfigPresets.small(),
        'medium': NetworkConfigPresets.medium(),
        'large': NetworkConfigPresets.large(),
        'deep': NetworkConfigPresets.deep(),
        'experimental': NetworkConfigPresets.experimental()
    }
    
    config = presets.get(preset, NetworkConfigPresets.medium())
    
    # Override with custom parameters
    config.update(kwargs)
    
    # Create network
    network = ScalableHormoneNetwork(**config)
    
    # Print summary
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork created: {preset}")
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: {config.get('hidden_layers', [])}")
    
    return network
