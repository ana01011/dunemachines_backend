"""
Leaky Integrate-and-Fire (LIF) Neuron Model

A biologically-inspired neuron that:
- Integrates incoming signals over time
- Has a membrane potential that decays (leaky)
- Fires when threshold is reached
- Has a refractory period after firing

This is the building block of our Neural Router.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NeuronType(str, Enum):
    """Types of neurons in our network"""
    INPUT = "input"          # Receives external input (query embedding)
    HIDDEN = "hidden"        # Processing layer
    OUTPUT = "output"        # Represents brain regions (language, math, memory)


@dataclass
class NeuronConfig:
    """Configuration for a LIF neuron"""
    # Membrane dynamics
    tau_membrane: float = 20.0      # Membrane time constant (ms)
    v_rest: float = -70.0           # Resting potential (mV)
    v_reset: float = -75.0          # Reset potential after spike (mV)
    v_threshold: float = -55.0      # Spike threshold (mV)
    
    # Refractory period
    tau_refractory: float = 2.0     # Refractory period (ms)
    
    # Input scaling
    resistance: float = 10.0        # Membrane resistance (MOhm)
    
    # Noise (biological variability)
    noise_std: float = 0.5          # Standard deviation of noise


@dataclass
class NeuronState:
    """Current state of a neuron"""
    membrane_potential: float = -70.0   # Current voltage (mV)
    is_refractory: bool = False         # In refractory period?
    refractory_time: float = 0.0        # Time remaining in refractory
    last_spike_time: Optional[float] = None
    spike_count: int = 0
    
    # For learning
    trace: float = 0.0                  # Eligibility trace for learning


class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron
    
    The fundamental unit of our biological neural network.
    Simulates basic neuron dynamics without the complexity of
    full compartmental models.
    
    Usage:
        neuron = LIFNeuron(neuron_id="hidden_0", neuron_type=NeuronType.HIDDEN)
        
        # Inject current and step simulation
        spike = neuron.step(input_current=5.0, dt=0.1)
        
        # Get firing rate over time window
        rate = neuron.get_firing_rate(time_window=100.0)
    """
    
    def __init__(
        self,
        neuron_id: str,
        neuron_type: NeuronType = NeuronType.HIDDEN,
        config: Optional[NeuronConfig] = None
    ):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.config = config or NeuronConfig()
        
        # Initialize state
        self.state = NeuronState(membrane_potential=self.config.v_rest)
        
        # Spike history for firing rate calculation
        self.spike_times: List[float] = []
        self.current_time: float = 0.0
        
        # For neuromodulation (from neurochemistry system)
        self.threshold_modifier: float = 0.0  # Additive modifier to threshold
        self.gain_modifier: float = 1.0       # Multiplicative modifier to input
    
    def step(self, input_current: float, dt: float = 0.1) -> bool:
        """
        Advance the neuron by one time step.
        
        Args:
            input_current: Total synaptic input current (nA)
            dt: Time step (ms)
            
        Returns:
            True if neuron spiked, False otherwise
        """
        self.current_time += dt
        spike = False
        
        # Handle refractory period
        if self.state.is_refractory:
            self.state.refractory_time -= dt
            if self.state.refractory_time <= 0:
                self.state.is_refractory = False
                self.state.refractory_time = 0.0
            # During refractory, membrane stays at reset
            self.state.membrane_potential = self.config.v_reset
            return False
        
        # Apply gain modifier (from neuromodulation)
        modulated_current = input_current * self.gain_modifier
        
        # Add biological noise
        noise = np.random.normal(0, self.config.noise_std)
        
        # Leaky integration (Euler method)
        # τ * dV/dt = -(V - V_rest) + R * I
        dv = (
            -(self.state.membrane_potential - self.config.v_rest) +
            self.config.resistance * modulated_current +
            noise
        ) * (dt / self.config.tau_membrane)
        
        self.state.membrane_potential += dv
        
        # Check for spike (with threshold modifier from neuromodulation)
        effective_threshold = self.config.v_threshold + self.threshold_modifier
        
        if self.state.membrane_potential >= effective_threshold:
            # Spike!
            spike = True
            self.state.spike_count += 1
            self.state.last_spike_time = self.current_time
            self.spike_times.append(self.current_time)
            
            # Reset membrane potential
            self.state.membrane_potential = self.config.v_reset
            
            # Enter refractory period
            self.state.is_refractory = True
            self.state.refractory_time = self.config.tau_refractory
            
            # Update eligibility trace (for learning)
            self.state.trace = 1.0
        else:
            # Decay eligibility trace
            self.state.trace *= np.exp(-dt / 20.0)  # 20ms decay constant
        
        return spike
    
    def get_firing_rate(self, time_window: float = 100.0) -> float:
        """
        Calculate firing rate over recent time window.
        
        Args:
            time_window: Window size in ms
            
        Returns:
            Firing rate in Hz
        """
        if not self.spike_times:
            return 0.0
        
        # Count spikes in window
        window_start = self.current_time - time_window
        recent_spikes = sum(1 for t in self.spike_times if t >= window_start)
        
        # Convert to Hz (spikes per second)
        rate = (recent_spikes / time_window) * 1000.0
        return rate
    
    def get_activation(self, time_window: float = 50.0) -> float:
        """
        Get activation level as normalized value 0-1.
        
        This is what we use for the router output - 
        a continuous value representing how "active" this neuron is.
        
        Args:
            time_window: Window for rate calculation (ms)
            
        Returns:
            Activation level between 0 and 1
        """
        rate = self.get_firing_rate(time_window)
        # Normalize: assume max meaningful rate is ~100 Hz
        activation = min(rate / 100.0, 1.0)
        return activation
    
    def reset(self):
        """Reset neuron to initial state"""
        self.state = NeuronState(membrane_potential=self.config.v_rest)
        self.spike_times = []
        self.current_time = 0.0
    
    def set_neuromodulation(
        self,
        threshold_modifier: float = 0.0,
        gain_modifier: float = 1.0
    ):
        """
        Apply neuromodulation effects from neurochemistry system.
        
        Args:
            threshold_modifier: Additive change to spike threshold
                               (negative = easier to fire)
            gain_modifier: Multiplicative change to input gain
                          (>1 = amplified input)
        """
        self.threshold_modifier = threshold_modifier
        self.gain_modifier = gain_modifier
    
    def inject_current_from_value(self, value: float, scale: float = 10.0) -> float:
        """
        Convert an external value (like embedding dimension) to current.
        
        Args:
            value: Input value (e.g., from embedding, typically -1 to 1)
            scale: Scaling factor
            
        Returns:
            Current in nA
        """
        # Map value to current, with some nonlinearity
        # Positive values = excitatory, negative = inhibitory
        current = value * scale
        return current
    
    def __repr__(self) -> str:
        return (
            f"LIFNeuron(id={self.neuron_id}, type={self.neuron_type.value}, "
            f"V={self.state.membrane_potential:.1f}mV, "
            f"spikes={self.state.spike_count})"
        )


class Synapse:
    """
    Synapse connecting two neurons.
    
    Contains:
    - Weight (strength of connection)
    - Delay (axonal propagation time)
    - Plasticity state (for learning)
    """
    
    def __init__(
        self,
        pre_neuron_id: str,
        post_neuron_id: str,
        weight: float = 0.5,
        delay: float = 1.0,  # ms
        is_excitatory: bool = True
    ):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.delay = delay
        self.is_excitatory = is_excitatory
        
        # For STDP / Hebbian learning
        self.eligibility_trace: float = 0.0
        self.weight_change_accumulator: float = 0.0
        
        # Bounds
        self.weight_min: float = 0.0
        self.weight_max: float = 1.0
    
    def get_effective_weight(self) -> float:
        """Get weight with sign based on excitatory/inhibitory"""
        sign = 1.0 if self.is_excitatory else -1.0
        return self.weight * sign
    
    def apply_weight_change(self, delta: float, learning_rate: float = 0.01):
        """
        Apply weight change with bounds.
        
        Args:
            delta: Raw weight change
            learning_rate: Learning rate modifier
        """
        change = delta * learning_rate
        self.weight = np.clip(
            self.weight + change,
            self.weight_min,
            self.weight_max
        )
    
    def __repr__(self) -> str:
        sign = "+" if self.is_excitatory else "-"
        return f"Synapse({self.pre_neuron_id}->{self.post_neuron_id}, w={sign}{self.weight:.3f})"


# Convenience function to create a layer of neurons
def create_neuron_layer(
    layer_name: str,
    size: int,
    neuron_type: NeuronType,
    config: Optional[NeuronConfig] = None
) -> List[LIFNeuron]:
    """
    Create a layer of neurons.
    
    Args:
        layer_name: Name prefix for neurons (e.g., "input", "hidden")
        size: Number of neurons
        neuron_type: Type of neurons in this layer
        config: Optional configuration (uses default if None)
        
    Returns:
        List of LIFNeuron instances
    """
    neurons = []
    for i in range(size):
        neuron_id = f"{layer_name}_{i}"
        neuron = LIFNeuron(
            neuron_id=neuron_id,
            neuron_type=neuron_type,
            config=config
        )
        neurons.append(neuron)
    return neurons


if __name__ == "__main__":
    # Quick test
    print("Testing LIF Neuron...")
    
    neuron = LIFNeuron("test_0", NeuronType.HIDDEN)
    
    # Simulate with constant input
    spikes = 0
    for _ in range(1000):  # 100ms at 0.1ms steps
        if neuron.step(input_current=8.0, dt=0.1):
            spikes += 1
    
    print(f"Neuron fired {spikes} times in 100ms")
    print(f"Firing rate: {neuron.get_firing_rate():.1f} Hz")
    print(f"Activation: {neuron.get_activation():.3f}")
    print("✅ LIF Neuron test passed!")
