"""
Test the enhanced neurochemistry system with opponent processes
"""

import sys
import os

# Add path for imports
sys.path.insert(0, '/root/openhermes_backend')

# Import directly from enhanced modules (bypass the __init__ files)
import importlib.util

# Load state_enhanced module directly
spec_state = importlib.util.spec_from_file_location(
    "state_enhanced", 
    "/root/openhermes_backend/app/neurochemistry/core/state_enhanced.py"
)
state_enhanced = importlib.util.module_from_spec(spec_state)
spec_state.loader.exec_module(state_enhanced)

# Load dynamics_enhanced module directly
spec_dynamics = importlib.util.spec_from_file_location(
    "dynamics_enhanced",
    "/root/openhermes_backend/app/neurochemistry/core/dynamics_enhanced.py"
)
dynamics_enhanced = importlib.util.module_from_spec(spec_dynamics)
spec_dynamics.loader.exec_module(dynamics_enhanced)

# Now use the enhanced classes
NeurochemicalState = state_enhanced.NeurochemicalState
NeurochemicalDynamics = dynamics_enhanced.NeurochemicalDynamics

def test_enhanced_system():
    print("="*60)
    print("TESTING ENHANCED NEUROCHEMISTRY WITH OPPONENT PROCESSES")
    print("="*60)
    
    # Initialize system
    state = NeurochemicalState()
    dynamics = NeurochemicalDynamics(state)
    
    # Test scenarios
    scenarios = [
        ("Baseline", {}),
        ("Joy (Dopamine spike)", {'reward': 0.8, 'novelty': 0.5}),
        ("Stress (Cortisol spike)", {'threat': 0.7, 'uncertainty': 0.5}),
        ("Relaxation (Cortisol reduction)", {'relaxation': 0.8}),
        ("Exercise (Endorphins)", {'exercise': 0.8, 'pleasure': 0.3}),
        ("Social (Oxytocin)", {'social': 0.8, 'trust': 0.6}),
    ]
    
    for name, inputs in scenarios:
        print(f"\n{name}:")
        print("-" * 40)
        
        # Run for 5 seconds
        for _ in range(50):
            dynamics.step(0.1, inputs)
        
        # Show results
        print(f"Hormones:")
        for h, v in state.hormones.items():
            baseline = state.baselines['slow'][h]
            ratio = state.opponent_ratios.get(h, 1.0)
            print(f"  {h:15}: {v:6.2f} (baseline: {baseline:6.2f}, ratio: {ratio:.2f})")
        
        print(f"\nResources:")
        for r, v in state.resources.items():
            print(f"  {r:15}: {v:.3f}")
        
        print(f"\nCost: {state.calculate_total_cost()['total']:.2f}")
        print(f"Mood: {state.get_mood()}")
        print(f"Allostatic Load: {state.allostatic_load:.4f}")
        
        # Show spike history
        spikes_detected = []
        for h in state.hormones:
            avg_spike = state.get_average_spike(h)
            if avg_spike > 0:
                spikes_detected.append(f"{h}={avg_spike:.1f}")
        if spikes_detected:
            print(f"Spike History: {', '.join(spikes_detected)}")

if __name__ == "__main__":
    test_enhanced_system()
