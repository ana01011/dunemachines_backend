"""
Debug and fix the seeking calculation and remaining dynamics issues
"""

import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.interface import NeurochemicalSystem
import numpy as np

# Test seeking calculation
system = NeurochemicalSystem()

# Check the minimization calculation
print("Testing Seeking Calculation:")
print("-" * 40)

# Process a message to get some state change
response = system.process_message("I am sad")

# Check the raw values
PE = system.state.get_prediction_error()
receptors = system.state.receptors
gradient = system.minimization.calculate_cost_gradient()

print(f"Prediction Errors: {PE}")
print(f"Receptors: {receptors}")
print(f"Cost Gradient: {gradient}")
print(f"Seeking output: {response['seeking']}")

# Check the seeking calculation method
import inspect
print("\nSeeking calculation code:")
print(inspect.getsource(system.minimization.calculate_seeking_intensity))
