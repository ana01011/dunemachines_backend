"""
Test that the fixes work correctly
"""

import sys
sys.path.append('/root/openhermes_backend')
from app.neurochemistry.interface import NeurochemicalSystem

system = NeurochemicalSystem()

print("Testing Fixed System:")
print("="*50)

# Test 1: Stress response
print("\n1. STRESS RESPONSE TEST:")
system.reset()
initial = system.process_message("")
for _ in range(5):
    response = system.process_message("URGENT! EMERGENCY! HELP!")
print(f"Cortisol: {initial['hormones']['cortisol']:.1f} → {response['hormones']['cortisol']:.1f}")
print(f"Adrenaline: {initial['hormones']['adrenaline']:.1f} → {response['hormones']['adrenaline']:.1f}")
print(f"Mood: {initial['mood']} → {response['mood']}")

# Test 2: Resource depletion
print("\n2. RESOURCE DEPLETION TEST:")
system.reset()
initial_resources = system.state.e_atp
for _ in range(20):
    system.process_message("Intense activity!")
final_resources = system.state.e_atp
print(f"ATP: {initial_resources:.2f} → {final_resources:.2f}")

# Test 3: Allostatic load
print("\n3. ALLOSTATIC LOAD TEST:")
system.reset()
for _ in range(30):
    system.process_message("Chronic stress...")
print(f"Allostatic Load: {system.state.allostatic_load:.4f}")

# Test 4: Cost function
print("\n4. COST FUNCTION TEST:")
system.reset()
costs = []
for i in range(20):
    response = system.process_message("Normal conversation")
    costs.append(response['cost']['total'])
print(f"Cost trend: {costs[0]:.1f} → {costs[10]:.1f} → {costs[-1]:.1f}")

print("\n✅ If adrenaline > 50 and ATP < 1.0, fixes are working!")

