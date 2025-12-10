"""
Test the stronger fixes
"""

import sys
sys.path.append('/root/openhermes_backend')
from app.neurochemistry.interface import NeurochemicalSystem
import numpy as np

print("="*60)
print("TESTING STRONGER FIXES")
print("="*60)

system = NeurochemicalSystem()

# Test 1: Stress should cause BIG adrenaline spike
print("\n1. ADRENALINE STRESS TEST:")
system.reset()
baseline_state = system.process_message("")
print(f"Baseline: A={baseline_state['hormones']['adrenaline']:.1f}")

# Single stress message
stress_response = system.process_message("EMERGENCY! URGENT! HELP NOW!!!")
print(f"After 1 stress: A={stress_response['hormones']['adrenaline']:.1f}")

# Multiple stress
for _ in range(3):
    stress_response = system.process_message("EMERGENCY! URGENT! HELP NOW!!!")
print(f"After 4 stress: A={stress_response['hormones']['adrenaline']:.1f}")
print(f"✅ GOOD if > 50")

# Test 2: ATP should deplete with activity
print("\n2. ATP DEPLETION TEST:")
system.reset()
initial_atp = system.state.e_atp
print(f"Initial ATP: {initial_atp:.2f}")

# Light activity
for _ in range(5):
    system.process_message("Working hard!")
mid_atp = system.state.e_atp
print(f"After 5 messages: {mid_atp:.2f}")

# Heavy activity
for _ in range(10):
    system.process_message("INTENSE EXERCISE! RUNNING FAST!")
final_atp = system.state.e_atp
print(f"After exercise: {final_atp:.2f}")
print(f"✅ GOOD if < 95")

# Test 3: Allostatic load with chronic stress
print("\n3. ALLOSTATIC LOAD TEST:")
system.reset()
initial_load = system.state.allostatic_load
print(f"Initial load: {initial_load:.4f}")

# Apply chronic stress
for _ in range(10):
    system.process_message("Chronic stress and worry...")
    system.simulate_time_passage(1.0)  # Let it accumulate

final_load = system.state.allostatic_load
print(f"After chronic stress: {final_load:.4f}")
print(f"✅ GOOD if > 0.01")

# Test 4: Check all hormones respond appropriately
print("\n4. COMPREHENSIVE HORMONE TEST:")
scenarios = [
    ("Joy", "Amazing! Best day ever! So happy!"),
    ("Stress", "Emergency! Crisis! Help!"),
    ("Exercise", "Running marathon! Push harder!"),
    ("Sad", "So sad and depressed..."),
    ("Relax", "Peaceful meditation calm...")
]

for scenario, message in scenarios:
    system.reset()
    baseline = system.process_message("")
    
    # Process scenario
    for _ in range(3):
        response = system.process_message(message)
    
    print(f"\n{scenario}:")
    print(f"  D: {baseline['hormones']['dopamine']:.0f} → {response['hormones']['dopamine']:.0f}")
    print(f"  S: {baseline['hormones']['serotonin']:.0f} → {response['hormones']['serotonin']:.0f}")
    print(f"  C: {baseline['hormones']['cortisol']:.0f} → {response['hormones']['cortisol']:.0f}")
    print(f"  A: {baseline['hormones']['adrenaline']:.0f} → {response['hormones']['adrenaline']:.0f}")
    print(f"  E: {baseline['hormones']['endorphins']:.0f} → {response['hormones']['endorphins']:.0f}")
    print(f"  Mood: {baseline['mood']} → {response['mood']}")
    print(f"  ATP: {system.state.e_atp:.1f}%")

print("\n" + "="*60)
print("EXPECTED GOOD RESULTS:")
print("- Stress: Adrenaline > 50, Cortisol > 40")
print("- Exercise: Endorphins > 60, ATP < 95")
print("- Sad: Serotonin < 40, Dopamine < 45")
print("- Relax: Adrenaline < 15, Cortisol < 30")
print("="*60)

