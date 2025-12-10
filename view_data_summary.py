"""
View and summarize the animation data
"""

import json
import pandas as pd
import numpy as np

# Load the data
with open('/root/openhermes_backend/neurochemistry_animation_data.json', 'r') as f:
    data = json.load(f)

print("="*60)
print("NEUROCHEMISTRY DATA SUMMARY")
print("="*60)

# Show scenario breakdown
scenarios = data['scenario']
unique_scenarios = list(set(scenarios))
print("\nScenario Breakdown:")
for scenario in unique_scenarios:
    count = scenarios.count(scenario)
    print(f"  {scenario:15s}: {count:3d} data points")

# Show hormone statistics
print("\nHormone Statistics:")
print(f"{'Hormone':<15s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'StdDev':>8s}")
print("-"*55)

for hormone in ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 'oxytocin', 'norepinephrine', 'endorphins']:
    values = data['hormones'][hormone]
    print(f"{hormone:<15s} {min(values):8.1f} {max(values):8.1f} {np.mean(values):8.1f} {np.std(values):8.1f}")

# Show mood distribution
print("\nMood Distribution:")
moods = data['mood']
unique_moods = list(set(moods))
for mood in unique_moods:
    count = moods.count(mood)
    percentage = (count / len(moods)) * 100
    print(f"  {mood:15s}: {count:3d} ({percentage:5.1f}%)")

# Show cost analysis
print("\nCost Function Analysis:")
costs = data['cost']
print(f"  Initial Cost: {costs[0]:.1f}")
print(f"  Final Cost:   {costs[-1]:.1f}")
print(f"  Min Cost:     {min(costs):.1f}")
print(f"  Max Cost:     {max(costs):.1f}")
print(f"  Mean Cost:    {np.mean(costs):.1f}")

# Show resource usage
print("\nResource Levels:")
for resource in ['tyrosine', 'tryptophan', 'atp']:
    values = data['resources'][resource]
    print(f"  {resource:12s}: Min={min(values):.2f}, Max={max(values):.2f}, Final={values[-1]:.2f}")

# Show allostatic load
print("\nAllostatic Load:")
load = data['allostatic_load']
print(f"  Max Load:     {max(load):.4f}")
print(f"  Final Load:   {load[-1]:.4f}")

# Export key moments to CSV for easy viewing
print("\n" + "="*60)
print("KEY MOMENTS EXTRACT")
print("="*60)

# Create a simplified dataframe with key moments
df = pd.DataFrame({
    'time': data['time'],
    'scenario': data['scenario'],
    'mood': data['mood'],
    'dopamine': data['hormones']['dopamine'],
    'serotonin': data['hormones']['serotonin'],
    'cortisol': data['hormones']['cortisol'],
    'adrenaline': data['hormones']['adrenaline'],
    'oxytocin': data['hormones']['oxytocin'],
    'endorphins': data['hormones']['endorphins'],
    'cost': data['cost']
})

# Show key transitions
key_indices = [0, 10, 30, 50, 70, 90, 110, 130, 149]
print("\nKey Moments:")
print(df.iloc[key_indices].to_string())

# Save simplified version
df.to_csv('/root/openhermes_backend/neurochemistry_simplified.csv', index=False)
print("\n📁 Simplified data saved to neurochemistry_simplified.csv")
