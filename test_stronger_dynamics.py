"""
Test with stronger, more visible dynamics
"""

import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.interface import NeurochemicalSystem
import numpy as np

def test_stronger():
    system = NeurochemicalSystem(user_id="test_user")
    
    print("="*60)
    print("TESTING STRONGER DYNAMICS")
    print("="*60)
    
    # Increase timestep for bigger changes
    system.dt = 1.0  # 1 second timesteps instead of 0.1
    
    scenarios = [
        ("🎉 EXTREME JOY", "THIS IS THE BEST DAY OF MY LIFE!!! I WON THE LOTTERY!!! AMAZING!!!"),
        ("😰 PANIC ATTACK", "EMERGENCY!!! HELP!!! DANGER!!! I'M TERRIFIED!!!"),
        ("💔 DEEP SADNESS", "I'm so depressed. Everything is horrible. I hate my life."),
        ("🤗 LOVE & CONNECTION", "I love you so much! Let's be together forever! You're amazing!"),
        ("🏃 INTENSE EXERCISE", "Running marathon! Push harder! Feel the burn! Endorphins!!!"),
        ("😴 DEEP RELAXATION", "So peaceful... meditation... calm... sleepy... rest..."),
    ]
    
    for title, message in scenarios:
        print(f"\n{title}")
        print("-"*40)
        
        # Reset to baseline
        system.reset()
        
        # Get initial state
        initial = system.process_message("")
        
        # Process intense message multiple times for stronger effect
        for i in range(5):  # Process 5 times to amplify effect
            response = system.process_message(message)
        
        # Show changes
        print(f"Dopamine: {initial['hormones']['dopamine']:.1f} → {response['hormones']['dopamine']:.1f} "
              f"(Δ={response['hormones']['dopamine']-initial['hormones']['dopamine']:+.1f})")
        print(f"Serotonin: {initial['hormones']['serotonin']:.1f} → {response['hormones']['serotonin']:.1f} "
              f"(Δ={response['hormones']['serotonin']-initial['hormones']['serotonin']:+.1f})")
        print(f"Cortisol: {initial['hormones']['cortisol']:.1f} → {response['hormones']['cortisol']:.1f} "
              f"(Δ={response['hormones']['cortisol']-initial['hormones']['cortisol']:+.1f})")
        print(f"Adrenaline: {initial['hormones']['adrenaline']:.1f} → {response['hormones']['adrenaline']:.1f} "
              f"(Δ={response['hormones']['adrenaline']-initial['hormones']['adrenaline']:+.1f})")
        print(f"Oxytocin: {initial['hormones']['oxytocin']:.1f} → {response['hormones']['oxytocin']:.1f} "
              f"(Δ={response['hormones']['oxytocin']-initial['hormones']['oxytocin']:+.1f})")
        print(f"Endorphins: {initial['hormones']['endorphins']:.1f} → {response['hormones']['endorphins']:.1f} "
              f"(Δ={response['hormones']['endorphins']-initial['hormones']['endorphins']:+.1f})")
        
        print(f"\nMood: {initial['mood']} → {response['mood']}")
        print(f"Efficiency: {response['efficiency']:.2f}")
        print(f"Allostatic Load: {response['allostatic_load']:.2%}")
        
        # Show seeking behavior
        print(f"Seeking: D={response['seeking']['dopamine_seeking']:.2f}, "
              f"S={response['seeking']['serotonin_seeking']:.2f}, "
              f"O={response['seeking']['oxytocin_seeking']:.2f}")

if __name__ == "__main__":
    test_stronger()
