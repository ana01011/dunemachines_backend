"""
Complete system test showing realistic scenarios
"""

import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.interface import NeurochemicalSystem
import numpy as np
import matplotlib.pyplot as plt

def test_daily_cycle():
    """Simulate a full day with various activities"""
    
    system = NeurochemicalSystem()
    system.dt = 0.5  # 0.5 second timesteps
    
    # Timeline: A day in the life
    scenarios = [
        # Morning
        (0, "Wake up", "Good morning! Time to start the day!", "neutral"),
        (10, "Coffee", "Having my morning coffee, feeling more alert", "stimulant"),
        (20, "Work stress", "Urgent deadline! So much pressure at work!", "stress"),
        (40, "Small success", "Yes! Solved that problem!", "achievement"),
        
        # Midday
        (60, "Lunch break", "Taking a break, eating healthy food", "rest"),
        (80, "Social", "Great conversation with colleagues, feeling connected", "social"),
        (100, "Exercise", "Going for a run! Feel the burn! Endorphins kicking in!", "exercise"),
        (120, "Post-workout", "Feeling amazing after that workout!", "euphoria"),
        
        # Afternoon
        (140, "Focused work", "Deep focus, getting things done", "focus"),
        (160, "Frustration", "This isn't working, getting frustrated", "frustration"),
        (180, "Help received", "Colleague helped me out, feeling grateful", "gratitude"),
        
        # Evening
        (200, "Relaxation", "Finally home, time to relax and unwind", "relaxation"),
        (220, "Family time", "Spending quality time with loved ones", "love"),
        (240, "Entertainment", "Watching favorite show, laughing a lot", "joy"),
        (260, "Wind down", "Getting sleepy, preparing for bed", "tiredness"),
        (280, "Sleep", "Going to sleep, peaceful and calm", "sleep"),
    ]
    
    # Storage for results
    timeline = []
    hormones = {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                                 'oxytocin', 'norepinephrine', 'endorphins']}
    moods = []
    behaviors = {b: [] for b in ['energy', 'mood', 'focus', 'creativity', 'stress']}
    
    print("="*60)
    print("DAILY CYCLE SIMULATION")
    print("="*60)
    
    for time_point, activity, message, expected_mood in scenarios:
        # Process the activity
        response = system.process_message(message)
        
        # Store results
        timeline.append(time_point)
        for h in hormones:
            hormones[h].append(response['hormones'][h])
        moods.append(response['mood'])
        for b in behaviors:
            if b in response['behavioral']:
                behaviors[b].append(response['behavioral'][b])
        
        # Print summary
        print(f"\n⏰ Time: {time_point:3d}s | {activity:15s}")
        print(f"Message: '{message[:40]}...'")
        print(f"Mood: {response['mood']:10s} | Expected: {expected_mood}")
        print(f"D:{response['hormones']['dopamine']:5.1f} "
              f"S:{response['hormones']['serotonin']:5.1f} "
              f"C:{response['hormones']['cortisol']:5.1f} "
              f"A:{response['hormones']['adrenaline']:5.1f} "
              f"O:{response['hormones']['oxytocin']:5.1f} "
              f"E:{response['hormones']['endorphins']:5.1f}")
        
        # Let some time pass between activities
        if time_point < 280:
            system.simulate_time_passage(2, rest=False)
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Daily Neurochemical Cycle', fontsize=16, fontweight='bold')
    
    # Plot primary hormones
    ax1 = axes[0, 0]
    ax1.plot(timeline, hormones['dopamine'], 'r-', label='Dopamine', linewidth=2)
    ax1.plot(timeline, hormones['serotonin'], 'b-', label='Serotonin', linewidth=2)
    ax1.plot(timeline, hormones['cortisol'], 'orange', label='Cortisol', linewidth=2)
    ax1.set_title('Primary Neurotransmitters')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot stress/energy hormones
    ax2 = axes[0, 1]
    ax2.plot(timeline, hormones['adrenaline'], 'red', label='Adrenaline', linewidth=2)
    ax2.plot(timeline, hormones['norepinephrine'], 'darkred', label='Norepinephrine', linewidth=2)
    ax2.set_title('Stress/Energy Hormones')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot social/pleasure hormones
    ax3 = axes[1, 0]
    ax3.plot(timeline, hormones['oxytocin'], 'pink', label='Oxytocin', linewidth=2)
    ax3.plot(timeline, hormones['endorphins'], 'purple', label='Endorphins', linewidth=2)
    ax3.set_title('Social/Pleasure Hormones')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Level')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot behavioral parameters
    ax4 = axes[1, 1]
    ax4.plot(timeline[:len(behaviors['energy'])], behaviors['energy'], 'g-', label='Energy', linewidth=2)
    ax4.plot(timeline[:len(behaviors['mood'])], behaviors['mood'], 'm-', label='Mood', linewidth=2)
    ax4.plot(timeline[:len(behaviors['focus'])], behaviors['focus'], 'c-', label='Focus', linewidth=2)
    ax4.set_title('Behavioral States')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Parameter Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Plot activity markers
    ax5 = axes[2, 0]
    activity_names = [s[1] for s in scenarios]
    activity_times = [s[0] for s in scenarios]
    ax5.scatter(activity_times, range(len(activity_names)), c=activity_times, cmap='viridis', s=100)
    ax5.set_yticks(range(len(activity_names)))
    ax5.set_yticklabels(activity_names, fontsize=8)
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Activity Timeline')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Plot mood transitions
    ax6 = axes[2, 1]
    unique_moods = list(set(moods))
    mood_indices = [unique_moods.index(m) for m in moods]
    ax6.plot(timeline, mood_indices, 'ko-', linewidth=2, markersize=8)
    ax6.set_yticks(range(len(unique_moods)))
    ax6.set_yticklabels(unique_moods)
    ax6.set_xlabel('Time (s)')
    ax6.set_title('Mood State Evolution')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/root/openhermes_backend/daily_cycle.png', dpi=150, bbox_inches='tight')
    print("\n📊 Daily cycle graph saved as 'daily_cycle.png'")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("-"*40)
    print(f"Dopamine range: {min(hormones['dopamine']):.1f} - {max(hormones['dopamine']):.1f}")
    print(f"Serotonin range: {min(hormones['serotonin']):.1f} - {max(hormones['serotonin']):.1f}")
    print(f"Cortisol range: {min(hormones['cortisol']):.1f} - {max(hormones['cortisol']):.1f}")
    print(f"Endorphin peak: {max(hormones['endorphins']):.1f} (during exercise)")
    print(f"Oxytocin peak: {max(hormones['oxytocin']):.1f} (during family time)")
    print(f"Unique moods: {set(moods)}")

if __name__ == "__main__":
    test_daily_cycle()
