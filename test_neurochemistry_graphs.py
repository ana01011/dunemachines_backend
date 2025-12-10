"""
Comprehensive test with visualization of the 7D neurochemistry system
"""

import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.interface import NeurochemicalSystem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import json

def test_with_graphs():
    """Test with comprehensive visualization"""
    
    # Create system
    system = NeurochemicalSystem(user_id="test_user")
    
    # Prepare data storage
    time_points = []
    hormones_history = {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                                         'oxytocin', 'norepinephrine', 'endorphins']}
    receptors_history = {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline',
                                          'oxytocin', 'norepinephrine', 'endorphins']}
    baselines_history = {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline',
                                          'oxytocin', 'norepinephrine', 'endorphins']}
    behavioral_history = {b: [] for b in ['energy', 'mood', 'focus', 'creativity', 
                                          'empathy', 'confidence', 'stress', 'motivation']}
    cost_history = {c: [] for c in ['deviation', 'change', 'metabolic', 'uncertainty', 
                                     'allostatic', 'total']}
    resources_history = {'tyrosine': [], 'tryptophan': [], 'atp': []}
    allostatic_history = []
    mood_history = []
    
    # Simulation timeline (messages at different times)
    timeline = [
        (0, None, "Initial state"),
        (2, "This is amazing! I'm so happy!", "Positive feedback"),
        (5, "Normal conversation", "Neutral"),
        (8, "URGENT! Emergency situation!", "Stress/Urgency"),
        (12, "Calm down, everything is okay", "Calming"),
        (15, "Let's work together as a team", "Social bonding"),
        (18, "I'm worried about this problem", "Anxiety"),
        (22, "Great job! You did it!", "Reward"),
        (25, "Time to rest", "Rest period"),
        (30, None, "Recovery")
    ]
    
    # Run simulation
    current_time = 0
    for target_time, message, label in timeline:
        # Simulate time until next event
        while current_time < target_time:
            if message is None:
                # Just time passage
                system.simulate_time_passage(0.5, rest=(label == "Recovery"))
            else:
                if current_time >= target_time - 0.1:
                    # Process message
                    response = system.process_message(message)
            
            # Record state
            response = system.get_state_summary()
            full_response = system.process_message("")  # Get full state without new input
            
            time_points.append(current_time)
            
            # Record hormones
            for h in hormones_history:
                hormones_history[h].append(full_response['hormones'][h])
            
            # Record receptors
            for r in receptors_history:
                receptors_history[r].append(full_response['receptors'][r])
                
            # Record baselines
            for b in baselines_history:
                baselines_history[b].append(full_response['baselines'][b])
            
            # Record behavioral
            for b in behavioral_history:
                behavioral_history[b].append(full_response['behavioral'][b])
            
            # Record costs
            for c in cost_history:
                if c in full_response['cost']:
                    cost_history[c].append(full_response['cost'][c])
            
            # Record resources
            resources_history['tyrosine'].append(full_response['resources']['tyrosine'])
            resources_history['tryptophan'].append(full_response['resources']['tryptophan'])
            resources_history['atp'].append(full_response['resources']['atp'])
            
            # Record allostatic load
            allostatic_history.append(full_response['allostatic_load'])
            
            # Record mood
            mood_history.append(response['mood'])
            
            current_time += 0.5
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Color scheme for hormones
    hormone_colors = {
        'dopamine': '#FF6B6B',      # Red
        'serotonin': '#4ECDC4',     # Teal
        'cortisol': '#FFE66D',      # Yellow
        'adrenaline': '#FF8C42',    # Orange
        'oxytocin': '#95E1D3',      # Mint
        'norepinephrine': '#C77DFF', # Purple
        'endorphins': '#FFB6C1'     # Pink
    }
    
    # Plot 1: All Hormones Over Time
    ax1 = fig.add_subplot(gs[0, :])
    for hormone, color in hormone_colors.items():
        ax1.plot(time_points, hormones_history[hormone], label=hormone.capitalize(), 
                color=color, linewidth=2)
    
    # Add event markers
    for target_time, message, label in timeline:
        if message is not None:
            ax1.axvline(x=target_time, color='gray', linestyle='--', alpha=0.5)
            ax1.text(target_time, 95, label, rotation=90, fontsize=8, va='top')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Hormone Level')
    ax1.set_title('7D Hormone Dynamics Over Time', fontsize=14, fontweight='bold')
    ax1.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Receptors Sensitivity
    ax2 = fig.add_subplot(gs[1, 0])
    for receptor in ['dopamine', 'serotonin', 'cortisol']:
        ax2.plot(time_points, receptors_history[receptor], label=receptor.capitalize(),
                linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Receptor Sensitivity')
    ax2.set_title('Primary Receptor Adaptation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Baselines Evolution
    ax3 = fig.add_subplot(gs[1, 1])
    for baseline in ['dopamine', 'serotonin', 'cortisol']:
        ax3.plot(time_points, baselines_history[baseline], label=baseline.capitalize(),
                linewidth=2, linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Baseline Level')
    ax3.set_title('Adaptive Baselines (Minimization Principle)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost Components
    ax4 = fig.add_subplot(gs[1, 2])
    for cost_type in ['deviation', 'metabolic', 'uncertainty']:
        if cost_type in cost_history and cost_history[cost_type]:
            ax4.plot(time_points, cost_history[cost_type], label=cost_type.capitalize(),
                    linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Cost')
    ax4.set_title('Cost Function Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Behavioral Parameters
    ax5 = fig.add_subplot(gs[2, :])
    behavioral_colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for i, (behavior, values) in enumerate(behavioral_history.items()):
        ax5.plot(time_points, values, label=behavior.capitalize(), 
                color=behavioral_colors[i], linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Parameter Value')
    ax5.set_title('Behavioral Outputs', fontsize=12, fontweight='bold')
    ax5.legend(ncol=8, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # Plot 6: Resources
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(time_points, resources_history['tyrosine'], label='Tyrosine', color='blue')
    ax6.plot(time_points, resources_history['tryptophan'], label='Tryptophan', color='green')
    ax6.plot(time_points, resources_history['atp'], label='ATP', color='red')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Resource Level (%)')
    ax6.set_title('Metabolic Resources')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.1])
    
    # Plot 7: Allostatic Load
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(time_points, allostatic_history, color='darkred', linewidth=2)
    ax7.fill_between(time_points, 0, allostatic_history, alpha=0.3, color='darkred')
    ax7.axhline(y=0.3, color='orange', linestyle='--', label='Threshold')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Allostatic Load')
    ax7.set_title('Chronic Stress Accumulation')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1])
    
    # Plot 8: Total Cost Over Time
    ax8 = fig.add_subplot(gs[3, 2])
    if 'total' in cost_history and cost_history['total']:
        ax8.plot(time_points, cost_history['total'], color='purple', linewidth=2)
        ax8.fill_between(time_points, 0, cost_history['total'], alpha=0.3, color='purple')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Total Cost')
    ax8.set_title('System Cost (Minimization Target)')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Phase Space - Dopamine vs Serotonin
    ax9 = fig.add_subplot(gs[4, 0])
    scatter = ax9.scatter(hormones_history['dopamine'], hormones_history['serotonin'], 
                         c=time_points, cmap='viridis', s=20, alpha=0.6)
    ax9.set_xlabel('Dopamine')
    ax9.set_ylabel('Serotonin')
    ax9.set_title('Phase Space: D-S Trajectory')
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax9, label='Time (s)')
    
    # Plot 10: Phase Space - Cortisol vs Adrenaline
    ax10 = fig.add_subplot(gs[4, 1])
    scatter2 = ax10.scatter(hormones_history['cortisol'], hormones_history['adrenaline'],
                           c=time_points, cmap='plasma', s=20, alpha=0.6)
    ax10.set_xlabel('Cortisol')
    ax10.set_ylabel('Adrenaline')
    ax10.set_title('Phase Space: Stress Response')
    ax10.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax10, label='Time (s)')
    
    # Plot 11: Mood State Evolution
    ax11 = fig.add_subplot(gs[4, 2])
    mood_colors = {'neutral': 'gray', 'euphoric': 'gold', 'depressed': 'darkblue',
                   'anxious': 'red', 'focused': 'green', 'content': 'lightblue',
                   'joyful': 'yellow', 'stressed': 'orange'}
    
    # Convert mood to numeric for plotting
    mood_numeric = []
    mood_labels = list(set(mood_history))
    for mood in mood_history:
        mood_numeric.append(mood_labels.index(mood))
    
    ax11.plot(time_points, mood_numeric, linewidth=2, color='darkgreen')
    ax11.set_yticks(range(len(mood_labels)))
    ax11.set_yticklabels(mood_labels)
    ax11.set_xlabel('Time (s)')
    ax11.set_title('Mood State Transitions')
    ax11.grid(True, alpha=0.3, axis='x')
    
    # Plot 12: Correlation Matrix
    ax12 = fig.add_subplot(gs[5, :])
    
    # Calculate correlations between key hormones
    hormone_data = np.array([hormones_history[h] for h in ['dopamine', 'serotonin', 
                                                            'cortisol', 'adrenaline', 'oxytocin']])
    correlation = np.corrcoef(hormone_data)
    
    im = ax12.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax12.set_xticks(range(5))
    ax12.set_yticks(range(5))
    ax12.set_xticklabels(['D', 'S', 'C', 'A', 'O'])
    ax12.set_yticklabels(['D', 'S', 'C', 'A', 'O'])
    ax12.set_title('Hormone Correlation Matrix')
    
    # Add correlation values
    for i in range(5):
        for j in range(5):
            ax12.text(j, i, f'{correlation[i, j]:.2f}', ha='center', va='center')
    
    plt.colorbar(im, ax=ax12, label='Correlation')
    
    # Main title
    fig.suptitle('7D Neurochemical System Dynamics - Complete Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('/root/openhermes_backend/neurochemistry_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Graph saved as 'neurochemistry_analysis.png'")
    print("\n📈 Key Observations:")
    print(f"- Dopamine range: {min(hormones_history['dopamine']):.1f} - {max(hormones_history['dopamine']):.1f}")
    print(f"- Serotonin range: {min(hormones_history['serotonin']):.1f} - {max(hormones_history['serotonin']):.1f}")
    print(f"- Cortisol range: {min(hormones_history['cortisol']):.1f} - {max(hormones_history['cortisol']):.1f}")
    print(f"- Max allostatic load: {max(allostatic_history):.2f}")
    print(f"- Average total cost: {np.mean(cost_history['total']):.2f}")
    print(f"- Mood states observed: {set(mood_history)}")

if __name__ == "__main__":
    test_with_graphs()
