"""
Comprehensive Animation of Enhanced Neurochemistry System
Shows opponent processes, minimization principle, and all dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import sys
import os

# Add path for imports
sys.path.insert(0, '/root/openhermes_backend')

# Import enhanced modules
import importlib.util

spec_state = importlib.util.spec_from_file_location(
    "state_enhanced", 
    "/root/openhermes_backend/app/neurochemistry/core/state_enhanced.py"
)
state_enhanced = importlib.util.module_from_spec(spec_state)
spec_state.loader.exec_module(state_enhanced)

spec_dynamics = importlib.util.spec_from_file_location(
    "dynamics_enhanced",
    "/root/openhermes_backend/app/neurochemistry/core/dynamics_enhanced.py"
)
dynamics_enhanced = importlib.util.module_from_spec(spec_dynamics)
spec_dynamics.loader.exec_module(dynamics_enhanced)

NeurochemicalState = state_enhanced.NeurochemicalState
NeurochemicalDynamics = dynamics_enhanced.NeurochemicalDynamics

class EnhancedNeurochemistryAnimation:
    def __init__(self, duration=120, dt=0.1):
        self.state = NeurochemicalState()
        self.dynamics = NeurochemicalDynamics(self.state)
        self.duration = duration
        self.dt = dt
        self.frames = int(duration / dt)
        
        # Storage for animation data
        self.time_data = []
        self.hormone_data = {h: [] for h in self.state.hormones}
        self.baseline_data = {h: [] for h in self.state.hormones}
        self.ratio_data = {h: [] for h in self.state.hormones}
        self.resource_data = {r: [] for r in self.state.resources}
        self.cost_data = {'total': [], 'production': [], 'maintenance': []}
        self.spike_data = {h: [] for h in self.state.hormones}
        self.mood_data = []
        self.load_data = []
        
        # Scenario timeline
        self.scenarios = [
            (0, 10, "Baseline", {}),
            (10, 20, "Joy (Dopamine)", {'reward': 0.8, 'novelty': 0.5}),
            (20, 30, "Stress (Cortisol)", {'threat': 0.7, 'uncertainty': 0.5}),
            (30, 40, "Exercise", {'exercise': 0.8, 'pleasure': 0.3}),
            (40, 50, "Relaxation", {'relaxation': 0.8}),
            (50, 60, "Social", {'social': 0.8, 'trust': 0.6}),
            (60, 70, "Sadness", {'threat': 0.3, 'uncertainty': 0.6}),
            (70, 80, "Recovery", {}),
            (80, 90, "Multiple Joy", {'reward': 0.8}),  # Test baseline adaptation
            (90, 100, "Multiple Joy 2", {'reward': 0.8}),  # Should need less production
            (100, 110, "Multiple Joy 3", {'reward': 0.8}),  # Even less production
            (110, 120, "Final Recovery", {}),
        ]
        
    def get_current_inputs(self, t):
        """Get inputs for current time based on scenarios"""
        for start, end, name, inputs in self.scenarios:
            if start <= t < end:
                return name, inputs
        return "Baseline", {}
    
    def init_figure(self):
        """Initialize the figure with subplots"""
        self.fig = plt.figure(figsize=(20, 12), facecolor='#0a0a0a')
        self.fig.suptitle('Enhanced Neurochemistry: Opponent Processes & Minimization Principle', 
                          fontsize=16, color='white', fontweight='bold')
        
        # Create grid
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 1. Main hormone levels with baselines
        self.ax_hormones = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_hormones.set_facecolor('#1a1a1a')
        self.ax_hormones.set_title('Hormone Levels & Baselines', color='white')
        self.ax_hormones.set_xlabel('Time (s)', color='white')
        self.ax_hormones.set_ylabel('Level', color='white')
        self.ax_hormones.grid(True, alpha=0.2)
        
        # 2. Opponent ratios
        self.ax_ratios = self.fig.add_subplot(gs[0, 2:4])
        self.ax_ratios.set_facecolor('#1a1a1a')
        self.ax_ratios.set_title('Opponent Process Ratios', color='white')
        self.ax_ratios.set_xlabel('Time (s)', color='white')
        self.ax_ratios.set_ylabel('Ratio', color='white')
        self.ax_ratios.grid(True, alpha=0.2)
        
        # 3. Cost minimization
        self.ax_cost = self.fig.add_subplot(gs[1, 2])
        self.ax_cost.set_facecolor('#1a1a1a')
        self.ax_cost.set_title('Cost Minimization', color='white')
        self.ax_cost.set_xlabel('Time (s)', color='white')
        self.ax_cost.set_ylabel('Cost', color='white')
        self.ax_cost.grid(True, alpha=0.2)
        
        # 4. Spike history & baseline adaptation
        self.ax_spikes = self.fig.add_subplot(gs[1, 3])
        self.ax_spikes.set_facecolor('#1a1a1a')
        self.ax_spikes.set_title('Spike Detection & Baseline Shift', color='white')
        self.ax_spikes.set_xlabel('Time (s)', color='white')
        self.ax_spikes.set_ylabel('Amplitude', color='white')
        self.ax_spikes.grid(True, alpha=0.2)
        
        # 5. Resources & metabolism
        self.ax_resources = self.fig.add_subplot(gs[2, 0])
        self.ax_resources.set_facecolor('#1a1a1a')
        self.ax_resources.set_title('Metabolic Resources', color='white')
        self.ax_resources.set_xlabel('Time (s)', color='white')
        self.ax_resources.set_ylabel('Level', color='white')
        self.ax_resources.grid(True, alpha=0.2)
        
        # 6. Production vs baseline (minimization visualization)
        self.ax_production = self.fig.add_subplot(gs[2, 1])
        self.ax_production.set_facecolor('#1a1a1a')
        self.ax_production.set_title('Production from Baseline', color='white')
        self.ax_production.set_xlabel('Time (s)', color='white')
        self.ax_production.set_ylabel('Production', color='white')
        self.ax_production.grid(True, alpha=0.2)
        
        # 7. Opponent pairs interaction
        self.ax_opponents = self.fig.add_subplot(gs[2, 2:4])
        self.ax_opponents.set_facecolor('#1a1a1a')
        self.ax_opponents.set_title('Opponent Pair Dynamics', color='white')
        self.ax_opponents.set_xlabel('Time (s)', color='white')
        self.ax_opponents.set_ylabel('Level', color='white')
        self.ax_opponents.grid(True, alpha=0.2)
        
        # 8. Mood & allostatic load
        self.ax_mood = self.fig.add_subplot(gs[3, 0:2])
        self.ax_mood.set_facecolor('#1a1a1a')
        self.ax_mood.set_title('Mood State & Allostatic Load', color='white')
        self.ax_mood.set_xlabel('Time (s)', color='white')
        self.ax_mood.grid(True, alpha=0.2)
        
        # 9. Scenario timeline
        self.ax_scenario = self.fig.add_subplot(gs[3, 2:4])
        self.ax_scenario.set_facecolor('#1a1a1a')
        self.ax_scenario.set_title('Current Scenario', color='white')
        self.ax_scenario.set_xlim(0, self.duration)
        self.ax_scenario.set_ylim(0, 1)
        
        # Initialize lines
        self.lines = {}
        colors = {
            'dopamine': '#FF6B6B',
            'serotonin': '#4ECDC4',
            'cortisol': '#FFA500',
            'adrenaline': '#FF4444',
            'oxytocin': '#FF69B4',
            'norepinephrine': '#9B59B6',
            'endorphins': '#3498DB'
        }
        
        # Hormone lines
        for hormone, color in colors.items():
            self.lines[f'{hormone}'] = self.ax_hormones.plot(
                [], [], label=hormone, color=color, linewidth=2
            )[0]
            self.lines[f'{hormone}_baseline'] = self.ax_hormones.plot(
                [], [], '--', color=color, alpha=0.5, linewidth=1
            )[0]
        
        self.ax_hormones.legend(loc='upper right', facecolor='#2a2a2a', 
                               edgecolor='white', labelcolor='white')
        
        # Ratio lines
        for hormone, color in colors.items():
            self.lines[f'{hormone}_ratio'] = self.ax_ratios.plot(
                [], [], label=f'{hormone[:3]}', color=color, linewidth=1.5
            )[0]
        
        # Cost lines
        self.lines['cost_total'] = self.ax_cost.plot(
            [], [], 'w-', label='Total', linewidth=2
        )[0]
        self.lines['cost_production'] = self.ax_cost.plot(
            [], [], 'r-', label='Production', linewidth=1
        )[0]
        
        # Resource lines
        self.lines['tyrosine'] = self.ax_resources.plot(
            [], [], 'y-', label='Tyrosine', linewidth=2
        )[0]
        self.lines['tryptophan'] = self.ax_resources.plot(
            [], [], 'c-', label='Tryptophan', linewidth=2
        )[0]
        self.lines['atp'] = self.ax_resources.plot(
            [], [], 'g-', label='ATP', linewidth=2
        )[0]
        
        # Opponent pair lines
        self.lines['dopamine_serotonin'] = self.ax_opponents.plot(
            [], [], 'r-', label='Dopamine', linewidth=2
        )[0]
        self.lines['serotonin_dopamine'] = self.ax_opponents.plot(
            [], [], 'c-', label='Serotonin', linewidth=2
        )[0]
        self.lines['cortisol_oxytocin'] = self.ax_opponents.plot(
            [], [], color='orange', label='Cortisol', linewidth=2
        )[0]
        self.lines['oxytocin_cortisol'] = self.ax_opponents.plot(
            [], [], color='pink', label='Oxytocin', linewidth=2
        )[0]
        
        # Allostatic load
        self.lines['allostatic'] = self.ax_mood.plot(
            [], [], 'r-', label='Allostatic Load', linewidth=2
        )[0]
        
        # Set text color for all axes
        for ax in [self.ax_hormones, self.ax_ratios, self.ax_cost, self.ax_spikes,
                   self.ax_resources, self.ax_production, self.ax_opponents, 
                   self.ax_mood, self.ax_scenario]:
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
        
        return self.lines.values()
    
    def animate(self, frame):
        """Animation function"""
        t = frame * self.dt
        scenario_name, inputs = self.get_current_inputs(t)
        
        # Step dynamics
        self.dynamics.step(self.dt, inputs)
        
        # Collect data
        self.time_data.append(t)
        
        for h in self.state.hormones:
            self.hormone_data[h].append(self.state.hormones[h])
            self.baseline_data[h].append(self.state.baselines['slow'][h])
            self.ratio_data[h].append(self.state.opponent_ratios.get(h, 1.0))
            self.spike_data[h].append(self.state.get_average_spike(h))
        
        for r in self.state.resources:
            self.resource_data[r].append(self.state.resources[r])
        
        costs = self.state.calculate_total_cost()
        self.cost_data['total'].append(costs['total'])
        self.cost_data['production'].append(costs['production'] * 10)  # Scale for visibility
        
        self.mood_data.append(self.state.get_mood())
        self.load_data.append(self.state.allostatic_load * 100)  # Scale to percentage
        
        # Update plots
        if len(self.time_data) > 1:
            # Hormones and baselines
            for hormone in self.state.hormones:
                self.lines[f'{hormone}'].set_data(
                    self.time_data, self.hormone_data[hormone]
                )
                self.lines[f'{hormone}_baseline'].set_data(
                    self.time_data, self.baseline_data[hormone]
                )
                self.lines[f'{hormone}_ratio'].set_data(
                    self.time_data, self.ratio_data[hormone]
                )
            
            # Cost
            self.lines['cost_total'].set_data(
                self.time_data, self.cost_data['total']
            )
            self.lines['cost_production'].set_data(
                self.time_data, self.cost_data['production']
            )
            
            # Resources
            self.lines['tyrosine'].set_data(
                self.time_data, self.resource_data['tyrosine']
            )
            self.lines['tryptophan'].set_data(
                self.time_data, self.resource_data['tryptophan']
            )
            self.lines['atp'].set_data(
                self.time_data, self.resource_data['atp']
            )
            
            # Opponent pairs
            self.lines['dopamine_serotonin'].set_data(
                self.time_data, self.hormone_data['dopamine']
            )
            self.lines['serotonin_dopamine'].set_data(
                self.time_data, self.hormone_data['serotonin']
            )
            self.lines['cortisol_oxytocin'].set_data(
                self.time_data, self.hormone_data['cortisol']
            )
            self.lines['oxytocin_cortisol'].set_data(
                self.time_data, self.hormone_data['oxytocin']
            )
            
            # Allostatic load
            self.lines['allostatic'].set_data(
                self.time_data, self.load_data
            )
            
            # Update axis limits
            self.ax_hormones.set_xlim(0, max(10, t))
            self.ax_hormones.set_ylim(0, 100)
            
            self.ax_ratios.set_xlim(0, max(10, t))
            self.ax_ratios.set_ylim(0, 5)
            
            self.ax_cost.set_xlim(0, max(10, t))
            self.ax_cost.set_ylim(0, max(self.cost_data['total'] + [100]))
            
            self.ax_resources.set_xlim(0, max(10, t))
            self.ax_resources.set_ylim(0, 1.1)
            
            self.ax_opponents.set_xlim(0, max(10, t))
            self.ax_opponents.set_ylim(0, 100)
            
            self.ax_mood.set_xlim(0, max(10, t))
            self.ax_mood.set_ylim(0, 100)
            
            # Draw spike bars
            self.ax_spikes.clear()
            self.ax_spikes.set_facecolor('#1a1a1a')
            self.ax_spikes.set_title('Current Spike Amplitudes', color='white')
            self.ax_spikes.grid(True, alpha=0.2)
            
            spike_hormones = []
            spike_values = []
            for h in self.state.hormones:
                if self.spike_data[h][-1] > 0:
                    spike_hormones.append(h[:3])
                    spike_values.append(self.spike_data[h][-1])
            
            if spike_hormones:
                self.ax_spikes.bar(spike_hormones, spike_values, 
                                  color=['#FF6B6B', '#4ECDC4', '#FFA500', '#FF4444',
                                        '#FF69B4', '#9B59B6', '#3498DB'][:len(spike_hormones)])
            
            # Draw production bars
            self.ax_production.clear()
            self.ax_production.set_facecolor('#1a1a1a')
            self.ax_production.set_title('Production Above Baseline', color='white')
            self.ax_production.grid(True, alpha=0.2)
            
            production_values = []
            for h in self.state.hormones:
                production = max(0, self.state.hormones[h] - self.state.baselines['slow'][h])
                production_values.append(production)
            
            self.ax_production.bar(['Dop', 'Ser', 'Cor', 'Adr', 'Oxy', 'Nor', 'End'],
                                  production_values,
                                  color=['#FF6B6B', '#4ECDC4', '#FFA500', '#FF4444',
                                        '#FF69B4', '#9B59B6', '#3498DB'])
            
            # Update scenario indicator
            self.ax_scenario.clear()
            self.ax_scenario.set_facecolor('#1a1a1a')
            self.ax_scenario.set_title(f'Scenario: {scenario_name}', color='white', fontsize=14)
            self.ax_scenario.set_xlim(0, self.duration)
            self.ax_scenario.set_ylim(0, 1)
            
            # Draw scenario timeline
            for start, end, name, _ in self.scenarios:
                color = 'yellow' if start <= t < end else 'gray'
                alpha = 0.8 if start <= t < end else 0.3
                self.ax_scenario.barh(0.5, end - start, left=start, 
                                     height=0.3, color=color, alpha=alpha)
                if start <= t < end:
                    self.ax_scenario.text((start + end) / 2, 0.5, name,
                                        ha='center', va='center', color='black',
                                        fontweight='bold')
            
            # Current mood text
            self.ax_mood.text(t, 90, f'Mood: {self.mood_data[-1]}', 
                            color='white', fontsize=10, ha='right')
            
            # Format axes
            for ax in [self.ax_spikes, self.ax_production, self.ax_scenario]:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
        
        return self.lines.values()
    
    def create_animation(self, filename='enhanced_neurochemistry.mp4'):
        """Create and save the animation"""
        print("Initializing animation...")
        self.init_figure()
        
        print(f"Creating {self.duration} second animation...")
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.frames,
            interval=50, blit=False
        )
        
        print(f"Saving animation to {filename}...")
        writer = animation.FFMpegWriter(fps=20, bitrate=2000)
        anim.save(filename, writer=writer, dpi=100)
        
        plt.close()
        print(f"✅ Animation saved as {filename}")
        
        return filename

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED NEUROCHEMISTRY ANIMATION")
    print("Visualizing Opponent Processes & Minimization Principle")
    print("="*60)
    
    animator = EnhancedNeurochemistryAnimation(duration=120, dt=0.1)
    output_file = animator.create_animation('/root/openhermes_backend/enhanced_neurochemistry.mp4')
    
    print(f"\n✅ Animation complete!")
    print(f"📹 Video saved as: {output_file}")
    print("\nThe animation shows:")
    print("1. Hormone levels with adaptive baselines")
    print("2. Opponent process ratios")
    print("3. Cost minimization over time")
    print("4. Spike detection and baseline adaptation")
    print("5. Resource depletion and recovery")
    print("6. Production above baseline (minimization target)")
    print("7. Opponent pair dynamics")
    print("8. Mood states and allostatic load")
    print("\nNotice how repeated dopamine spikes lead to baseline adaptation,")
    print("reducing the production cost for subsequent spikes!")
