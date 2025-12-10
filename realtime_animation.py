"""
Real-time animated visualization of 7D neurochemical dynamics
Shows hormone waves, baseline adaptation, and minimization principle
"""

import sys
sys.path.append('/root/openhermes_backend')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from collections import deque
import time

from app.neurochemistry.interface import NeurochemicalSystem

class NeurochemicalAnimator:
    def __init__(self, window_size=100):
        """Initialize the animation system"""
        self.system = NeurochemicalSystem()
        self.system.dt = 0.2  # 200ms timesteps for smooth animation
        
        # Data storage
        self.window_size = window_size
        self.time_data = deque(maxlen=window_size)
        self.hormone_data = {h: deque(maxlen=window_size) for h in 
                            ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                             'oxytocin', 'norepinephrine', 'endorphins']}
        self.baseline_data = {h: deque(maxlen=window_size) for h in 
                             ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                              'oxytocin', 'norepinephrine', 'endorphins']}
        self.receptor_data = {h: deque(maxlen=window_size) for h in 
                             ['dopamine', 'serotonin', 'cortisol']}
        self.cost_data = deque(maxlen=window_size)
        self.mood_data = deque(maxlen=window_size)
        
        # Scenario timeline
        self.scenarios = [
            (5, "baseline", "Resting state"),
            (15, "joy", "Wonderful news! So happy!"),
            (25, "stress", "Urgent problem! Very stressful!"),
            (35, "exercise", "Running! Feeling the burn!"),
            (45, "relax", "Deep breathing, meditation, calm..."),
            (55, "social", "Hugging loved ones, feeling connected"),
            (65, "sad", "Feeling down and sad..."),
            (75, "recovery", "Time passing, returning to baseline"),
        ]
        self.scenario_index = 0
        self.last_scenario_time = 0
        
        # Animation state
        self.current_time = 0
        self.frame_count = 0
        
        # Setup the figure
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Real-Time 7D Neurochemical Dynamics', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Main hormone plot (top, spans 2 columns)
        self.ax_hormones = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_hormones.set_title('Hormone Levels & Baseline Adaptation')
        self.ax_hormones.set_xlabel('Time (seconds)')
        self.ax_hormones.set_ylabel('Level')
        self.ax_hormones.set_ylim([0, 100])
        self.ax_hormones.grid(True, alpha=0.3)
        
        # Initialize hormone lines
        self.hormone_lines = {}
        self.baseline_lines = {}
        colors = {
            'dopamine': '#FF6B6B',
            'serotonin': '#4ECDC4',
            'cortisol': '#FFE66D',
            'adrenaline': '#FF8C42',
            'oxytocin': '#95E1D3',
            'norepinephrine': '#C77DFF',
            'endorphins': '#FFB6C1'
        }
        
        for hormone, color in colors.items():
            # Hormone level line (solid)
            line, = self.ax_hormones.plot([], [], color=color, linewidth=2, 
                                         label=hormone.capitalize(), alpha=0.8)
            self.hormone_lines[hormone] = line
            
            # Baseline line (dashed)
            if hormone in ['dopamine', 'serotonin', 'cortisol']:
                baseline_line, = self.ax_hormones.plot([], [], color=color, 
                                                      linewidth=1, linestyle='--', alpha=0.5)
                self.baseline_lines[hormone] = baseline_line
        
        self.ax_hormones.legend(loc='upper left', ncol=4, fontsize=8)
        
        # Cost function plot (top right)
        self.ax_cost = self.fig.add_subplot(gs[0, 2])
        self.ax_cost.set_title('Minimization Cost')
        self.ax_cost.set_xlabel('Time (s)')
        self.ax_cost.set_ylabel('Total Cost')
        self.ax_cost.grid(True, alpha=0.3)
        self.cost_line, = self.ax_cost.plot([], [], 'purple', linewidth=2)
        
        # Receptor sensitivity plot (middle right)
        self.ax_receptors = self.fig.add_subplot(gs[1, 2])
        self.ax_receptors.set_title('Receptor Adaptation')
        self.ax_receptors.set_xlabel('Time (s)')
        self.ax_receptors.set_ylabel('Sensitivity')
        self.ax_receptors.set_ylim([0, 1])
        self.ax_receptors.grid(True, alpha=0.3)
        
        self.receptor_lines = {}
        for hormone in ['dopamine', 'serotonin', 'cortisol']:
            line, = self.ax_receptors.plot([], [], label=f'R_{hormone[:3]}', linewidth=2)
            self.receptor_lines[hormone] = line
        self.ax_receptors.legend(loc='upper right', fontsize=8)
        
        # Wave visualization (bottom left)
        self.ax_waves = self.fig.add_subplot(gs[2, 0:2])
        self.ax_waves.set_title('Hormone Waves (Deviation from Baseline)')
        self.ax_waves.set_xlabel('Time (s)')
        self.ax_waves.set_ylabel('Wave Amplitude')
        self.ax_waves.set_ylim([-30, 30])
        self.ax_waves.grid(True, alpha=0.3)
        self.ax_waves.axhline(y=0, color='black', linewidth=0.5)
        
        self.wave_lines = {}
        for hormone in ['dopamine', 'serotonin', 'cortisol']:
            line, = self.ax_waves.plot([], [], linewidth=1.5, label=f'{hormone[:3]} wave')
            self.wave_lines[hormone] = line
        self.ax_waves.legend(loc='upper right', fontsize=8)
        
        # Current state display (bottom right)
        self.ax_state = self.fig.add_subplot(gs[2:4, 2])
        self.ax_state.set_title('Current State')
        self.ax_state.set_xlim([0, 1])
        self.ax_state.set_ylim([0, 8])
        self.ax_state.axis('off')
        
        # State text elements
        self.state_texts = {
            'mood': self.ax_state.text(0.1, 7, 'Mood: ', fontsize=12, fontweight='bold'),
            'scenario': self.ax_state.text(0.1, 6, 'Scenario: ', fontsize=10),
            'd_val': self.ax_state.text(0.1, 5, 'D: ', fontsize=9),
            's_val': self.ax_state.text(0.1, 4.5, 'S: ', fontsize=9),
            'c_val': self.ax_state.text(0.1, 4, 'C: ', fontsize=9),
            'a_val': self.ax_state.text(0.1, 3.5, 'A: ', fontsize=9),
            'o_val': self.ax_state.text(0.1, 3, 'O: ', fontsize=9),
            'n_val': self.ax_state.text(0.1, 2.5, 'N: ', fontsize=9),
            'e_val': self.ax_state.text(0.1, 2, 'E: ', fontsize=9),
            'cost': self.ax_state.text(0.1, 1, 'Cost: ', fontsize=9),
            'time': self.ax_state.text(0.1, 0.5, 'Time: ', fontsize=9),
        }
        
        # Phase space plot (bottom left corner)
        self.ax_phase = self.fig.add_subplot(gs[3, 0])
        self.ax_phase.set_title('D-S Phase Space')
        self.ax_phase.set_xlabel('Dopamine')
        self.ax_phase.set_ylabel('Serotonin')
        self.ax_phase.set_xlim([30, 80])
        self.ax_phase.set_ylim([30, 80])
        self.ax_phase.grid(True, alpha=0.3)
        self.phase_line, = self.ax_phase.plot([], [], 'b-', linewidth=1, alpha=0.5)
        self.phase_point, = self.ax_phase.plot([], [], 'ro', markersize=8)
        
        # Seeking behavior bars (bottom middle)
        self.ax_seeking = self.fig.add_subplot(gs[3, 1])
        self.ax_seeking.set_title('Seeking Behavior')
        self.ax_seeking.set_ylim([0, 1])
        self.ax_seeking.set_ylabel('Intensity')
        self.seeking_bars = self.ax_seeking.bar(['D', 'S', 'O', 'E'], [0, 0, 0, 0])
        
    def update(self, frame):
        """Update function for animation"""
        self.frame_count = frame
        self.current_time = frame * self.system.dt
        
        # Check for scenario changes
        if self.scenario_index < len(self.scenarios):
            scenario_time, scenario_type, message = self.scenarios[self.scenario_index]
            if self.current_time >= scenario_time:
                # Process the scenario
                print(f"Time {self.current_time:.1f}s: {message}")
                self.process_scenario(scenario_type, message)
                self.scenario_index += 1
                self.last_scenario_time = self.current_time
        
        # Natural time passage between scenarios
        else:
            self.system.simulate_time_passage(self.system.dt, rest=False)
        
        # Get current state
        response = self.system.process_message("")
        
        # Store data
        self.time_data.append(self.current_time)
        for hormone in self.hormone_data:
            self.hormone_data[hormone].append(response['hormones'][hormone])
            self.baseline_data[hormone].append(response['baselines'][hormone])
        
        for receptor in self.receptor_data:
            self.receptor_data[receptor].append(response['receptors'][receptor])
        
        self.cost_data.append(response['cost']['total'])
        self.mood_data.append(response['mood'])
        
        # Update plots
        if len(self.time_data) > 1:
            time_array = np.array(self.time_data)
            
            # Update hormone lines
            for hormone in self.hormone_data:
                hormone_array = np.array(self.hormone_data[hormone])
                self.hormone_lines[hormone].set_data(time_array, hormone_array)
                
                # Update baseline lines
                if hormone in self.baseline_lines:
                    baseline_array = np.array(self.baseline_data[hormone])
                    self.baseline_lines[hormone].set_data(time_array, baseline_array)
            
            # Update wave lines (deviation from baseline)
            for hormone in self.wave_lines:
                hormone_array = np.array(self.hormone_data[hormone])
                baseline_array = np.array(self.baseline_data[hormone])
                wave = hormone_array - baseline_array
                self.wave_lines[hormone].set_data(time_array, wave)
            
            # Update receptor lines
            for receptor in self.receptor_lines:
                receptor_array = np.array(self.receptor_data[receptor])
                self.receptor_lines[receptor].set_data(time_array, receptor_array)
            
            # Update cost line
            cost_array = np.array(self.cost_data)
            self.cost_line.set_data(time_array, cost_array)
            
            # Update phase space
            d_array = np.array(self.hormone_data['dopamine'])
            s_array = np.array(self.hormone_data['serotonin'])
            self.phase_line.set_data(d_array, s_array)
            self.phase_point.set_data([d_array[-1]], [s_array[-1]])
            
            # Update seeking bars
            seeking = response['seeking']
            heights = [
                seeking.get('dopamine_seeking', 0),
                seeking.get('serotonin_seeking', 0),
                seeking.get('oxytocin_seeking', 0),
                seeking.get('endorphin_seeking', 0)
            ]
            for bar, height in zip(self.seeking_bars, heights):
                bar.set_height(height)
            
            # Adjust x-axis limits
            if self.current_time > self.window_size * self.system.dt:
                x_min = self.current_time - self.window_size * self.system.dt
                x_max = self.current_time
            else:
                x_min = 0
                x_max = self.window_size * self.system.dt
            
            self.ax_hormones.set_xlim([x_min, x_max])
            self.ax_waves.set_xlim([x_min, x_max])
            self.ax_cost.set_xlim([x_min, x_max])
            self.ax_receptors.set_xlim([x_min, x_max])
            
            # Auto-scale cost y-axis
            if len(cost_array) > 0:
                self.ax_cost.set_ylim([0, max(cost_array) * 1.1])
        
        # Update state text
        self.state_texts['mood'].set_text(f"Mood: {response['mood']}")
        current_scenario = self.scenarios[min(self.scenario_index, len(self.scenarios)-1)][2] if self.scenarios else "None"
        self.state_texts['scenario'].set_text(f"Scenario: {current_scenario[:30]}")
        self.state_texts['d_val'].set_text(f"D: {response['hormones']['dopamine']:.1f}")
        self.state_texts['s_val'].set_text(f"S: {response['hormones']['serotonin']:.1f}")
        self.state_texts['c_val'].set_text(f"C: {response['hormones']['cortisol']:.1f}")
        self.state_texts['a_val'].set_text(f"A: {response['hormones']['adrenaline']:.1f}")
        self.state_texts['o_val'].set_text(f"O: {response['hormones']['oxytocin']:.1f}")
        self.state_texts['n_val'].set_text(f"N: {response['hormones']['norepinephrine']:.1f}")
        self.state_texts['e_val'].set_text(f"E: {response['hormones']['endorphins']:.1f}")
        self.state_texts['cost'].set_text(f"Cost: {response['cost']['total']:.2f}")
        self.state_texts['time'].set_text(f"Time: {self.current_time:.1f}s")
        
        # Color the mood text based on mood
        mood_colors = {
            'neutral': 'gray',
            'joyful': 'gold',
            'euphoric': 'gold',
            'stressed': 'red',
            'anxious': 'orange',
            'sad': 'blue',
            'depressed': 'darkblue',
            'calm': 'green',
            'focused': 'purple',
            'loved': 'pink',
            'energized': 'orange'
        }
        mood_color = mood_colors.get(response['mood'], 'black')
        self.state_texts['mood'].set_color(mood_color)
        
        return (list(self.hormone_lines.values()) + 
                list(self.baseline_lines.values()) +
                list(self.wave_lines.values()) +
                list(self.receptor_lines.values()) +
                [self.cost_line, self.phase_line, self.phase_point] +
                list(self.seeking_bars) +
                list(self.state_texts.values()))
    
    def process_scenario(self, scenario_type, message):
        """Process different scenario types"""
        if scenario_type == "joy":
            self.system.process_message(message)
            self.system.process_message(message)  # Double for stronger effect
        elif scenario_type == "stress":
            self.system.process_message(message)
            self.system.process_message(message)
        elif scenario_type == "exercise":
            for _ in range(5):  # Multiple to simulate continuous exercise
                self.system.process_message(message)
        elif scenario_type == "relax":
            for _ in range(3):
                self.system.process_message(message)
        elif scenario_type == "social":
            for _ in range(3):
                self.system.process_message(message)
        elif scenario_type == "sad":
            self.system.process_message(message)
            self.system.process_message(message)
        elif scenario_type == "recovery":
            # Just let time pass
            pass
        else:
            self.system.simulate_time_passage(self.system.dt, rest=False)
    
    def animate(self):
        """Start the animation"""
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update, 
            frames=500,  # Total frames
            interval=50,  # 50ms per frame (20 FPS)
            blit=True,
            repeat=True
        )
        
        # Save as video if ffmpeg is available
        try:
            print("Saving animation as neurochemistry_animation.mp4...")
            anim.save('/root/openhermes_backend/neurochemistry_animation.mp4', 
                     writer='ffmpeg', fps=20, bitrate=2000)
            print("✅ Animation saved!")
        except:
            print("⚠️ Could not save video (ffmpeg not available)")
        
        # Show the animation
        plt.show()
        
        return anim

def main():
    print("="*60)
    print("REAL-TIME NEUROCHEMICAL ANIMATION")
    print("="*60)
    print("\nShowing:")
    print("- Hormone levels (solid lines)")
    print("- Adaptive baselines (dashed lines)")
    print("- Wave amplitudes (deviation from baseline)")
    print("- Receptor sensitivity changes")
    print("- Cost function minimization")
    print("- Phase space trajectory")
    print("- Seeking behaviors")
    print("\nScenarios will trigger automatically...")
    
    animator = NeurochemicalAnimator(window_size=150)
    anim = animator.animate()
    
    print("\n✅ Animation complete!")

if __name__ == "__main__":
    main()
