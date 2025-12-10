"""
Comprehensive animated visualization of the 7D neurochemical system
Shows baseline shifts, hormone waves, and cost minimization principle
"""

import sys
sys.path.append('/root/openhermes_backend')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from collections import deque

from app.neurochemistry.interface import NeurochemicalSystem

class ComprehensiveNeurochemAnimation:
    def __init__(self):
        """Initialize the comprehensive animation system"""
        self.system = NeurochemicalSystem()
        self.system.dt = 0.5  # 500ms timesteps
        
        # Data storage with longer history
        self.window_size = 200
        self.time_data = deque(maxlen=self.window_size)
        
        # Hormones
        self.hormones = {h: deque(maxlen=self.window_size) for h in 
                        ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                         'oxytocin', 'norepinephrine', 'endorphins']}
        
        # Baselines (showing adaptation)
        self.baselines = {h: deque(maxlen=self.window_size) for h in 
                         ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                          'oxytocin', 'norepinephrine', 'endorphins']}
        
        # Wave amplitudes (hormone - baseline)
        self.waves = {h: deque(maxlen=self.window_size) for h in 
                     ['dopamine', 'serotonin', 'cortisol']}
        
        # Cost components
        self.costs = {
            'total': deque(maxlen=self.window_size),
            'deviation': deque(maxlen=self.window_size),
            'metabolic': deque(maxlen=self.window_size),
            'uncertainty': deque(maxlen=self.window_size)
        }
        
        # Other metrics
        self.receptors = {h: deque(maxlen=self.window_size) for h in 
                         ['dopamine', 'serotonin', 'cortisol']}
        self.seeking = deque(maxlen=self.window_size)
        self.efficiency = deque(maxlen=self.window_size)
        self.mood_history = deque(maxlen=self.window_size)
        self.allostatic_load = deque(maxlen=self.window_size)
        
        # Scenario timeline with diverse stimuli
        self.scenarios = [
            # Time, Duration, Type, Message, Color
            (0, 5, "baseline", "Resting baseline state", "gray"),
            (5, 8, "joy", "Wonderful news! Dopamine spike!", "gold"),
            (13, 8, "stress", "Urgent stress! Cortisol rising!", "red"),
            (21, 10, "exercise", "Running! Endorphins releasing!", "orange"),
            (31, 8, "social", "Bonding time! Oxytocin boost!", "pink"),
            (39, 8, "sadness", "Feeling sad... Serotonin drop", "darkblue"),
            (47, 10, "relaxation", "Deep meditation... Calming down", "green"),
            (57, 8, "focus", "Deep work! Norepinephrine up!", "purple"),
            (65, 10, "recovery", "Returning to baseline...", "lightgray"),
        ]
        
        self.current_scenario_idx = 0
        self.scenario_start_time = 0
        self.current_time = 0
        self.frame_count = 0
        
        # Setup the figure
        self.setup_figure()
        
    def setup_figure(self):
        """Create the comprehensive figure layout"""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Main title
        self.fig.suptitle('7D Neurochemical Dynamics - Baseline Adaptation & Cost Minimization', 
                          fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.94, bottom=0.06)
        
        # 1. Main hormone plot with baselines (top, spans 3 columns)
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:3])
        self.setup_main_plot()
        
        # 2. Cost minimization plot (top right)
        self.ax_cost = self.fig.add_subplot(gs[0, 3])
        self.setup_cost_plot()
        
        # 3. Efficiency meter (middle right)
        self.ax_efficiency = self.fig.add_subplot(gs[1, 3])
        self.setup_efficiency_plot()
        
        # 4. Wave visualization (bottom left)
        self.ax_waves = self.fig.add_subplot(gs[2, 0:2])
        self.setup_waves_plot()
        
        # 5. Phase space (bottom middle)
        self.ax_phase = self.fig.add_subplot(gs[2, 2])
        self.setup_phase_plot()
        
        # 6. Receptor adaptation (bottom right)
        self.ax_receptors = self.fig.add_subplot(gs[2, 3])
        self.setup_receptors_plot()
        
        # 7. Baseline shift visualization (very bottom left)
        self.ax_baseline_shift = self.fig.add_subplot(gs[3, 0])
        self.setup_baseline_shift_plot()
        
        # 8. Seeking behavior (very bottom middle-left)
        self.ax_seeking = self.fig.add_subplot(gs[3, 1])
        self.setup_seeking_plot()
        
        # 9. Mood indicator (very bottom middle-right)
        self.ax_mood = self.fig.add_subplot(gs[3, 2])
        self.setup_mood_plot()
        
        # 10. Current state display (very bottom right)
        self.ax_state = self.fig.add_subplot(gs[3, 3])
        self.setup_state_display()
        
    def setup_main_plot(self):
        """Setup main hormone and baseline plot"""
        self.ax_main.set_title('Hormone Levels & Adaptive Baselines', fontsize=12, fontweight='bold')
        self.ax_main.set_xlabel('Time (seconds)')
        self.ax_main.set_ylabel('Level')
        self.ax_main.set_ylim([0, 100])
        self.ax_main.grid(True, alpha=0.2)
        
        # Hormone colors
        self.colors = {
            'dopamine': '#FF6B6B',
            'serotonin': '#4ECDC4',
            'cortisol': '#FFE66D',
            'adrenaline': '#FF8C42',
            'oxytocin': '#95E1D3',
            'norepinephrine': '#C77DFF',
            'endorphins': '#FFB6C1'
        }
        
        # Create lines for hormones (solid) and baselines (dashed)
        self.hormone_lines = {}
        self.baseline_lines = {}
        
        for hormone, color in self.colors.items():
            # Hormone line
            line, = self.ax_main.plot([], [], color=color, linewidth=2, 
                                      label=hormone[:3].upper(), alpha=0.8)
            self.hormone_lines[hormone] = line
            
            # Baseline line (only for key hormones)
            if hormone in ['dopamine', 'serotonin', 'cortisol']:
                baseline, = self.ax_main.plot([], [], color=color, linewidth=1.5,
                                              linestyle='--', alpha=0.5)
                self.baseline_lines[hormone] = baseline
        
        self.ax_main.legend(loc='upper left', ncol=7, fontsize=8, framealpha=0.9)
        
    def setup_cost_plot(self):
        """Setup cost minimization plot"""
        self.ax_cost.set_title('Cost Minimization', fontsize=10, fontweight='bold')
        self.ax_cost.set_xlabel('Time (s)', fontsize=8)
        self.ax_cost.set_ylabel('Cost', fontsize=8)
        self.ax_cost.grid(True, alpha=0.2)
        
        self.cost_line_total, = self.ax_cost.plot([], [], 'purple', linewidth=2, label='Total')
        self.cost_line_deviation, = self.ax_cost.plot([], [], 'blue', linewidth=1, 
                                                       alpha=0.5, label='Deviation')
        self.cost_line_metabolic, = self.ax_cost.plot([], [], 'red', linewidth=1, 
                                                      alpha=0.5, label='Metabolic')
        
        self.ax_cost.legend(loc='upper right', fontsize=6)
        
    def setup_efficiency_plot(self):
        """Setup efficiency meter"""
        self.ax_efficiency.set_title('System Efficiency', fontsize=10, fontweight='bold')
        self.ax_efficiency.set_xlim([0, 1])
        self.ax_efficiency.set_ylim([0, 1])
        self.ax_efficiency.axis('off')
        
        # Create efficiency gauge
        self.efficiency_bar = Rectangle((0.2, 0.2), 0.6, 0.0, 
                                       facecolor='green', alpha=0.7)
        self.ax_efficiency.add_patch(self.efficiency_bar)
        
        # Add scale
        for i in range(11):
            y = 0.2 + i * 0.06
            self.ax_efficiency.plot([0.15, 0.18], [y, y], 'k-', linewidth=0.5)
            if i % 5 == 0:
                self.ax_efficiency.text(0.1, y, f'{i*10}', fontsize=6, ha='right')
        
        self.efficiency_text = self.ax_efficiency.text(0.5, 0.9, 'Efficiency: 0%', 
                                                       fontsize=10, ha='center')
        
    def setup_waves_plot(self):
        """Setup wave amplitude visualization"""
        self.ax_waves.set_title('Hormone Waves (Deviation from Baseline)', fontsize=10, fontweight='bold')
        self.ax_waves.set_xlabel('Time (s)', fontsize=8)
        self.ax_waves.set_ylabel('Wave Amplitude', fontsize=8)
        self.ax_waves.set_ylim([-30, 30])
        self.ax_waves.axhline(y=0, color='black', linewidth=1, alpha=0.3)
        self.ax_waves.grid(True, alpha=0.2)
        
        self.wave_lines = {}
        wave_colors = ['red', 'blue', 'orange']
        for i, hormone in enumerate(['dopamine', 'serotonin', 'cortisol']):
            line, = self.ax_waves.plot([], [], color=wave_colors[i], 
                                       linewidth=1.5, label=f'{hormone[:3]} wave')
            self.wave_lines[hormone] = line
        
        self.ax_waves.legend(loc='upper right', fontsize=8)
        
    def setup_phase_plot(self):
        """Setup phase space plot"""
        self.ax_phase.set_title('D-S Phase Space', fontsize=10, fontweight='bold')
        self.ax_phase.set_xlabel('Dopamine', fontsize=8)
        self.ax_phase.set_ylabel('Serotonin', fontsize=8)
        self.ax_phase.set_xlim([30, 80])
        self.ax_phase.set_ylim([20, 70])
        self.ax_phase.grid(True, alpha=0.2)
        
        # Trajectory line
        self.phase_line, = self.ax_phase.plot([], [], 'b-', linewidth=1, alpha=0.5)
        # Current position
        self.phase_point, = self.ax_phase.plot([], [], 'ro', markersize=8)
        
        # Add mood regions
        self.ax_phase.text(60, 60, 'Happy', fontsize=8, alpha=0.3, ha='center')
        self.ax_phase.text(40, 30, 'Sad', fontsize=8, alpha=0.3, ha='center')
        self.ax_phase.text(70, 40, 'Stressed', fontsize=8, alpha=0.3, ha='center')
        
    def setup_receptors_plot(self):
        """Setup receptor sensitivity plot"""
        self.ax_receptors.set_title('Receptor Adaptation', fontsize=10, fontweight='bold')
        self.ax_receptors.set_xlabel('Time (s)', fontsize=8)
        self.ax_receptors.set_ylabel('Sensitivity', fontsize=8)
        self.ax_receptors.set_ylim([0, 1])
        self.ax_receptors.grid(True, alpha=0.2)
        
        self.receptor_lines = {}
        colors = ['red', 'blue', 'orange']
        for i, hormone in enumerate(['dopamine', 'serotonin', 'cortisol']):
            line, = self.ax_receptors.plot([], [], color=colors[i], 
                                          linewidth=1.5, label=f'R_{hormone[:1]}')
            self.receptor_lines[hormone] = line
        
        self.ax_receptors.legend(loc='upper right', fontsize=8)
        
    def setup_baseline_shift_plot(self):
        """Setup baseline shift visualization"""
        self.ax_baseline_shift.set_title('Baseline Shifts', fontsize=9, fontweight='bold')
        self.ax_baseline_shift.set_ylim([-5, 5])
        self.ax_baseline_shift.set_ylabel('Shift', fontsize=8)
        self.ax_baseline_shift.set_xticks([0, 1, 2])
        self.ax_baseline_shift.set_xticklabels(['D', 'S', 'C'], fontsize=8)
        
        self.baseline_bars = self.ax_baseline_shift.bar([0, 1, 2], [0, 0, 0], 
                                                        color=['red', 'blue', 'orange'])
        
    def setup_seeking_plot(self):
        """Setup seeking behavior plot"""
        self.ax_seeking.set_title('Seeking Intensity', fontsize=9, fontweight='bold')
        self.ax_seeking.set_ylim([0, 1])
        self.ax_seeking.set_ylabel('Intensity', fontsize=8)
        self.ax_seeking.set_xticks([0, 1, 2])
        self.ax_seeking.set_xticklabels(['Dopa', 'Sero', 'Oxy'], fontsize=8)
        
        self.seeking_bars = self.ax_seeking.bar([0, 1, 2], [0, 0, 0], 
                                                color=['red', 'blue', 'pink'])
        
    def setup_mood_plot(self):
        """Setup mood indicator"""
        self.ax_mood.set_title('Current Mood', fontsize=9, fontweight='bold')
        self.ax_mood.set_xlim([0, 1])
        self.ax_mood.set_ylim([0, 1])
        self.ax_mood.axis('off')
        
        # Mood circle indicator
        self.mood_circle = Circle((0.5, 0.5), 0.3, facecolor='gray', alpha=0.5)
        self.ax_mood.add_patch(self.mood_circle)
        
        self.mood_text = self.ax_mood.text(0.5, 0.5, 'Neutral', fontsize=12,
                                           ha='center', va='center', fontweight='bold')
        
    def setup_state_display(self):
        """Setup current state display"""
        self.ax_state.set_title('System State', fontsize=9, fontweight='bold')
        self.ax_state.set_xlim([0, 1])
        self.ax_state.set_ylim([0, 1])
        self.ax_state.axis('off')
        
        # State text elements
        self.state_texts = {
            'scenario': self.ax_state.text(0.05, 0.9, '', fontsize=8),
            'time': self.ax_state.text(0.05, 0.75, '', fontsize=8),
            'cost': self.ax_state.text(0.05, 0.6, '', fontsize=8),
            'load': self.ax_state.text(0.05, 0.45, '', fontsize=8),
            'd_val': self.ax_state.text(0.05, 0.3, '', fontsize=7),
            's_val': self.ax_state.text(0.05, 0.2, '', fontsize=7),
            'c_val': self.ax_state.text(0.05, 0.1, '', fontsize=7),
        }
        
    def update(self, frame):
        """Update animation frame"""
        self.frame_count = frame
        self.current_time = frame * self.system.dt
        
        # Process scenarios
        if self.current_scenario_idx < len(self.scenarios):
            start, duration, stype, message, color = self.scenarios[self.current_scenario_idx]
            
            if self.current_time >= start and self.current_time < start + duration:
                # Process current scenario
                if self.current_time == start:
                    self.scenario_start_time = self.current_time
                    print(f"Time {self.current_time:.1f}s: {message}")
                
                # Apply scenario multiple times for stronger effect
                if stype == "exercise":
                    for _ in range(2):
                        response = self.system.process_message(message)
                elif stype == "stress":
                    response = self.system.process_message(message + " URGENT!")
                else:
                    response = self.system.process_message(message)
            elif self.current_time >= start + duration:
                self.current_scenario_idx += 1
                response = self.system.process_message("")
        else:
            # Natural decay
            self.system.simulate_time_passage(self.system.dt)
            response = self.system.process_message("")
        
        # Store data
        self.time_data.append(self.current_time)
        
        for hormone in self.hormones:
            self.hormones[hormone].append(response['hormones'][hormone])
            self.baselines[hormone].append(response['baselines'][hormone])
            if hormone in self.receptors:
                self.receptors[hormone].append(response['receptors'][hormone])
        
        # Calculate waves
        for hormone in self.waves:
            wave = response['hormones'][hormone] - response['baselines'][hormone]
            self.waves[hormone].append(wave)
        
        # Store costs
        self.costs['total'].append(response['cost']['total'])
        self.costs['deviation'].append(response['cost'].get('deviation', 0))
        self.costs['metabolic'].append(response['cost'].get('metabolic', 0))
        
        # Store other metrics
        self.efficiency.append(response['efficiency'])
        self.seeking.append(response['seeking']['total_seeking'])
        self.mood_history.append(response['mood'])
        self.allostatic_load.append(response['allostatic_load'])
        
        # Update plots
        if len(self.time_data) > 1:
            self.update_plots(response)
        
        # Return all artists for blitting
        artists = []
        artists.extend(self.hormone_lines.values())
        artists.extend(self.baseline_lines.values())
        artists.extend(self.wave_lines.values())
        artists.extend(self.receptor_lines.values())
        artists.extend([self.cost_line_total, self.cost_line_deviation, 
                       self.cost_line_metabolic])
        artists.extend([self.phase_line, self.phase_point])
        artists.extend(self.baseline_bars)
        artists.extend(self.seeking_bars)
        artists.extend([self.mood_circle, self.mood_text])
        artists.extend([self.efficiency_bar, self.efficiency_text])
        artists.extend(self.state_texts.values())
        
        return artists
        
    def update_plots(self, response):
        """Update all plot elements"""
        time_array = np.array(self.time_data)
        
        # Determine x-axis window
        if self.current_time > 30:
            x_min = self.current_time - 30
            x_max = self.current_time
        else:
            x_min = 0
            x_max = 30
        
        # Update main hormone plot
        for hormone in self.hormone_lines:
            hormone_array = np.array(self.hormones[hormone])
            self.hormone_lines[hormone].set_data(time_array, hormone_array)
            
            if hormone in self.baseline_lines:
                baseline_array = np.array(self.baselines[hormone])
                self.baseline_lines[hormone].set_data(time_array, baseline_array)
        
        self.ax_main.set_xlim([x_min, x_max])
        
        # Update wave plot
        for hormone in self.wave_lines:
            wave_array = np.array(self.waves[hormone])
            self.wave_lines[hormone].set_data(time_array, wave_array)
        
        self.ax_waves.set_xlim([x_min, x_max])
        
        # Update cost plot
        cost_total = np.array(self.costs['total'])
        cost_deviation = np.array(self.costs['deviation'])
        cost_metabolic = np.array(self.costs['metabolic'])
        
        self.cost_line_total.set_data(time_array, cost_total)
        self.cost_line_deviation.set_data(time_array, cost_deviation)
        self.cost_line_metabolic.set_data(time_array, cost_metabolic)
        
        self.ax_cost.set_xlim([x_min, x_max])
        if len(cost_total) > 0:
            self.ax_cost.set_ylim([0, max(cost_total[-50:] if len(cost_total) > 50 else cost_total) * 1.1])
        
        # Update phase space
        d_array = np.array(self.hormones['dopamine'])
        s_array = np.array(self.hormones['serotonin'])
        self.phase_line.set_data(d_array, s_array)
        if len(d_array) > 0:
            self.phase_point.set_data([d_array[-1]], [s_array[-1]])
        
        # Update receptor plot
        for hormone in self.receptor_lines:
            receptor_array = np.array(self.receptors[hormone])
            self.receptor_lines[hormone].set_data(time_array, receptor_array)
        
        self.ax_receptors.set_xlim([x_min, x_max])
        
        # Update baseline shift bars
        if len(self.baselines['dopamine']) > 10:
            shifts = [
                self.baselines['dopamine'][-1] - self.baselines['dopamine'][-10],
                self.baselines['serotonin'][-1] - self.baselines['serotonin'][-10],
                self.baselines['cortisol'][-1] - self.baselines['cortisol'][-10]
            ]
            for bar, shift in zip(self.baseline_bars, shifts):
                bar.set_height(shift)
                bar.set_color('green' if shift > 0 else 'red')
        
        # Update seeking bars
        seeking_values = [
            response['seeking']['dopamine_seeking'],
            response['seeking']['serotonin_seeking'],
            response['seeking']['oxytocin_seeking']
        ]
        for bar, value in zip(self.seeking_bars, seeking_values):
            bar.set_height(value)
        
        # Update mood indicator
        mood = response['mood']
        mood_colors = {
            'neutral': 'gray', 'calm': 'lightgreen', 'happy': 'yellow',
            'joyful': 'gold', 'euphoric': 'orange', 'stressed': 'red',
            'anxious': 'darkred', 'sad': 'blue', 'depressed': 'darkblue',
            'focused': 'purple', 'motivated': 'green', 'loved': 'pink',
            'energized': 'orange', 'tired': 'brown', 'alert': 'red',
            'content': 'lightblue', 'balanced': 'green', 'worried': 'orange'
        }
        
        color = mood_colors.get(mood, 'gray')
        self.mood_circle.set_facecolor(color)
        self.mood_text.set_text(mood.capitalize())
        
        # Update efficiency meter
        eff = response['efficiency']
        self.efficiency_bar.set_height(0.6 * min(1, eff / 10))
        self.efficiency_text.set_text(f'Efficiency: {eff:.1f}')
        
        # Update state display
        if self.current_scenario_idx < len(self.scenarios):
            current_scenario = self.scenarios[self.current_scenario_idx][2]
        else:
            current_scenario = "recovery"
        
        self.state_texts['scenario'].set_text(f'Scenario: {current_scenario}')
        self.state_texts['time'].set_text(f'Time: {self.current_time:.1f}s')
        self.state_texts['cost'].set_text(f'Cost: {response["cost"]["total"]:.1f}')
        self.state_texts['load'].set_text(f'Load: {response["allostatic_load"]:.3f}')
        self.state_texts['d_val'].set_text(f'D: {response["hormones"]["dopamine"]:.1f}')
        self.state_texts['s_val'].set_text(f'S: {response["hormones"]["serotonin"]:.1f}')
        self.state_texts['c_val'].set_text(f'C: {response["hormones"]["cortisol"]:.1f}')
        
    def animate(self):
        """Create and save animation"""
        print("\n" + "="*60)
        print("COMPREHENSIVE NEUROCHEMISTRY ANIMATION")
        print("="*60)
        print("\nCreating animation with:")
        print("- Hormone dynamics and baseline adaptation")
        print("- Cost minimization visualization")
        print("- Wave amplitudes showing oscillations")
        print("- Receptor sensitivity changes")
        print("- Phase space trajectory")
        print("- Seeking behavior intensity")
        print("- Mood transitions")
        print("- System efficiency tracking")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update,
            frames=300,  # 150 seconds at 0.5s timestep
            interval=50,  # 50ms per frame = 20 FPS
            blit=True,
            repeat=True
        )
        
        # Save animation
        try:
            print("\nSaving animation as comprehensive_neurochemistry.mp4...")
            anim.save('/root/openhermes_backend/comprehensive_neurochemistry.mp4',
                     writer='ffmpeg', fps=20, bitrate=3000, dpi=100)
            print("✅ Animation saved successfully!")
            
            # Also save as GIF for easier viewing
            print("Creating GIF version...")
            anim.save('/root/openhermes_backend/comprehensive_neurochemistry.gif',
                     writer='pillow', fps=10, dpi=80)
            print("✅ GIF version saved!")
            
        except Exception as e:
            print(f"⚠️ Could not save video: {e}")
            print("Saving as image sequence instead...")
            
            # Save key frames as images
            for i in range(0, 300, 30):
                self.update(i)
                plt.savefig(f'/root/openhermes_backend/frame_{i:03d}.png', dpi=100)
            print("✅ Saved as image sequence")
        
        plt.show()
        return anim

def main():
    """Run the comprehensive animation"""
    animator = ComprehensiveNeurochemAnimation()
    anim = animator.animate()
    
    print("\n" + "="*60)
    print("ANIMATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. comprehensive_neurochemistry.mp4 - Full video")
    print("  2. comprehensive_neurochemistry.gif - GIF version")
    print("\nTo download to Windows:")
    print("  scp root@srv949439:/root/openhermes_backend/comprehensive_neurochemistry.mp4 ./")
    print("\nOr view via browser:")
    print("  python -m http.server 8080")
    print("  Then go to: http://YOUR_IP:8080/comprehensive_neurochemistry.mp4")

if __name__ == "__main__":
    main()
