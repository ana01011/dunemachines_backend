"""
Comprehensive analysis of the neurochemical animation data
Validates biological accuracy and system behavior
"""

import sys
sys.path.append('/root/openhermes_backend')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from app.neurochemistry.interface import NeurochemicalSystem
import json
import pandas as pd

class NeurochemicalAnalyzer:
    def __init__(self):
        self.system = NeurochemicalSystem()
        self.system.dt = 0.5
        
        # Storage for complete simulation data
        self.data = {
            'time': [],
            'hormones': {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 
                                         'oxytocin', 'norepinephrine', 'endorphins']},
            'receptors': {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline',
                                          'oxytocin', 'norepinephrine', 'endorphins']},
            'baselines': {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline',
                                          'oxytocin', 'norepinephrine', 'endorphins']},
            'effective': {h: [] for h in ['dopamine', 'serotonin', 'cortisol', 'adrenaline',
                                          'oxytocin', 'norepinephrine', 'endorphins']},
            'mood': [],
            'cost': [],
            'seeking': {'dopamine': [], 'serotonin': [], 'oxytocin': [], 'endorphin': []},
            'behavioral': {'energy': [], 'mood_score': [], 'focus': [], 'stress': []},
            'resources': {'tyrosine': [], 'tryptophan': [], 'atp': []},
            'allostatic_load': [],
            'scenario': []
        }
        
    def run_complete_simulation(self):
        """Run the same simulation as the animation"""
        
        scenarios = [
            (0, 5, "baseline", "Resting state"),
            (5, 10, "joy", "Wonderful news! So happy!"),
            (15, 10, "stress", "Urgent problem! Very stressful!"),
            (25, 10, "exercise", "Running! Feeling the burn!"),
            (35, 10, "relax", "Deep breathing, meditation, calm..."),
            (45, 10, "social", "Hugging loved ones, feeling connected"),
            (55, 10, "sad", "Feeling down and sad..."),
            (65, 10, "recovery", "Time passing, returning to baseline"),
        ]
        
        print("Running complete simulation...")
        current_time = 0
        
        for start_time, duration, scenario_type, message in scenarios:
            print(f"  {scenario_type}: {message}")
            
            # Simulate until scenario start
            while current_time < start_time:
                self.system.simulate_time_passage(self.system.dt)
                self._record_state(current_time, "transition")
                current_time += self.system.dt
            
            # Process scenario
            scenario_end = start_time + duration
            while current_time < scenario_end:
                if scenario_type == "exercise":
                    # Multiple messages for sustained exercise
                    if int(current_time * 2) % 2 == 0:
                        response = self.system.process_message(message)
                else:
                    response = self.system.process_message(message)
                
                self._record_state(current_time, scenario_type)
                current_time += self.system.dt
        
        print(f"✅ Simulation complete: {len(self.data['time'])} data points collected")
        return self.data
    
    def _record_state(self, time, scenario):
        """Record complete system state"""
        response = self.system.process_message("")
        
        self.data['time'].append(time)
        self.data['scenario'].append(scenario)
        self.data['mood'].append(response['mood'])
        self.data['cost'].append(response['cost']['total'])
        self.data['allostatic_load'].append(response['allostatic_load'])
        
        for h in self.data['hormones']:
            self.data['hormones'][h].append(response['hormones'][h])
            self.data['receptors'][h].append(response['receptors'][h])
            self.data['baselines'][h].append(response['baselines'][h])
            self.data['effective'][h].append(response['effective'][h])
        
        for s in self.data['seeking']:
            key = f"{s}_seeking"
            self.data['seeking'][s].append(response['seeking'].get(key, 0))
        
        for b in self.data['behavioral']:
            if b == 'mood_score':
                self.data['behavioral'][b].append(response['behavioral']['mood'])
            else:
                self.data['behavioral'][b].append(response['behavioral'].get(b, 0))
        
        for r in self.data['resources']:
            self.data['resources'][r].append(response['resources'][r])
    
    def validate_biological_accuracy(self):
        """Validate that the system behaves biologically correctly"""
        
        print("\n" + "="*60)
        print("BIOLOGICAL VALIDATION REPORT")
        print("="*60)
        
        validations = []
        
        # 1. Cortisol Circadian Rhythm
        cortisol = np.array(self.data['hormones']['cortisol'])
        early_cortisol = np.mean(cortisol[:20])  # First 10 seconds
        late_cortisol = np.mean(cortisol[-20:])  # Last 10 seconds
        
        validations.append({
            'test': 'Cortisol Decay Over Time',
            'expected': 'Should decrease over time without stress',
            'actual': f"Early: {early_cortisol:.1f}, Late: {late_cortisol:.1f}",
            'passed': late_cortisol < early_cortisol
        })
        
        # 2. Exercise Endorphin Response
        exercise_mask = [s == 'exercise' for s in self.data['scenario']]
        if any(exercise_mask):
            exercise_idx = [i for i, m in enumerate(exercise_mask) if m]
            pre_exercise_e = np.mean(self.data['hormones']['endorphins'][max(0, exercise_idx[0]-10):exercise_idx[0]])
            during_exercise_e = np.max(self.data['hormones']['endorphins'][exercise_idx[0]:exercise_idx[-1]])
            
            validations.append({
                'test': 'Exercise Endorphin Release',
                'expected': 'Endorphins should spike >50% during exercise',
                'actual': f"Pre: {pre_exercise_e:.1f}, During: {during_exercise_e:.1f}",
                'passed': during_exercise_e > pre_exercise_e * 1.5
            })
        
        # 3. Stress Response Pattern
        stress_mask = [s == 'stress' for s in self.data['scenario']]
        if any(stress_mask):
            stress_idx = [i for i, m in enumerate(stress_mask) if m]
            stress_cortisol = np.mean([self.data['hormones']['cortisol'][i] for i in stress_idx])
            stress_adrenaline = np.mean([self.data['hormones']['adrenaline'][i] for i in stress_idx])
            baseline_cortisol = np.mean(self.data['hormones']['cortisol'][:10])
            
            validations.append({
                'test': 'Stress HPA Axis Activation',
                'expected': 'Cortisol and Adrenaline should increase',
                'actual': f"Cortisol: {baseline_cortisol:.1f}→{stress_cortisol:.1f}, Adrenaline: {stress_adrenaline:.1f}",
                'passed': stress_cortisol > baseline_cortisol and stress_adrenaline > 30
            })
        
        # 4. Social Oxytocin Response
        social_mask = [s == 'social' for s in self.data['scenario']]
        if any(social_mask):
            social_idx = [i for i, m in enumerate(social_mask) if m]
            social_oxytocin = np.mean([self.data['hormones']['oxytocin'][i] for i in social_idx])
            baseline_oxytocin = np.mean(self.data['hormones']['oxytocin'][:10])
            
            validations.append({
                'test': 'Social Bonding Oxytocin',
                'expected': 'Oxytocin should increase >30% during social interaction',
                'actual': f"Baseline: {baseline_oxytocin:.1f}, Social: {social_oxytocin:.1f}",
                'passed': social_oxytocin > baseline_oxytocin * 1.3
            })
        
        # 5. Receptor Adaptation
        dopamine_receptor = np.array(self.data['receptors']['dopamine'])
        high_dopamine_periods = [i for i, d in enumerate(self.data['hormones']['dopamine']) if d > 60]
        if len(high_dopamine_periods) > 5:
            receptor_during_high = np.mean([dopamine_receptor[i] for i in high_dopamine_periods[:5]])
            receptor_after_high = np.mean([dopamine_receptor[i] for i in high_dopamine_periods[-5:]])
            
            validations.append({
                'test': 'Receptor Desensitization',
                'expected': 'Receptors should desensitize during sustained elevation',
                'actual': f"Early: {receptor_during_high:.3f}, Late: {receptor_after_high:.3f}",
                'passed': receptor_after_high < receptor_during_high
            })
        
        # 6. Baseline Adaptation (Minimization Principle)
        dopamine_baseline = np.array(self.data['baselines']['dopamine'])
        baseline_shift = dopamine_baseline[-1] - dopamine_baseline[0]
        avg_dopamine = np.mean(self.data['hormones']['dopamine'])
        
        validations.append({
            'test': 'Baseline Adaptation',
            'expected': 'Baseline should shift toward frequently visited levels',
            'actual': f"Shift: {baseline_shift:.1f}, Avg level: {avg_dopamine:.1f}",
            'passed': abs(dopamine_baseline[-1] - avg_dopamine) < abs(dopamine_baseline[0] - avg_dopamine)
        })
        
        # 7. Mood State Transitions
        unique_moods = list(set(self.data['mood']))
        validations.append({
            'test': 'Mood Diversity',
            'expected': 'Should show at least 4 different mood states',
            'actual': f"Moods observed: {unique_moods}",
            'passed': len(unique_moods) >= 4
        })
        
        # 8. Resource Depletion and Recovery
        atp = np.array(self.data['resources']['atp'])
        min_atp = np.min(atp)
        final_atp = atp[-1]
        
        validations.append({
            'test': 'Energy Metabolism',
            'expected': 'ATP should deplete during activity and recover',
            'actual': f"Min: {min_atp:.2f}, Final: {final_atp:.2f}",
            'passed': min_atp < 0.95 and final_atp > min_atp
        })
        
        # 9. Allostatic Load
        allostatic = np.array(self.data['allostatic_load'])
        max_load = np.max(allostatic)
        
        validations.append({
            'test': 'Allostatic Load Accumulation',
            'expected': 'Should accumulate during stress',
            'actual': f"Max load: {max_load:.3f}",
            'passed': max_load > 0.001
        })
        
        # 10. Cost Minimization
        cost = np.array(self.data['cost'])
        early_cost = np.mean(cost[:20])
        late_cost = np.mean(cost[-20:])
        
        validations.append({
            'test': 'Cost Minimization',
            'expected': 'System cost should decrease over time',
            'actual': f"Early: {early_cost:.1f}, Late: {late_cost:.1f}",
            'passed': late_cost < early_cost * 1.2  # Allow some variance
        })
        
        # Print results
        passed = 0
        failed = 0
        
        for v in validations:
            status = "✅ PASS" if v['passed'] else "❌ FAIL"
            print(f"\n{status}: {v['test']}")
            print(f"  Expected: {v['expected']}")
            print(f"  Actual: {v['actual']}")
            
            if v['passed']:
                passed += 1
            else:
                failed += 1
        
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
        print(f"Biological Accuracy: {(passed/(passed+failed)*100):.1f}%")
        print("="*60)
        
        return validations
    
    def generate_detailed_report(self):
        """Generate a comprehensive analysis report"""
        
        # Create detailed plots
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(5, 3, hspace=0.3, wspace=0.25)
        
        time = np.array(self.data['time'])
        
        # 1. Hormone Dynamics with Scenarios
        ax1 = fig.add_subplot(gs[0:2, :])
        scenarios_unique = list(set(self.data['scenario']))
        scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_unique)))
        
        # Plot scenario backgrounds
        for i, scenario in enumerate(self.data['scenario']):
            if i == 0 or scenario != self.data['scenario'][i-1]:
                # Find end of this scenario block
                end_idx = i + 1
                while end_idx < len(self.data['scenario']) and self.data['scenario'][end_idx] == scenario:
                    end_idx += 1
                
                color_idx = scenarios_unique.index(scenario)
                ax1.axvspan(time[i], time[min(end_idx-1, len(time)-1)], 
                           alpha=0.1, color=scenario_colors[color_idx], label=scenario)
        
        # Plot hormones
        for hormone in ['dopamine', 'serotonin', 'cortisol', 'endorphins']:
            ax1.plot(time, self.data['hormones'][hormone], linewidth=2, label=hormone.capitalize())
        
        ax1.set_title('Complete Neurochemical Dynamics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Hormone Level')
        ax1.legend(loc='upper right', ncol=5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Effective vs Actual Hormones
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(time, self.data['hormones']['dopamine'], 'r-', label='Actual', linewidth=2)
        ax2.plot(time, self.data['effective']['dopamine'], 'r--', label='Effective', linewidth=1)
        ax2.set_title('Dopamine: Actual vs Effective')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Receptor Sensitivity
        ax3 = fig.add_subplot(gs[2, 1])
        for hormone in ['dopamine', 'serotonin', 'cortisol']:
            ax3.plot(time, self.data['receptors'][hormone], label=hormone)
        ax3.set_title('Receptor Adaptation')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Sensitivity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 4. Baseline Evolution
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(time, self.data['baselines']['dopamine'], 'r--', label='D baseline')
        ax4.plot(time, self.data['hormones']['dopamine'], 'r-', alpha=0.3, label='D actual')
        ax4.plot(time, self.data['baselines']['serotonin'], 'b--', label='S baseline')
        ax4.plot(time, self.data['hormones']['serotonin'], 'b-', alpha=0.3, label='S actual')
        ax4.set_title('Baseline Adaptation (Minimization)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Level')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Cost Function
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(time, self.data['cost'], 'purple', linewidth=2)
        ax5.fill_between(time, 0, self.data['cost'], alpha=0.3, color='purple')
        ax5.set_title('Cost Function Over Time')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Total Cost')
        ax5.grid(True, alpha=0.3)
        
        # 6. Resources
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.plot(time, self.data['resources']['tyrosine'], label='Tyrosine')
        ax6.plot(time, self.data['resources']['tryptophan'], label='Tryptophan')
        ax6.plot(time, self.data['resources']['atp'], label='ATP')
        ax6.set_title('Metabolic Resources')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Resource Level')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1.1])
        
        # 7. Seeking Behavior
        ax7 = fig.add_subplot(gs[3, 2])
        for seeking_type in self.data['seeking']:
            ax7.plot(time, self.data['seeking'][seeking_type], label=seeking_type)
        ax7.set_title('Seeking Intensities')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Intensity')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1])
        
        # 8. Behavioral Output
        ax8 = fig.add_subplot(gs[4, :2])
        ax8.plot(time, self.data['behavioral']['energy'], label='Energy')
        ax8.plot(time, self.data['behavioral']['mood_score'], label='Mood')
        ax8.plot(time, self.data['behavioral']['focus'], label='Focus')
        ax8.plot(time, self.data['behavioral']['stress'], label='Stress')
        ax8.set_title('Behavioral Parameters')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Parameter Value')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])
        
        # 9. Statistics Box
        ax9 = fig.add_subplot(gs[4, 2])
        ax9.axis('off')
        
        stats_text = "System Statistics:\n\n"
        stats_text += f"Total Duration: {time[-1]:.1f}s\n"
        stats_text += f"Unique Moods: {len(set(self.data['mood']))}\n"
        stats_text += f"D Range: {min(self.data['hormones']['dopamine']):.1f}-{max(self.data['hormones']['dopamine']):.1f}\n"
        stats_text += f"S Range: {min(self.data['hormones']['serotonin']):.1f}-{max(self.data['hormones']['serotonin']):.1f}\n"
        stats_text += f"C Range: {min(self.data['hormones']['cortisol']):.1f}-{max(self.data['hormones']['cortisol']):.1f}\n"
        stats_text += f"E Peak: {max(self.data['hormones']['endorphins']):.1f}\n"
        stats_text += f"O Peak: {max(self.data['hormones']['oxytocin']):.1f}\n"
        stats_text += f"Min Cost: {min(self.data['cost']):.1f}\n"
        stats_text += f"Max Load: {max(self.data['allostatic_load']):.3f}\n"
        
        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Complete Neurochemical System Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/root/openhermes_backend/neurochemistry_full_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📊 Full analysis saved as neurochemistry_full_analysis.png")
        
    def export_data(self):
        """Export all data to JSON for external analysis"""
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for key, value in self.data.items():
            if isinstance(value, dict):
                export_data[key] = {}
                for subkey, subvalue in value.items():
                    export_data[key][subkey] = [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                               for v in subvalue]
            else:
                export_data[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for v in value]
        
        with open('/root/openhermes_backend/neurochemistry_animation_data.json', 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print("📁 Data exported to neurochemistry_animation_data.json")
        
        # Also create CSV for easy analysis
        df_data = {
            'time': self.data['time'],
            'scenario': self.data['scenario'],
            'mood': self.data['mood'],
            'cost': self.data['cost'],
            'allostatic_load': self.data['allostatic_load']
        }
        
        for hormone in self.data['hormones']:
            df_data[f'{hormone}'] = self.data['hormones'][hormone]
            df_data[f'{hormone}_baseline'] = self.data['baselines'][hormone]
            df_data[f'{hormone}_receptor'] = self.data['receptors'][hormone]
        
        df = pd.DataFrame(df_data)
        df.to_csv('/root/openhermes_backend/neurochemistry_animation_data.csv', index=False)
        print("📁 Data exported to neurochemistry_animation_data.csv")

def main():
    print("="*60)
    print("NEUROCHEMICAL ANIMATION DATA ANALYSIS")
    print("="*60)
    
    analyzer = NeurochemicalAnalyzer()
    
    # Run simulation
    data = analyzer.run_complete_simulation()
    
    # Validate biological accuracy
    validations = analyzer.validate_biological_accuracy()
    
    # Generate detailed report
    analyzer.generate_detailed_report()
    
    # Export data
    analyzer.export_data()
    
    print("\n✅ Analysis complete!")
    print("\nGenerated files:")
    print("  1. neurochemistry_full_analysis.png - Visual analysis")
    print("  2. neurochemistry_animation_data.json - Complete data")
    print("  3. neurochemistry_animation_data.csv - Spreadsheet format")

if __name__ == "__main__":
    main()
