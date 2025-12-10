"""
Enhanced Neurochemical State with Opponent Processes and Spike Tracking
"""

import numpy as np
from typing import Dict, List, Tuple
import time

# Import constants - try multiple paths
try:
    from .constants import *
except ImportError:
    try:
        from constants import *
    except ImportError:
        # If constants aren't found, define minimal required ones
        HILL_COEFFICIENTS = {
            'dopamine': 2.5, 'serotonin': 2.0, 'cortisol': 3.0,
            'adrenaline': 4.0, 'oxytocin': 2.0, 'norepinephrine': 3.0,
            'endorphins': 2.5
        }
        HALF_SATURATION = {
            'dopamine': 40.0, 'serotonin': 50.0, 'cortisol': 45.0,
            'adrenaline': 35.0, 'oxytocin': 30.0, 'norepinephrine': 40.0,
            'endorphins': 25.0
        }
        COST_WEIGHTS = {
            'production': 10.0, 'maintenance': 2.0, 'change': 1.0,
            'metabolic': 0.5, 'allostatic': 1.0
        }

class NeurochemicalState:
    """Complete 7D neurochemical state with opponent processes"""
    
    def __init__(self):
        # Primary hormone levels (0-100)
        self.hormones = {
            'dopamine': 50.0,
            'serotonin': 60.0,
            'cortisol': 30.0,
            'adrenaline': 20.0,
            'oxytocin': 40.0,
            'norepinephrine': 45.0,
            'endorphins': 35.0
        }
        
        # Receptor sensitivities (0-1)
        self.receptors = {h: 0.8 for h in self.hormones.keys()}
        
        # Adaptive baselines
        self.baselines = {
            'fast': self.hormones.copy(),
            'slow': self.hormones.copy()
        }
        
        # Expected states for prediction error
        self.expected = self.hormones.copy()
        
        # Metabolic resources
        self.resources = {
            'tyrosine': 1.0,
            'tryptophan': 1.0,
            'atp': 1.0
        }
        
        # Allostatic load
        self.allostatic_load = 0.0
        
        # Spike tracking for minimization principle
        self.spike_history = {h: [] for h in self.hormones.keys()}
        self.spike_window = 30.0  # seconds
        
        # Opponent process ratios
        self.opponent_ratios = {}
        
        # Production tracking
        self.production_history = {h: [] for h in self.hormones.keys()}
        
        # History for analysis
        self.history = []
        self.time = 0.0
        self.last_update = time.time()
    
    def calculate_opponent_ratios(self) -> Dict[str, float]:
        """Calculate opponent process ratios for each hormone"""
        h = self.hormones
        
        # Dopamine vs Serotonin (reward vs contentment)
        self.opponent_ratios['dopamine'] = (
            (h['dopamine'] + 0.5 * h['norepinephrine']) / 
            (h['serotonin'] + 0.5 * h['oxytocin'] + 1.0)
        )
        
        # Serotonin vs Dopamine (contentment vs reward seeking)
        self.opponent_ratios['serotonin'] = (
            (h['serotonin'] + 0.5 * h['oxytocin']) /
            (h['dopamine'] + 0.5 * h['norepinephrine'] + 1.0)
        )
        
        # Cortisol vs Oxytocin (stress vs bonding)
        self.opponent_ratios['cortisol'] = (
            h['cortisol'] /
            (h['oxytocin'] + 0.3 * h['serotonin'] + 1.0)
        )
        
        # Adrenaline vs Endorphins (arousal vs relaxation)
        self.opponent_ratios['adrenaline'] = (
            (h['adrenaline'] + 0.3 * h['norepinephrine']) /
            (h['endorphins'] + 0.3 * h['oxytocin'] + 1.0)
        )
        
        # Oxytocin vs Cortisol (bonding vs stress)
        self.opponent_ratios['oxytocin'] = (
            (h['oxytocin'] + 0.3 * h['serotonin']) /
            (h['cortisol'] + 0.3 * h['adrenaline'] + 1.0)
        )
        
        # Norepinephrine (focus) - balanced by serotonin
        self.opponent_ratios['norepinephrine'] = (
            h['norepinephrine'] /
            (h['serotonin'] + 0.5 * h['endorphins'] + 1.0)
        )
        
        # Endorphins vs Adrenaline (relaxation vs arousal)
        self.opponent_ratios['endorphins'] = (
            (h['endorphins'] + 0.3 * h['oxytocin']) /
            (h['adrenaline'] + 0.3 * h['cortisol'] + 1.0)
        )
        
        return self.opponent_ratios
    
    def track_spike(self, hormone: str):
        """Track hormone spikes for baseline adaptation"""
        current_level = self.hormones[hormone]
        baseline = self.baselines['slow'][hormone]
        
        # Detect spike (50% above baseline)
        if current_level > baseline * 1.5:
            spike_amplitude = current_level - baseline
            self.spike_history[hormone].append((self.time, spike_amplitude))
            
            # Remove old spikes outside window
            self.spike_history[hormone] = [
                (t, a) for t, a in self.spike_history[hormone]
                if self.time - t < self.spike_window
            ]
    
    def get_average_spike(self, hormone: str) -> float:
        """Get average spike amplitude for a hormone"""
        if not self.spike_history[hormone]:
            return 0.0
        
        recent_spikes = [
            amplitude for time, amplitude in self.spike_history[hormone]
            if self.time - time < self.spike_window
        ]
        
        return np.mean(recent_spikes) if recent_spikes else 0.0
    
    def calculate_production_cost(self, hormone: str) -> float:
        """Calculate production cost from baseline"""
        baseline = self.baselines['slow'][hormone]
        current = self.hormones[hormone]
        
        # Production is what's above baseline
        production = max(0, current - baseline)
        
        # Track production
        self.production_history[hormone].append((self.time, production))
        
        # Cost scales with production squared
        return production ** 2 / 100
    
    def calculate_effective_hormones(self) -> Dict[str, float]:
        """Calculate effective hormone levels using Hill equation"""
        effective = {}
        for hormone in self.hormones:
            X = self.hormones[hormone]
            R = self.receptors[hormone]
            n = HILL_COEFFICIENTS.get(hormone, 2.5)
            K = HALF_SATURATION.get(hormone, 40)
            
            # Hill equation with receptor sensitivity
            effective[hormone] = R * (X**n / (K**n + X**n))
            
        return effective
    
    def calculate_total_cost(self) -> Dict[str, float]:
        """Calculate cost based on minimization principle"""
        costs = {}
        
        # Production cost (main focus)
        production = sum(self.calculate_production_cost(h) for h in self.hormones)
        costs['production'] = production
        
        # Maintenance cost
        maintenance = sum(self.hormones[h] * 0.01 for h in self.hormones)
        costs['maintenance'] = maintenance
        
        # Change cost (smoothness)
        change = 0
        if len(self.history) > 1:
            for hormone in self.hormones:
                prev = self.history[-2]['hormones'][hormone]
                curr = self.hormones[hormone]
                change += abs(curr - prev)
        costs['change'] = change / 100
        
        # Metabolic cost
        costs['metabolic'] = (3.0 - sum(self.resources.values())) * 10
        
        # Allostatic cost
        costs['allostatic'] = self.allostatic_load ** 2
        
        # Total with emphasis on production minimization
        costs['total'] = (
            costs['production'] * COST_WEIGHTS.get('production', 10.0) +
            costs['maintenance'] * COST_WEIGHTS.get('maintenance', 2.0) +
            costs['change'] * COST_WEIGHTS.get('change', 1.0) +
            costs['metabolic'] * COST_WEIGHTS.get('metabolic', 0.5) +
            costs['allostatic'] * COST_WEIGHTS.get('allostatic', 1.0)
        )
        
        return costs
    
    def update_baseline_with_minimization(self, dt: float):
        """Update baselines using minimization principle"""
        tau_fast = 100.0  # Fast adaptation
        tau_slow = 1000.0  # Slow adaptation
        
        for hormone in self.hormones:
            # Get average spike amplitude
            avg_spike = self.get_average_spike(hormone)
            
            # Get opponent ratio
            r = self.opponent_ratios.get(hormone, 1.0)
            
            # Calculate target baseline (minimization formula)
            if avg_spike > 0:
                # New baseline = (current + avg_spike) / (1 + opponent_ratio)
                target = (self.baselines['slow'][hormone] + avg_spike) / (1 + r)
                
                # Adapt faster if ratio favors it
                if r < 1:  # Less opposition
                    tau = tau_fast
                else:
                    tau = tau_slow
                
                # Update baseline
                self.baselines['slow'][hormone] += (
                    (target - self.baselines['slow'][hormone]) * dt / tau
                )
            
            # Fast baseline tracks recent average
            self.baselines['fast'][hormone] += (
                (self.hormones[hormone] - self.baselines['fast'][hormone]) * 
                dt / tau_fast
            )
    
    def get_mood(self) -> str:
        """Determine mood from hormone state"""
        h = self.hormones
        
        # Calculate mood indicators
        pleasure = (h['dopamine'] + h['endorphins']) / 2
        calm = (h['serotonin'] + h['oxytocin']) / 2
        stress = (h['cortisol'] + h['adrenaline']) / 2
        focus = h['norepinephrine']
        
        # Determine primary mood
        if pleasure > 70 and stress < 40:
            return "euphoric"
        elif pleasure > 60 and calm > 50:
            return "joyful"
        elif stress > 60 and h['adrenaline'] > 50:
            return "anxious"
        elif stress > 50 and h['cortisol'] > 50:
            return "stressed"
        elif calm > 60 and stress < 30:
            return "serene"
        elif h['dopamine'] > 60 and focus > 50:
            return "motivated"
        elif h['serotonin'] < 40 and pleasure < 40:
            return "depressed"
        elif h['oxytocin'] > 60:
            return "loving"
        elif focus > 60:
            return "focused"
        elif h['endorphins'] > 50:
            return "relaxed"
        else:
            return "neutral"
