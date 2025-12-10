"""
Enhanced Neurochemical Dynamics with Opponent Processes
"""

import numpy as np
from typing import Dict, Optional
import time

# Import constants - try multiple paths
try:
    from .constants import HOMEOSTASIS_RATES
except ImportError:
    # Define minimal constants if import fails
    HOMEOSTASIS_RATES = {
        'dopamine': 0.05, 'serotonin': 0.03, 'cortisol': 0.08,
        'adrenaline': 0.15, 'oxytocin': 0.04, 'norepinephrine': 0.10,
        'endorphins': 0.12
    }

class NeurochemicalDynamics:
    """Implements enhanced dynamics with opponent processes"""
    
    def __init__(self, state=None):
        # Import state here to avoid circular import
        if state is None:
            try:
                from .state_enhanced import NeurochemicalState
            except ImportError:
                from state_enhanced import NeurochemicalState
            state = NeurochemicalState()
        
        self.state = state
        self.dt = 0.1  # Time step
        
    def calculate_opponent_recovery(self, hormone: str) -> float:
        """Calculate recovery force based on opponent ratios"""
        h = self.state.hormones
        
        # Define opponent pairs
        opponents = {
            'dopamine': ['serotonin', 'oxytocin'],
            'serotonin': ['dopamine', 'norepinephrine'],
            'cortisol': ['oxytocin', 'serotonin'],
            'adrenaline': ['endorphins', 'oxytocin'],
            'oxytocin': ['cortisol', 'adrenaline'],
            'norepinephrine': ['serotonin', 'endorphins'],
            'endorphins': ['adrenaline', 'cortisol']
        }
        
        recovery = 0.0
        if hormone in opponents:
            for opponent in opponents[hormone]:
                # Recovery proportional to opponent level
                recovery -= (h[hormone] - self.state.baselines['slow'][hormone]) * \
                          (h[opponent] / 100) * 0.1
        
        return recovery
    
    def calculate_nonlinear_decay(self, hormone: str) -> float:
        """Non-linear decay with opponent influence"""
        level = self.state.hormones[hormone]
        baseline = self.state.baselines['slow'][hormone]
        ratio = self.state.opponent_ratios.get(hormone, 1.0)
        
        # Base decay rate
        lambda_base = HOMEOSTASIS_RATES.get(hormone, 0.05)
        
        # Decay faster when:
        # 1. Hormone is high (quadratic)
        # 2. Opponent ratio is unfavorable
        # 3. Above baseline
        decay_rate = lambda_base * (1 + ratio) * (level/100)**2
        
        # Never plateau - ensure continuous decay
        if level > baseline * 1.5:
            decay_rate *= 2.0  # Accelerate decay for high levels
        
        return -decay_rate * (level - baseline)
    
    def calculate_production(self, hormone: str, inputs: Dict) -> float:
        """Calculate hormone production based on inputs"""
        production = 0.0
        
        # Input-driven production
        if hormone == 'dopamine':
            if inputs.get('reward', 0) > 0:
                production += inputs['reward'] * 30 * (1 - self.state.hormones['dopamine']/100)
            if inputs.get('novelty', 0) > 0:
                production += inputs['novelty'] * 20
                
        elif hormone == 'serotonin':
            if inputs.get('success', 0) > 0:
                production += inputs['success'] * 25
            if inputs.get('social', 0) > 0:
                production += inputs['social'] * 15
                
        elif hormone == 'cortisol':
            if inputs.get('threat', 0) > 0:
                production += inputs['threat'] * 40
            if inputs.get('uncertainty', 0) > 0:
                production += inputs['uncertainty'] * 20
            # Cortisol REDUCTION for relaxation
            if inputs.get('relaxation', 0) > 0:
                production -= inputs['relaxation'] * 30
                
        elif hormone == 'adrenaline':
            if inputs.get('urgency', 0) > 0:
                production += inputs['urgency'] * 50 * (1 - self.state.hormones['adrenaline']/100)
            if inputs.get('threat', 0) > 0:
                production += inputs['threat'] * 35
            # Adrenaline reduction for relaxation
            if inputs.get('relaxation', 0) > 0:
                production -= inputs['relaxation'] * 25
                
        elif hormone == 'oxytocin':
            if inputs.get('social', 0) > 0:
                production += inputs['social'] * 35
            if inputs.get('trust', 0) > 0:
                production += inputs['trust'] * 25
                
        elif hormone == 'norepinephrine':
            if inputs.get('attention', 0) > 0:
                production += inputs['attention'] * 30
            if inputs.get('urgency', 0) > 0:
                production += inputs['urgency'] * 20
                
        elif hormone == 'endorphins':
            if inputs.get('exercise', 0) > 0:
                production += inputs['exercise'] * 45
            if inputs.get('pleasure', 0) > 0:
                production += inputs['pleasure'] * 30
            if inputs.get('relaxation', 0) > 0:
                production += inputs['relaxation'] * 20
        
        # Scale by resources
        if hormone in ['dopamine', 'norepinephrine', 'adrenaline']:
            production *= self.state.resources['tyrosine']
        elif hormone == 'serotonin':
            production *= self.state.resources['tryptophan']
        
        production *= self.state.resources['atp']
        
        return production
    
    def update_resources(self, production_rates: Dict[str, float], dt: float):
        """Update metabolic resources based on production"""
        # Calculate usage
        tyrosine_usage = (
            abs(production_rates.get('dopamine', 0)) * 0.002 +
            abs(production_rates.get('norepinephrine', 0)) * 0.002 +
            abs(production_rates.get('adrenaline', 0)) * 0.001
        ) * dt
        
        tryptophan_usage = abs(production_rates.get('serotonin', 0)) * 0.003 * dt
        
        # ATP usage proportional to total activity
        total_production = sum(abs(p) for p in production_rates.values())
        atp_usage = total_production * 0.001 * dt
        
        # Deplete resources
        self.state.resources['tyrosine'] = max(0.1, 
            self.state.resources['tyrosine'] - tyrosine_usage)
        self.state.resources['tryptophan'] = max(0.1,
            self.state.resources['tryptophan'] - tryptophan_usage)
        self.state.resources['atp'] = max(0.1,
            self.state.resources['atp'] - atp_usage)
        
        # Recovery (slower than depletion)
        recovery_rate = 0.001 * dt
        for resource in self.state.resources:
            self.state.resources[resource] += (
                (1.0 - self.state.resources[resource]) * recovery_rate
            )
    
    def update_allostatic_load(self, dt: float):
        """Update allostatic load based on chronic stress"""
        cortisol = self.state.hormones['cortisol']
        threshold = 40.0  # Lower threshold
        
        if cortisol > threshold:
            # Accumulate load (quadratic for high stress)
            accumulation = ((cortisol - threshold) / 30) ** 2 * dt * 0.01
            self.state.allostatic_load += accumulation
        else:
            # Slow recovery
            self.state.allostatic_load *= (1 - 0.001 * dt)
        
        # Cap at 1.0
        self.state.allostatic_load = min(1.0, self.state.allostatic_load)
    
    def step(self, dt: float, inputs: Dict):
        """Single dynamics step with opponent processes"""
        # Calculate opponent ratios
        self.state.calculate_opponent_ratios()
        
        # Track spikes
        for hormone in self.state.hormones:
            self.state.track_spike(hormone)
        
        # Calculate changes for each hormone
        production_rates = {}
        changes = {}
        
        for hormone in self.state.hormones:
            # Production from inputs
            production = self.calculate_production(hormone, inputs)
            
            # Non-linear decay with opponent influence
            decay = self.calculate_nonlinear_decay(hormone)
            
            # Opponent-based recovery
            recovery = self.calculate_opponent_recovery(hormone)
            
            # Total change
            change = production + decay + recovery
            changes[hormone] = change
            production_rates[hormone] = production
        
        # Apply changes
        for hormone, change in changes.items():
            self.state.hormones[hormone] += change * dt
            # Bound between 0 and 100
            self.state.hormones[hormone] = np.clip(
                self.state.hormones[hormone], 0, 100
            )
        
        # Update other systems
        self.update_resources(production_rates, dt)
        self.update_allostatic_load(dt)
        self.state.update_baseline_with_minimization(dt)
        
        # Update receptors
        self.update_receptors(dt)
        
        # Update time
        self.state.time += dt
        
        # Store history
        self.state.history.append({
            'time': self.state.time,
            'hormones': self.state.hormones.copy(),
            'resources': self.state.resources.copy(),
            'cost': self.state.calculate_total_cost()
        })
    
    def update_receptors(self, dt: float):
        """Update receptor sensitivities"""
        for hormone in self.state.receptors:
            level = self.state.hormones[hormone]
            receptor = self.state.receptors[hormone]
            
            # Desensitization when hormone is high
            if level > 60:
                desensitization = 0.01 * (level/100) * dt
                self.state.receptors[hormone] -= desensitization
            
            # Recovery when hormone is low
            if level < 40:
                recovery = 0.005 * (1 - level/40) * dt
                self.state.receptors[hormone] += recovery
            
            # Bound between 0.1 and 1.0
            self.state.receptors[hormone] = np.clip(
                self.state.receptors[hormone], 0.1, 1.0
            )
