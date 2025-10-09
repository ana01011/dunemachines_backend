"""
Core neurochemical state with proper reward-prediction error for dopamine
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
from app.neurochemistry.core.constants import *

@dataclass
class TaskContext:
    """Context for current task/generation"""
    task_id: str = ""
    started_at: float = 0.0
    expected_quality: float = 0.7  # What AI expects to achieve
    progress: float = 0.0  # 0 to 1
    actual_quality: Optional[float] = None  # Measured after completion
    iteration: int = 0  # Which attempt is this

@dataclass 
class NeurochemicalState:
    """
    Complete neurochemical state with proper reward dynamics
    """
    
    # Current hormone levels X(t)
    dopamine: float = 50.0
    cortisol: float = 30.0
    adrenaline: float = 20.0
    serotonin: float = 60.0
    oxytocin: float = 40.0
    
    # Dynamic baselines B(t)
    dopamine_baseline: float = 50.0
    cortisol_baseline: float = 30.0
    adrenaline_baseline: float = 20.0
    serotonin_baseline: float = 60.0
    oxytocin_baseline: float = 40.0
    
    # Task and reward tracking
    current_task: Optional[TaskContext] = None
    recent_predictions: List[float] = field(default_factory=list)
    recent_outcomes: List[float] = field(default_factory=list)
    
    # Spike history
    dopamine_spikes: List[float] = field(default_factory=list)
    cortisol_spikes: List[float] = field(default_factory=list)
    
    # Adrenaline pool
    adrenaline_pool: float = 100.0
    
    # Time
    time: float = 0.0
    last_update: float = 0.0
    
    def start_task(self, task_id: str, expected_difficulty: float = 0.5):
        """Start a new task with expectation"""
        self.current_task = TaskContext(
            task_id=task_id,
            started_at=self.time,
            expected_quality=1.0 - expected_difficulty,
            iteration=0
        )
        
        # Anticipatory dopamine rise (slow and steady)
        # The harder the task, the more cautious the rise
        anticipation_rate = 2.0 * (1.0 - expected_difficulty)
        self.dopamine += anticipation_rate
        
        # Cortisol rises with difficulty
        self.cortisol += expected_difficulty * 5
        
        # Slight adrenaline for readiness
        self.adrenaline += expected_difficulty * 3
    
    def update_progress(self, progress: float):
        """Update during task execution"""
        if not self.current_task:
            return
            
        old_progress = self.current_task.progress
        self.current_task.progress = progress
        progress_delta = progress - old_progress
        
        if progress_delta > 0:
            # Making progress - small dopamine increases
            # But tempered by how well we think we're doing
            confidence = self.serotonin / 100.0
            self.dopamine += progress_delta * 5 * confidence
            
            # Reduce cortisol if progressing well
            if progress > 0.5:
                self.cortisol *= 0.98
        else:
            # Stuck or regressing
            self.cortisol += abs(progress_delta) * 3
            self.adrenaline += abs(progress_delta) * 2
    
    def complete_task(self, actual_quality: float):
        """Complete task and trigger reward-prediction error"""
        if not self.current_task:
            return
            
        self.current_task.actual_quality = actual_quality
        expected = self.current_task.expected_quality
        
        # THE CORE OF DOPAMINE: Reward Prediction Error
        prediction_error = actual_quality - expected
        
        # Store for learning
        self.recent_predictions.append(expected)
        self.recent_outcomes.append(actual_quality)
        if len(self.recent_predictions) > 20:
            self.recent_predictions.pop(0)
            self.recent_outcomes.pop(0)
        
        # Dopamine response to prediction error
        if prediction_error > 0:
            # POSITIVE SURPRISE - Big spike!
            spike_amplitude = REWARD_PREDICTION_GAIN * prediction_error * 30
            self.dopamine += spike_amplitude
            
            # Serotonin also rises with success
            self.serotonin += prediction_error * 10
            
            # Schedule opponent process (crash after spike)
            self._schedule_opponent_process(spike_amplitude)
            
        elif prediction_error < 0:
            # NEGATIVE SURPRISE - Immediate drop
            crash_amplitude = abs(prediction_error) * 20
            self.dopamine -= crash_amplitude
            
            # Cortisol rises with failure
            self.cortisol += abs(prediction_error) * 15
            
            # Serotonin drops with failure
            self.serotonin -= abs(prediction_error) * 8
        
        # Reset for next task
        self.current_task = None
    
    def retry_task(self, task_id: str):
        """Retry after failure - adjusted expectations"""
        if len(self.recent_outcomes) > 0:
            # Learn from recent performance
            avg_performance = np.mean(self.recent_outcomes[-5:])
            expected = min(avg_performance * 1.1, 0.9)  # Slightly optimistic
        else:
            expected = 0.6  # Conservative
            
        self.current_task = TaskContext(
            task_id=task_id,
            started_at=self.time,
            expected_quality=expected,
            iteration=1
        )
        
        # More cautious dopamine rise on retry
        self.dopamine += 1.0  # Smaller anticipation
        
        # Cortisol stays elevated (pressure)
        self.cortisol += 2.0
    
    def _schedule_opponent_process(self, spike_amplitude: float):
        """Schedule the crash after dopamine spike"""
        # This will be applied over next few time steps
        self.dopamine_spikes.append(spike_amplitude)
    
    def apply_dynamics(self, dt: float, event: Optional['Event'] = None):
        """Apply dynamics with proper reward processing"""
        X = self.get_state_vector()
        B = self.get_baseline_vector()
        W = X - B
        
        # Return to baseline
        Lambda = np.diag([
            LAMBDA_DOPAMINE,
            LAMBDA_CORTISOL,
            LAMBDA_ADRENALINE,
            LAMBDA_SEROTONIN,
            LAMBDA_OXYTOCIN
        ])
        
        dX_dt = -Lambda @ W
        
        # Interactions
        J = self.calculate_interaction_matrix()
        dX_dt += J @ X
        
        # Apply opponent process for dopamine if scheduled
        if len(self.dopamine_spikes) > 0:
            # Gradual crash after spike
            for spike in self.dopamine_spikes:
                crash_force = OPPONENT_PROCESS_STRENGTH * spike * np.exp(-self.time * 0.1)
                dX_dt[0] -= crash_force
            
            # Clear old spikes
            self.dopamine_spikes = [s for s in self.dopamine_spikes if s > 0.1]
        
        # Event response if provided
        if event:
            dX_dt += self.calculate_event_response(event)
        
        # Noise (reduced - we want more deterministic behavior)
        noise = np.array([
            np.random.normal(0, NOISE_AMPLITUDE['dopamine'] * 0.3),
            np.random.normal(0, NOISE_AMPLITUDE['cortisol'] * 0.3),
            np.random.normal(0, NOISE_AMPLITUDE['adrenaline'] * 0.3),
            np.random.normal(0, NOISE_AMPLITUDE['serotonin'] * 0.3),
            np.random.normal(0, NOISE_AMPLITUDE['oxytocin'] * 0.3)
        ])
        
        dX_dt += noise
        
        # Update
        X_new = X + dX_dt * dt
        X_new = np.clip(X_new, MIN_HORMONE, MAX_HORMONE)
        
        # Adrenaline pool
        self.update_adrenaline_pool(dt)
        X_new[2] = min(X_new[2], self.adrenaline_pool)
        
        # Set new values
        self.dopamine = X_new[0]
        self.cortisol = X_new[1] 
        self.adrenaline = X_new[2]
        self.serotonin = X_new[3]
        self.oxytocin = X_new[4]
        
        # Adapt baselines
        self.adapt_baselines(dt)
        
        # Update time
        self.time += dt
        self.last_update = self.time
    
    def get_state_vector(self) -> np.ndarray:
        return np.array([
            self.dopamine, self.cortisol, self.adrenaline,
            self.serotonin, self.oxytocin
        ])
    
    def get_baseline_vector(self) -> np.ndarray:
        return np.array([
            self.dopamine_baseline, self.cortisol_baseline,
            self.adrenaline_baseline, self.serotonin_baseline,
            self.oxytocin_baseline
        ])
    
    def calculate_interaction_matrix(self) -> np.ndarray:
        """Hormone interactions - key for emergent behavior"""
        J = np.zeros((5, 5))
        
        # Dopamine row - affected by others
        J[0, 1] = -INTERACTION_MATRIX['beta_DC'] * (self.cortisol/50)  # Cortisol suppresses more when high
        J[0, 2] = INTERACTION_MATRIX['alpha_DA']
        J[0, 3] = -INTERACTION_MATRIX['delta_DS'] * (1 - self.serotonin/100)  # Low serotonin suppresses
        
        # Cortisol row
        J[1, 0] = INTERACTION_MATRIX['gamma_CD'] * max(0, 50 - self.dopamine)/50
        J[1, 2] = INTERACTION_MATRIX['alpha_CA']
        J[1, 3] = -INTERACTION_MATRIX['mu_CS']
        
        # Adrenaline row
        J[2, 0] = INTERACTION_MATRIX['zeta_AD']
        J[2, 1] = INTERACTION_MATRIX['xi_AC']
        J[2, 3] = -INTERACTION_MATRIX['nu_AS']
        
        # Serotonin row
        J[3, 0] = -INTERACTION_MATRIX['theta_SD'] * (1 - self.dopamine/100)
        J[3, 1] = -INTERACTION_MATRIX['sigma_SC'] * (self.cortisol/50)
        J[3, 2] = -INTERACTION_MATRIX['rho_SA']
        J[3, 4] = INTERACTION_MATRIX['epsilon_SO']
        
        # Oxytocin row
        J[4, 0] = INTERACTION_MATRIX['kappa_OD']
        J[4, 3] = INTERACTION_MATRIX['lambda_OS']
        
        return J
    
    def calculate_event_response(self, event: 'Event') -> np.ndarray:
        """Event responses - but dopamine is mainly from task completion"""
        response = np.zeros(5)
        
        # Dopamine - only small responses to events
        # Main dopamine comes from task completion
        response[0] = event.novelty * 2  # Small novelty bonus
        
        # Cortisol - stress response
        response[1] = (
            event.complexity * 10 +
            event.urgency * 8 +
            event.uncertainty * 5
        )
        
        # Adrenaline - urgency
        response[2] = (
            event.urgency * 15 +
            event.intensity * 5
        )
        
        # Serotonin - social and success
        response[3] = (
            event.social_interaction * 8 +
            event.emotional_content * 5 -
            event.threat_level * 10  # Threats reduce serotonin
        )
        
        # Oxytocin - bonding
        response[4] = (
            event.social_interaction * 12 +
            event.emotional_content * 8 +
            event.trust_factor * 6
        )
        
        return response
    
    def adapt_baselines(self, dt: float):
        """Baseline adaptation with learning"""
        # Learn expected performance from history
        if len(self.recent_outcomes) > 5:
            avg_performance = np.mean(self.recent_outcomes[-10:])
            performance_trend = np.mean(np.diff(self.recent_outcomes[-5:]))
            
            # Adjust dopamine baseline based on average performance
            if avg_performance > 0.7:
                # Doing well - slightly raise baseline (but not too much)
                self.dopamine_baseline += 0.1 * dt
            elif avg_performance < 0.4:
                # Struggling - lower baseline
                self.dopamine_baseline -= 0.1 * dt
        
        # Stress adaptation
        if self.cortisol > 60:
            # Chronic stress - baseline creeps up
            self.cortisol_baseline += 0.05 * dt
        else:
            # Relax back to normal
            self.cortisol_baseline -= 0.02 * (self.cortisol_baseline - 30) * dt
        
        # Clamp baselines
        self.dopamine_baseline = np.clip(self.dopamine_baseline, 30, 60)
        self.cortisol_baseline = np.clip(self.cortisol_baseline, 20, 50)
        self.serotonin_baseline = np.clip(self.serotonin_baseline, 40, 70)
    
    def update_adrenaline_pool(self, dt: float):
        self.adrenaline_pool += ADRENALINE_REGEN_RATE * dt
        self.adrenaline_pool -= self.adrenaline * ADRENALINE_USAGE_RATE * dt
        self.adrenaline_pool = np.clip(self.adrenaline_pool, 0, MAX_HORMONE)
    
    def get_behavioral_parameters(self) -> Dict[str, float]:
        """Behavioral parameters emerge from hormone state"""
        W = self.get_state_vector() - self.get_baseline_vector()
        
        # Planning depth - emerges from cortisol and serotonin
        planning_depth = 5 * (1 + self.cortisol/100) * (self.serotonin/60)
        
        # Risk tolerance - emerges from dopamine vs cortisol balance
        risk_tolerance = 0.5 + (self.dopamine - 50)/100 - (self.cortisol - 30)/100
        
        # Processing speed - adrenaline driven
        processing_speed = 1.0 + self.adrenaline/50
        
        # Confidence - serotonin and recent success
        if len(self.recent_outcomes) > 0:
            recent_success = np.mean(self.recent_outcomes[-5:])
        else:
            recent_success = 0.5
        confidence = (self.serotonin/100) * 0.5 + recent_success * 0.5
        
        # Patience - inverse of adrenaline and cortisol
        patience = 1.0 - (self.adrenaline/100) * 0.5 - (self.cortisol/100) * 0.3
        
        # Creativity - dopamine and low cortisol
        creativity = (self.dopamine/100) * (1 - self.cortisol/100)
        
        # Empathy - oxytocin and serotonin
        empathy = (self.oxytocin/100) * 0.6 + (self.serotonin/100) * 0.4
        
        return {
            'planning_depth': max(1, planning_depth),
            'risk_tolerance': np.clip(risk_tolerance, 0, 1),
            'processing_speed': max(0.5, processing_speed),
            'confidence': np.clip(confidence, 0, 1),
            'patience': np.clip(patience, 0, 1),
            'creativity': np.clip(creativity, 0, 1),
            'empathy': np.clip(empathy, 0, 1)
        }
    
    def check_stability(self) -> bool:
        J = self.calculate_interaction_matrix()
        eigenvalues = np.linalg.eigvals(J)
        return np.max(np.abs(eigenvalues)) < 1.0
    
    def calculate_lyapunov_function(self) -> float:
        X = self.get_state_vector()
        B = self.get_baseline_vector()
        W = X - B
        return 0.5 * np.sum(W**2) + 0.25 * np.sum(B**2)

@dataclass
class Event:
    """Event that affects neurochemistry"""
    type: str
    intensity: float = 0.5
    complexity: float = 0.5
    urgency: float = 0.5
    emotional_content: float = 0.5
    novelty: float = 0.5
    social_interaction: float = 0.5
    uncertainty: float = 0.5
    trust_factor: float = 0.5
    threat_level: float = 0.0  # New: for fear/anger emergence
