"""
Learning System - Hebbian Learning with Reward Modulation
OMNIUS v2 - Biological learning for the Neural Router
"""
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from enum import Enum


class RewardType(str, Enum):
    """Types of reward signals"""
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_SUCCESS = "implicit_success"
    TASK_COMPLETION = "task_completion"
    ERROR_SIGNAL = "error_signal"
    TIMEOUT = "timeout"


@dataclass
class RewardSignal:
    """A reward signal for learning"""
    value: float
    reward_type: RewardType
    source: str
    decision_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "reward_type": self.reward_type.value,
            "source": self.source,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LearningEvent:
    """Record of a learning update"""
    id: str = field(default_factory=lambda: str(uuid4()))
    decision_id: str = ""
    reward: float = 0.0
    reward_type: RewardType = RewardType.IMPLICIT_SUCCESS
    learning_rate_used: float = 0.01
    weight_change_magnitude: float = 0.0
    neurochemistry_state: Dict[str, float] = field(default_factory=dict)
    pre_accuracy: Optional[float] = None
    post_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "decision_id": self.decision_id,
            "reward": self.reward,
            "reward_type": self.reward_type.value,
            "learning_rate_used": self.learning_rate_used,
            "weight_change_magnitude": self.weight_change_magnitude,
            "neurochemistry_state": self.neurochemistry_state,
            "pre_accuracy": self.pre_accuracy,
            "post_accuracy": self.post_accuracy,
            "timestamp": self.timestamp.isoformat()
        }


class HebbianLearner:
    """
    Hebbian Learning System with BCM modifications.
    
    Implements:
    - Basic Hebbian: neurons that fire together wire together
    - BCM rule: sliding threshold for LTP/LTD
    - Reward modulation: learning scaled by reward
    - Eligibility traces: for delayed reward
    - Neurochemistry modulation: hormones affect learning
    """
    
    def __init__(
        self,
        base_learning_rate: float = 0.01,
        eligibility_decay: float = 0.9,
        bcm_threshold: float = 0.5,
        weight_decay: float = 0.0001
    ):
        self.base_learning_rate = base_learning_rate
        self.eligibility_decay = eligibility_decay
        self.bcm_threshold = bcm_threshold
        self.weight_decay = weight_decay
        
        # Learning statistics
        self.total_updates = 0
        self.reward_history: List[float] = []
        self.learning_events: List[LearningEvent] = []
        
        # Running averages for BCM
        self.activity_average = 0.5
        self.activity_tau = 0.99
        
        # Neurochemistry state
        self.neuro_state: Dict[str, float] = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.5,
            "norepinephrine": 0.5
        }
    
    def set_neurochemistry(self, state: Dict[str, float]):
        """Update neurochemistry state"""
        self.neuro_state.update(state)
    
    def get_modulated_learning_rate(self) -> float:
        """
        Get learning rate modulated by neurochemistry.
        
        - Dopamine: increases learning (reward prediction)
        - Cortisol: decreases learning (stress)
        - Norepinephrine: increases attention/learning
        """
        dopamine = self.neuro_state.get("dopamine", 0.5)
        cortisol = self.neuro_state.get("cortisol", 0.5)
        norepinephrine = self.neuro_state.get("norepinephrine", 0.5)
        
        # Dopamine boosts learning
        dopamine_factor = 0.5 + dopamine
        
        # High cortisol reduces learning
        cortisol_factor = 1.0 - 0.5 * max(0, cortisol - 0.5)
        
        # Norepinephrine boosts learning
        norepi_factor = 0.8 + 0.4 * norepinephrine
        
        modulated_lr = self.base_learning_rate * dopamine_factor * cortisol_factor * norepi_factor
        
        return np.clip(modulated_lr, 0.001, 0.1)
    
    def compute_hebbian_update(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 1.0,
        eligibility_trace: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Hebbian weight update.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            reward: Reward signal (-1 to 1)
            eligibility_trace: Optional eligibility trace for delayed reward
        
        Returns:
            Tuple of (weight_delta, magnitude)
        """
        lr = self.get_modulated_learning_rate()
        
        if eligibility_trace is not None:
            # Use eligibility trace for temporal credit assignment
            delta_w = lr * reward * eligibility_trace
        else:
            # Standard Hebbian with BCM modification
            post_centered = post_activity - self.bcm_threshold
            delta_w = lr * reward * np.outer(pre_activity, post_centered)
        
        # Apply weight decay
        magnitude = float(np.mean(np.abs(delta_w)))
        
        return delta_w, magnitude
    
    def compute_bcm_threshold_update(self, post_activity: np.ndarray):
        """
        Update BCM sliding threshold based on recent activity.
        
        BCM rule: threshold adapts to average post-synaptic activity
        """
        avg_activity = np.mean(post_activity)
        self.activity_average = (
            self.activity_tau * self.activity_average + 
            (1 - self.activity_tau) * avg_activity
        )
        self.bcm_threshold = self.activity_average
    
    def process_reward(
        self,
        reward_signal: RewardSignal,
        router: Any,
        decision_id: str
    ) -> LearningEvent:
        """
        Process a reward signal and trigger learning.
        
        Args:
            reward_signal: The reward signal to process
            router: The NeuralRouter to update
            decision_id: ID of the routing decision
        
        Returns:
            LearningEvent with details of the update
        """
        # Get pre-learning accuracy estimate
        pre_accuracy = self._estimate_accuracy()
        
        # Apply learning to router
        learn_result = router.learn(
            reward=reward_signal.value,
            learning_rate_modifier=self.get_modulated_learning_rate() / self.base_learning_rate
        )
        
        # Update BCM threshold
        if router.last_output is not None:
            self.compute_bcm_threshold_update(router.last_output)
        
        # Track reward
        self.reward_history.append(reward_signal.value)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        # Get post-learning accuracy estimate
        post_accuracy = self._estimate_accuracy()
        
        # Create learning event
        event = LearningEvent(
            decision_id=decision_id,
            reward=reward_signal.value,
            reward_type=reward_signal.reward_type,
            learning_rate_used=learn_result.get("learning_rate", self.base_learning_rate),
            weight_change_magnitude=learn_result.get("weight_change", 0.0),
            neurochemistry_state=self.neuro_state.copy(),
            pre_accuracy=pre_accuracy,
            post_accuracy=post_accuracy
        )
        
        self.learning_events.append(event)
        self.total_updates += 1
        
        return event
    
    def _estimate_accuracy(self) -> float:
        """Estimate current accuracy from reward history"""
        if not self.reward_history:
            return 0.5
        
        recent = self.reward_history[-100:]
        positive = sum(1 for r in recent if r > 0)
        return positive / len(recent)
    
    def create_reward_from_feedback(
        self,
        rating: int,
        decision_id: str,
        feedback_type: str = "rating"
    ) -> RewardSignal:
        """
        Convert user feedback (1-5 stars) to reward signal.
        
        Args:
            rating: User rating 1-5
            decision_id: ID of the routing decision
            feedback_type: Type of feedback
        
        Returns:
            RewardSignal
        """
        # Convert 1-5 to -1 to 1
        reward_value = (rating - 3) / 2.0
        
        return RewardSignal(
            value=reward_value,
            reward_type=RewardType.EXPLICIT_FEEDBACK,
            source="user",
            decision_id=decision_id,
            metadata={"original_rating": rating, "feedback_type": feedback_type}
        )
    
    def create_reward_from_completion(
        self,
        success: bool,
        decision_id: str,
        response_time_ms: float = 0
    ) -> RewardSignal:
        """
        Create reward signal from task completion.
        
        Args:
            success: Whether task completed successfully
            decision_id: ID of the routing decision
            response_time_ms: Response time in milliseconds
        
        Returns:
            RewardSignal
        """
        if success:
            # Reward with penalty for slow responses
            time_penalty = min(0.3, response_time_ms / 10000)
            reward_value = 0.5 - time_penalty
        else:
            reward_value = -0.5
        
        return RewardSignal(
            value=reward_value,
            reward_type=RewardType.TASK_COMPLETION,
            source="system",
            decision_id=decision_id,
            metadata={"success": success, "response_time_ms": response_time_ms}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "total_updates": self.total_updates,
            "base_learning_rate": self.base_learning_rate,
            "modulated_learning_rate": self.get_modulated_learning_rate(),
            "bcm_threshold": self.bcm_threshold,
            "activity_average": self.activity_average,
            "estimated_accuracy": self._estimate_accuracy(),
            "reward_history_length": len(self.reward_history),
            "recent_avg_reward": np.mean(self.reward_history[-50:]) if self.reward_history else 0,
            "neurochemistry": self.neuro_state
        }
    
    def get_recent_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning events"""
        return [e.to_dict() for e in self.learning_events[-n:]]


def create_learner(
    learning_rate: float = 0.01,
    eligibility_decay: float = 0.9
) -> HebbianLearner:
    """Factory function to create a Hebbian learner"""
    return HebbianLearner(
        base_learning_rate=learning_rate,
        eligibility_decay=eligibility_decay
    )
