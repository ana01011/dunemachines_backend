"""
Expectation learning for adaptive predictions
"""
from typing import Dict, Optional, Tuple, List
import numpy as np
from collections import deque
from datetime import datetime
import logging

from ..core.constants import EventType

logger = logging.getLogger(__name__)

class ExpectationLearner:
    """
    Learns to predict outcomes and adjust expectations
    """
    
    def __init__(self, user_id: str):
        """
        Initialize expectation learner
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        
        # Expectation tracking
        self.expectations = {
            'task_difficulty': deque(maxlen=100),
            'task_duration': deque(maxlen=100),
            'success_probability': deque(maxlen=100),
            'quality_score': deque(maxlen=100)
        }
        
        # Prediction errors for learning
        self.prediction_errors = {
            'difficulty': deque(maxlen=50),
            'duration': deque(maxlen=50),
            'success': deque(maxlen=50),
            'quality': deque(maxlen=50)
        }
        
        # Learned parameters
        self.calibration = {
            'optimism_bias': 0.0,  # -1 (pessimistic) to 1 (optimistic)
            'confidence': 0.5,      # 0 (uncertain) to 1 (certain)
            'accuracy': 0.5,        # 0 (inaccurate) to 1 (accurate)
            'adaptability': 0.5     # 0 (rigid) to 1 (flexible)
        }
        
        # Task-specific expectations
        self.task_expectations = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.min_samples = 5
        
        # Running averages
        self.running_avg = {
            'difficulty': 0.5,
            'duration': 5.0,  # minutes
            'success_rate': 0.7,
            'quality': 0.7
        }
        
    def set_expectation(self, task_type: str, expectation_type: str, value: float):
        """
        Set an expectation for a task
        
        Args:
            task_type: Type of task
            expectation_type: Type of expectation (difficulty, duration, etc.)
            value: Expected value
        """
        if task_type not in self.task_expectations:
            self.task_expectations[task_type] = {}
            
        self.task_expectations[task_type][expectation_type] = {
            'value': value,
            'timestamp': datetime.now(),
            'confidence': self.calibration['confidence']
        }
        
        # Store in history
        if expectation_type in self.expectations:
            self.expectations[expectation_type].append({
                'value': value,
                'task': task_type,
                'time': datetime.now()
            })
            
        logger.debug(f"Set expectation: {task_type}.{expectation_type} = {value:.2f}")
    
    def record_outcome(self, task_type: str, outcome_type: str, 
                      actual_value: float, expected_value: Optional[float] = None):
        """
        Record actual outcome and learn from prediction error
        
        Args:
            task_type: Type of task
            outcome_type: Type of outcome
            actual_value: Actual observed value
            expected_value: Expected value (if not provided, uses stored expectation)
        """
        # Get expected value if not provided
        if expected_value is None:
            if task_type in self.task_expectations and \
               outcome_type in self.task_expectations[task_type]:
                expected_value = self.task_expectations[task_type][outcome_type]['value']
            else:
                expected_value = self.running_avg.get(outcome_type, 0.5)
                
        # Calculate prediction error
        error = actual_value - expected_value
        relative_error = error / (expected_value + 1e-6)
        
        # Store error
        if outcome_type in self.prediction_errors:
            self.prediction_errors[outcome_type].append({
                'error': error,
                'relative_error': relative_error,
                'expected': expected_value,
                'actual': actual_value,
                'task': task_type,
                'time': datetime.now()
            })
            
        # Update calibration
        self._update_calibration(outcome_type, error, relative_error)
        
        # Update running averages
        self._update_running_averages(outcome_type, actual_value)
        
        logger.debug(f"Outcome: {task_type}.{outcome_type} expected={expected_value:.2f}, "
                    f"actual={actual_value:.2f}, error={error:.2f}")
    
    def _update_calibration(self, outcome_type: str, error: float, relative_error: float):
        """
        Update calibration parameters based on prediction error
        
        Args:
            outcome_type: Type of outcome
            error: Absolute error
            relative_error: Relative error
        """
        # Update optimism bias
        if abs(relative_error) > 0.1:  # Significant error
            if error > 0:  # Underestimated (actual > expected)
                # We were pessimistic, increase optimism
                self.calibration['optimism_bias'] += self.learning_rate * 0.1
            else:  # Overestimated (actual < expected)
                # We were optimistic, decrease optimism
                self.calibration['optimism_bias'] -= self.learning_rate * 0.1
                
        # Clip optimism bias
        self.calibration['optimism_bias'] = np.clip(self.calibration['optimism_bias'], -1, 1)
        
        # Update accuracy based on error magnitude
        recent_errors = list(self.prediction_errors.get(outcome_type, []))[-10:]
        if len(recent_errors) >= self.min_samples:
            avg_abs_error = np.mean([abs(e['relative_error']) for e in recent_errors])
            # Lower error = higher accuracy
            self.calibration['accuracy'] = 1.0 / (1.0 + avg_abs_error)
            
        # Update confidence based on consistency of errors
        if len(recent_errors) >= self.min_samples:
            error_variance = np.var([e['relative_error'] for e in recent_errors])
            # Lower variance = higher confidence
            self.calibration['confidence'] = 1.0 / (1.0 + error_variance)
            
        # Update adaptability based on improvement trend
        if len(recent_errors) >= self.min_samples * 2:
            first_half = recent_errors[:len(recent_errors)//2]
            second_half = recent_errors[len(recent_errors)//2:]
            
            first_avg = np.mean([abs(e['relative_error']) for e in first_half])
            second_avg = np.mean([abs(e['relative_error']) for e in second_half])
            
            if second_avg < first_avg:  # Improving
                self.calibration['adaptability'] = min(1.0, 
                    self.calibration['adaptability'] + 0.05)
            else:  # Not improving
                self.calibration['adaptability'] = max(0.0,
                    self.calibration['adaptability'] - 0.02)
    
    def _update_running_averages(self, outcome_type: str, actual_value: float):
        """
        Update running averages with momentum
        
        Args:
            outcome_type: Type of outcome
            actual_value: Actual observed value
        """
        if outcome_type in self.running_avg:
            old_avg = self.running_avg[outcome_type]
            # Exponential moving average with momentum
            new_avg = self.momentum * old_avg + (1 - self.momentum) * actual_value
            self.running_avg[outcome_type] = new_avg
    
    def predict(self, task_type: str, prediction_type: str) -> Tuple[float, float]:
        """
        Make a prediction with confidence
        
        Args:
            task_type: Type of task
            prediction_type: Type of prediction
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        # Start with running average
        base_prediction = self.running_avg.get(prediction_type, 0.5)
        
        # Adjust for task-specific history
        if task_type in self.task_expectations and \
           prediction_type in self.task_expectations[task_type]:
            task_history = self.task_expectations[task_type][prediction_type]
            # Blend with task-specific expectation
            base_prediction = 0.7 * base_prediction + 0.3 * task_history['value']
            
        # Apply optimism bias
        if self.calibration['optimism_bias'] > 0:
            # Optimistic: increase positive predictions
            base_prediction *= (1 + 0.2 * self.calibration['optimism_bias'])
        else:
            # Pessimistic: decrease positive predictions
            base_prediction *= (1 + 0.2 * self.calibration['optimism_bias'])  # negative value
            
        # Clip to valid range
        if prediction_type in ['difficulty', 'success_rate', 'quality']:
            base_prediction = np.clip(base_prediction, 0, 1)
        elif prediction_type == 'duration':
            base_prediction = max(0.1, base_prediction)  # At least 0.1 minutes
            
        # Calculate confidence
        confidence = self.calibration['confidence'] * self.calibration['accuracy']
        
        return base_prediction, confidence
    
    def get_adjusted_reward_expectation(self, base_reward: float) -> float:
        """
        Adjust reward expectation based on learned calibration
        
        Args:
            base_reward: Base reward expectation
            
        Returns:
            Adjusted reward expectation
        """
        # Apply optimism bias
        adjusted = base_reward * (1 + 0.3 * self.calibration['optimism_bias'])
        
        # Apply accuracy correction
        if self.calibration['accuracy'] < 0.5:
            # Low accuracy, be more conservative
            adjusted *= (0.5 + self.calibration['accuracy'])
            
        # Apply confidence scaling
        if self.calibration['confidence'] < 0.3:
            # Low confidence, expect average
            adjusted = 0.7 * adjusted + 0.3 * 0.5
            
        return np.clip(adjusted, 0, 1)
    
    def should_update_strategy(self) -> bool:
        """
        Determine if strategy should be updated based on learning
        
        Returns:
            True if strategy update is recommended
        """
        # Check if we have enough data
        total_errors = sum(len(errors) for errors in self.prediction_errors.values())
        if total_errors < self.min_samples * 2:
            return False
            
        # Update if accuracy is low
        if self.calibration['accuracy'] < 0.4:
            return True
            
        # Update if not adapting well
        if self.calibration['adaptability'] < 0.3:
            return True
            
        # Update if consistently biased
        if abs(self.calibration['optimism_bias']) > 0.7:
            return True
            
        return False
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary of learned expectations
        
        Returns:
            Learning summary dictionary
        """
        summary = {
            'user_id': self.user_id,
            'calibration': self.calibration.copy(),
            'running_averages': self.running_avg.copy(),
            'total_predictions': sum(len(exp) for exp in self.expectations.values()),
            'total_errors_tracked': sum(len(err) for err in self.prediction_errors.values()),
            'recent_accuracy': 0.0,
            'improvement_trend': 'stable',
            'should_update_strategy': self.should_update_strategy()
        }
        
        # Calculate recent accuracy
        all_recent_errors = []
        for error_list in self.prediction_errors.values():
            all_recent_errors.extend(list(error_list)[-10:])
            
        if all_recent_errors:
            avg_error = np.mean([abs(e['relative_error']) for e in all_recent_errors])
            summary['recent_accuracy'] = 1.0 / (1.0 + avg_error)
            
        # Determine improvement trend
        if self.calibration['adaptability'] > 0.6:
            summary['improvement_trend'] = 'improving'
        elif self.calibration['adaptability'] < 0.4:
            summary['improvement_trend'] = 'declining'
            
        return summary
