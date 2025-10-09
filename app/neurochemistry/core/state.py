"""
Main neurochemical state management system
"""
from typing import Dict, Optional, List, Any
import asyncio
import yaml
import time
from datetime import datetime
import numpy as np
import logging
from pathlib import Path

from .constants import Hormone, EventType
from .event import Event
from ..hormones import Dopamine, Cortisol, Adrenaline, Serotonin, Oxytocin
from ..processors import EventProcessor, BaselineAdapter, StabilityController
from ..learning import PatternRecognizer, ExpectationLearner

logger = logging.getLogger(__name__)

class NeurochemicalState:
    """
    Central manager for all neurochemical processes
    """
    
    def __init__(self, user_id: str, config_path: str = None):
        """
        Initialize neurochemical state
        
        Args:
            user_id: User identifier
            config_path: Path to configuration file
        """
        self.user_id = user_id
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'neurochemical_config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['neurochemistry']
            
        # Initialize hormones
        self.hormones = {
            'dopamine': Dopamine(self.config['hormones']['dopamine']),
            'cortisol': Cortisol(self.config['hormones']['cortisol']),
            'adrenaline': Adrenaline(self.config['hormones']['adrenaline']),
            'serotonin': Serotonin(self.config['hormones']['serotonin']),
            'oxytocin': Oxytocin(self.config['hormones']['oxytocin'])
        }
        
        # Initialize processors
        self.event_processor = EventProcessor(self)
        self.baseline_adapter = BaselineAdapter(self)
        self.stability_controller = StabilityController(self)
        
        # Initialize learning components
        self.pattern_recognizer = PatternRecognizer(user_id)
        self.expectation_learner = ExpectationLearner(user_id)
        
        # State tracking
        self.last_update = time.time()
        self.update_task = None
        self.is_running = False
        
        # Metrics
        self.total_events_processed = 0
        self.total_updates = 0
        self.start_time = datetime.now()
        
        # State history for analysis
        self.state_history = []
        self.max_history_size = 1000
        
        logger.info(f"Initialized neurochemical state for user {user_id}")
    
    async def start(self):
        """Start the neurochemical update loop"""
        if self.is_running:
            logger.warning("Neurochemical state already running")
            return
            
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info(f"Started neurochemical state for user {self.user_id}")
    
    async def stop(self):
        """Stop the neurochemical update loop"""
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped neurochemical state for user {self.user_id}")
    
    async def _update_loop(self):
        """Continuous state update loop"""
        while self.is_running:
            try:
                dt = time.time() - self.last_update
                
                # Update all hormones
                for hormone in self.hormones.values():
                    hormone.update(dt, other_hormones=self.hormones)
                
                # Check and apply stability corrections
                self.stability_controller.check_and_correct()
                
                # Adapt baselines if needed
                if self.baseline_adapter.should_adapt():
                    shifts = self.baseline_adapter.adapt_all()
                    logger.debug(f"Baseline shifts: {shifts}")
                
                # Record state
                self._record_state()
                
                self.last_update = time.time()
                self.total_updates += 1
                
                # Sleep based on configuration
                await asyncio.sleep(self.config['update_frequency'])
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
    
    async def process_event(self, event: Event) -> Dict[str, Any]:
        """
        Process an event through the neurochemical system
        
        Args:
            event: Event to process
            
        Returns:
            Response dictionary with mood and behavioral changes
        """
        # Learn from event
        self.pattern_recognizer.observe(event)
        
        # Process through hormones
        hormone_responses = await self.event_processor.process(event)
        
        # Apply responses to hormones
        dt = time.time() - self.last_update
        for hormone_name, response in hormone_responses.items():
            if hormone_name in self.hormones:
                hormone = self.hormones[hormone_name]
                # Apply response as a change
                hormone.current_level += response
                # Ensure bounds
                hormone._apply_bounds()
        
        # Check stability after event
        interventions = self.stability_controller.check_and_correct()
        
        # Get current mood and behavior
        mood = self.get_mood()
        behavior = self.get_behavioral_parameters()
        
        self.total_events_processed += 1
        
        response = {
            'mood': mood,
            'behavior': behavior,
            'hormone_responses': hormone_responses,
            'interventions': interventions,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Processed event {event.type} for user {self.user_id}")
        return response
    
    def get_mood(self) -> Dict[str, Any]:
        """
        Get current mood based on neurochemical state
        
        Returns:
            Mood dictionary
        """
        # Calculate composite mood metrics
        dopamine = self.hormones['dopamine']
        cortisol = self.hormones['cortisol']
        adrenaline = self.hormones['adrenaline']
        serotonin = self.hormones['serotonin']
        oxytocin = self.hormones['oxytocin']
        
        # Arousal (energy level)
        arousal = (adrenaline.get_amplitude() / 50 + 
                  abs(dopamine.get_amplitude()) / 100)
        arousal = np.clip(arousal, -1, 1)
        
        # Valence (positive/negative emotion)
        valence = (dopamine.get_amplitude() / 50 - 
                  cortisol.get_amplitude() / 60 +
                  serotonin.current_level / 100)
        valence = np.clip(valence, -1, 1)
        
        # Focus (attention level)
        focus = 1.0 - abs(cortisol.current_level - 40) / 40
        if cortisol.current_level > 60:  # High stress reduces focus
            focus *= 0.7
        focus = np.clip(focus, 0, 1)
        
        # Confidence
        confidence = serotonin.get_confidence_modifier() if hasattr(serotonin, 'get_confidence_modifier') else 0.5
        
        # Social warmth
        warmth = oxytocin.current_level / 60 if oxytocin.current_level > 30 else 0.3
        warmth = np.clip(warmth, 0, 1)
        
        mood = {
            'arousal': round(arousal, 3),
            'valence': round(valence, 3),
            'focus': round(focus, 3),
            'confidence': round(confidence, 3),
            'warmth': round(warmth, 3),
            'description': self._get_mood_description(arousal, valence, focus),
            'color_temp': self._valence_to_color(valence),
            'energy_level': self._arousal_to_energy(arousal)
        }
        
        return mood
    
    def get_behavioral_parameters(self) -> Dict[str, float]:
        """
        Get behavioral parameters based on neurochemical state
        
        Returns:
            Behavioral parameters dictionary
        """
        config = self.config['behavior']
        
        # Get hormone states
        dopamine = self.hormones['dopamine']
        cortisol = self.hormones['cortisol']
        adrenaline = self.hormones['adrenaline']
        serotonin = self.hormones['serotonin']
        
        # Planning depth
        planning_base = config['planning_depth']['base']
        planning_depth = planning_base
        
        # Cortisol increases planning
        if hasattr(cortisol, 'get_attention_multiplier'):
            planning_depth *= cortisol.get_attention_multiplier()
        else:
            planning_depth *= (1 + cortisol.get_amplitude() / 100 * config['planning_depth']['cortisol_factor'])
        
        # Adrenaline decreases planning (urgency)
        planning_depth *= (1 - adrenaline.get_amplitude() / 100 * config['planning_depth'].get('adrenaline_factor', 0.015))
        
        planning_depth = np.clip(planning_depth, 
                                config['planning_depth']['min'], 
                                config['planning_depth']['max'])
        
        # Risk tolerance
        risk_base = config['risk_tolerance']['base']
        risk_tolerance = risk_base
        
        # Dopamine increases risk tolerance
        risk_tolerance += dopamine.get_amplitude() / 100 * config['risk_tolerance']['dopamine_factor']
        
        # Cortisol decreases risk tolerance
        risk_tolerance += cortisol.get_amplitude() / 100 * config['risk_tolerance']['cortisol_factor']
        
        # Serotonin provides balanced risk taking
        if serotonin.current_level > 50:
            risk_tolerance += (serotonin.current_level - 50) / 50 * config['risk_tolerance']['serotonin_factor']
        
        risk_tolerance = np.clip(risk_tolerance,
                                config['risk_tolerance']['min'],
                                config['risk_tolerance']['max'])
        
        # Processing speed
        speed_base = config['processing_speed']['base']
        
        # Get adrenaline performance modifiers
        if hasattr(adrenaline, 'get_performance_modifiers'):
            perf_mods = adrenaline.get_performance_modifiers()
            processing_speed = speed_base * perf_mods['processing_speed']
        else:
            processing_speed = speed_base * (1 + adrenaline.get_amplitude() / 100 * config['processing_speed']['adrenaline_factor'])
        
        processing_speed = np.clip(processing_speed,
                                  config['processing_speed']['min'],
                                  config['processing_speed']['max'])
        
        # Confidence
        confidence_base = config['confidence']['base']
        
        # Get serotonin confidence modifier
        if hasattr(serotonin, 'get_confidence_modifier'):
            confidence = confidence_base * serotonin.get_confidence_modifier()
        else:
            confidence = confidence_base + serotonin.current_level / 100 * config['confidence']['serotonin_factor']
        
        confidence = np.clip(confidence,
                            config['confidence']['min'],
                            config['confidence']['max'])
        
        parameters = {
            'planning_depth': round(planning_depth, 1),
            'risk_tolerance': round(risk_tolerance, 3),
            'processing_speed': round(processing_speed, 3),
            'confidence': round(confidence, 3),
            'should_clarify': cortisol.should_request_clarification() if hasattr(cortisol, 'should_request_clarification') else False,
            'should_help': oxytocin.should_offer_help(self.user_id) if hasattr(oxytocin, 'should_offer_help') else False
        }
        
        return parameters
    
    def _get_mood_description(self, arousal: float, valence: float, focus: float) -> str:
        """Generate natural language mood description"""
        descriptions = []
        
        # Arousal description
        if arousal > 0.5:
            descriptions.append("energetic")
        elif arousal < -0.5:
            descriptions.append("tired")
        else:
            descriptions.append("calm")
            
        # Valence description
        if valence > 0.5:
            descriptions.append("positive")
        elif valence < -0.5:
            descriptions.append("stressed")
        else:
            descriptions.append("neutral")
            
        # Focus description
        if focus > 0.7:
            descriptions.append("focused")
        elif focus < 0.3:
            descriptions.append("scattered")
            
        return ", ".join(descriptions)
    
    def _valence_to_color(self, valence: float) -> str:
        """Convert valence to color temperature"""
        if valence > 0.5:
            return "warm"  # Positive
        elif valence < -0.5:
            return "cool"  # Negative
        else:
            return "neutral"
    
    def _arousal_to_energy(self, arousal: float) -> str:
        """Convert arousal to energy level description"""
        if arousal > 0.5:
            return "high"
        elif arousal < -0.5:
            return "low"
        else:
            return "moderate"
    
    def _record_state(self):
        """Record current state for history"""
        if len(self.state_history) >= self.max_history_size:
            self.state_history.pop(0)
            
        state_record = {
            'timestamp': datetime.now(),
            'hormones': {
                name: {
                    'level': hormone.current_level,
                    'baseline': hormone.baseline,
                    'amplitude': hormone.get_amplitude()
                }
                for name, hormone in self.hormones.items()
            },
            'mood': self.get_mood(),
            'stability': self.stability_controller.stability_score if hasattr(self.stability_controller, 'stability_score') else 1.0
        }
        
        self.state_history.append(state_record)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        summary = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'total_events': self.total_events_processed,
            'total_updates': self.total_updates,
            'hormones': {},
            'mood': self.get_mood(),
            'behavior': self.get_behavioral_parameters(),
            'stability': self.stability_controller.get_stability_report(),
            'baselines': self.baseline_adapter.get_adaptation_summary(),
            'learning': {
                'patterns': self.pattern_recognizer.get_user_profile(),
                'expectations': self.expectation_learner.get_learning_summary()
            }
        }
        
        # Add hormone summaries
        for name, hormone in self.hormones.items():
            summary['hormones'][name] = hormone.get_state_summary()
            
        return summary
    
    async def reset(self):
        """Reset neurochemical state to baseline"""
        logger.info(f"Resetting neurochemical state for user {self.user_id}")
        
        # Reset all hormones
        for hormone in self.hormones.values():
            hormone.reset()
            
        # Clear history
        self.state_history.clear()
        
        # Reset counters
        self.total_events_processed = 0
        self.total_updates = 0
        self.start_time = datetime.now()
        
        logger.info("Neurochemical state reset complete")
