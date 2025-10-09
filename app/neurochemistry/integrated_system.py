"""
Integrated Neurochemistry System for Omnius
Manages per-user neurochemical states with 3D dimensional emergence
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json

from app.neurochemistry.core.dimensional_emergence import (
    DimensionalEmergence,
    DimensionalPosition
)
from app.neurochemistry.core.state_v2_fixed import NeurochemicalState
from app.neurochemistry.core.event import Event


class NeurochemicalAgent:
    """Manages neurochemical state for a single user"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state = NeurochemicalState(user_id=user_id)
        self.current_position = DimensionalPosition(0.0, 0.2, 0.5)  # Neutral start
        self.last_update = datetime.now()
        self.message_history = []
        self.active_task = None
        
    def analyze_message(self, message: str) -> Dict:
        """Analyze message for emotional and cognitive content"""
        # Simple heuristic analysis (can be enhanced with NLP)
        message_lower = message.lower()
        
        # Complexity analysis
        word_count = len(message.split())
        complexity = min(1.0, word_count / 100)
        
        # Urgency detection
        urgency_words = ['urgent', 'asap', 'immediately', 'now', 'quick', 'fast', 'hurry']
        urgency = 0.7 if any(word in message_lower for word in urgency_words) else 0.2
        
        # Emotional content
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'angry', 'frustrated', 'annoyed', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        emotional_valence = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        # Threat detection
        threat_words = ['error', 'fail', 'crash', 'broken', 'critical', 'emergency']
        threat_level = 0.8 if any(word in message_lower for word in threat_words) else 0.1
        
        return {
            'complexity': complexity,
            'urgency': urgency,
            'emotional_valence': emotional_valence,
            'threat_level': threat_level,
            'intensity': (urgency + abs(emotional_valence) + threat_level) / 3
        }
    
    def process_message(self, message: str) -> Dict:
        """Process user message and update neurochemical state"""
        # Analyze message
        analysis = self.analyze_message(message)
        
        # Create event based on analysis
        event = Event(
            type="user_message",
            intensity=analysis['intensity'],
            complexity=analysis['complexity'],
            urgency=analysis['urgency'],
            emotional_content=analysis['emotional_valence'],
            threat_level=analysis['threat_level']
        )
        
        # Apply neurochemical dynamics
        self.state.process_message_event(event)
        
        # Convert to 3D position
        self.current_position = DimensionalEmergence.hormones_to_position(self.state)
        
        # Store in history
        self.message_history.append({
            'timestamp': datetime.now(),
            'message': message[:100],  # Store truncated version
            'analysis': analysis,
            'position': self.current_position.to_vector()
        })
        
        # Trim history to last 10 messages
        if len(self.message_history) > 10:
            self.message_history.pop(0)
        
        return {
            'position': self.current_position.to_vector(),
            'hormones': {
                'dopamine': self.state.dopamine,
                'cortisol': self.state.cortisol,
                'adrenaline': self.state.adrenaline,
                'serotonin': self.state.serotonin,
                'oxytocin': self.state.oxytocin
            },
            'behaviors': DimensionalEmergence.position_to_behavior(self.current_position),
            'prompt_injection': DimensionalEmergence.create_prompt_injection(
                self.current_position,
                self.state
            )
        }
    
    def start_generation(self, expected_quality: float = 0.7):
        """Called when starting to generate a response"""
        self.active_task = {
            'start_time': datetime.now(),
            'expected_quality': expected_quality
        }
        # Anticipation increases dopamine slightly
        self.state.dopamine = min(100, self.state.dopamine + 5)
    
    def complete_generation(self, actual_quality: float):
        """Called after generating a response"""
        if not self.active_task:
            return
        
        expected = self.active_task['expected_quality']
        # Reward prediction error affects dopamine
        error = actual_quality - expected
        self.state.dopamine += error * 20  # Scale the error
        self.state.dopamine = max(0, min(100, self.state.dopamine))
        
        # Success affects serotonin
        if actual_quality > 0.7:
            self.state.serotonin = min(100, self.state.serotonin + 3)
        
        self.active_task = None
    
    async def background_update(self):
        """Background process for homeostasis"""
        while True:
            # Apply natural decay towards baselines
            self.state.apply_homeostasis(0.1)
            
            # Update position
            self.current_position = DimensionalEmergence.hormones_to_position(self.state)
            
            # Sleep for update interval
            await asyncio.sleep(5)  # Update every 5 seconds


class NeurochemicalOrchestrator:
    """Manages neurochemical states for all users"""
    
    def __init__(self):
        self.agents: Dict[str, NeurochemicalAgent] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}
    
    def get_or_create_agent(self, user_id: str) -> NeurochemicalAgent:
        """Get existing agent or create new one"""
        if user_id not in self.agents:
            self.agents[user_id] = NeurochemicalAgent(user_id)
            # Start background update task
            task = asyncio.create_task(self.agents[user_id].background_update())
            self.background_tasks[user_id] = task
        return self.agents[user_id]
    
    def process_user_message(self, user_id: str, message: str) -> Dict:
        """Process message for specific user"""
        agent = self.get_or_create_agent(user_id)
        return agent.process_message(message)
    
    def get_user_state(self, user_id: str) -> Dict:
        """Get current state for user"""
        if user_id not in self.agents:
            return {
                'position': 'V+0.00A0.20D0.50',  # Neutral
                'status': 'uninitialized'
            }
        
        agent = self.agents[user_id]
        return {
            'position': agent.current_position.to_vector(),
            'hormones': {
                'dopamine': agent.state.dopamine,
                'cortisol': agent.state.cortisol,
                'adrenaline': agent.state.adrenaline,
                'serotonin': agent.state.serotonin,
                'oxytocin': agent.state.oxytocin
            },
            'behaviors': DimensionalEmergence.position_to_behavior(agent.current_position),
            'history_length': len(agent.message_history)
        }
    
    async def shutdown(self):
        """Cleanup background tasks"""
        for task in self.background_tasks.values():
            task.cancel()
        await asyncio.gather(*self.background_tasks.values(), return_exceptions=True)


# Global orchestrator instance
orchestrator = NeurochemicalOrchestrator()


def create_enhanced_prompt(user_id: str, message: str) -> str:
    """Main function to be called by Omnius"""
    result = orchestrator.process_user_message(user_id, message)
    
    # Return the message with the 3D vector prepended
    # The vector will naturally influence the AI's response style
    enhanced = f"{result['prompt_injection']} {message}"
    
    return enhanced


def get_user_neurochemical_state(user_id: str) -> Dict:
    """Get detailed neurochemical state for monitoring"""
    return orchestrator.get_user_state(user_id)