"""
Complete Integrated Neurochemical System
Brings together all components for production use
"""

import asyncio
import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from app.neurochemistry.core.state_v2 import NeurochemicalState, Event
from app.neurochemistry.core.dimensional_emergence import (
    DimensionalPosition,
    DimensionalEmergence,
    ResponseGenerator
)

@dataclass
class NeurochemicalAgent:
    """
    Complete neurochemical agent for a user
    Maintains state, processes events, generates prompts
    """
    
    user_id: str
    state: NeurochemicalState = field(default_factory=NeurochemicalState)
    position_history: List[DimensionalPosition] = field(default_factory=list)
    current_position: Optional[DimensionalPosition] = None
    
    def process_message(self, message: str) -> Dict:
        """
        Process user message and update neurochemistry
        Returns position and behavioral parameters
        """
        
        # Analyze message
        complexity = self._analyze_complexity(message)
        urgency = self._detect_urgency(message)
        emotional_content = self._detect_emotion(message)
        
        # Create event
        event = Event(
            type="user_message",
            complexity=complexity,
            urgency=urgency,
            emotional_content=emotional_content,
            social_interaction=0.8,  # User interaction is social
            novelty=0.3
        )
        
        # Update neurochemical state
        self.state.apply_dynamics(0.1, event)
        
        # Get new position
        self.current_position = DimensionalEmergence.hormones_to_position(self.state)
        self.position_history.append(self.current_position)
        
        # Get behavioral parameters
        behaviors = DimensionalEmergence.position_to_behavior(self.current_position)
        style = DimensionalEmergence.position_to_response_style(self.current_position)
        
        return {
            "position": self.current_position.to_vector(),
            "coordinates": {
                "valence": self.current_position.valence,
                "arousal": self.current_position.arousal,
                "dominance": self.current_position.dominance
            },
            "behaviors": behaviors,
            "style": style,
            "prompt_injection": f"[{self.current_position.to_vector()}]"
        }
    
    def start_generation(self, task_id: str, expected_difficulty: float):
        """AI starts generating - dopamine anticipation"""
        self.state.start_task(task_id, expected_difficulty)
        self.current_position = DimensionalEmergence.hormones_to_position(self.state)
    
    def complete_generation(self, actual_quality: float):
        """Generation complete - dopamine reward/punishment"""
        self.state.complete_task(actual_quality)
        self.current_position = DimensionalEmergence.hormones_to_position(self.state)
    
    def get_response_template(self, user_message: str) -> Dict:
        """Get template for how AI should respond"""
        if not self.current_position:
            self.current_position = DimensionalEmergence.hormones_to_position(self.state)
        
        return ResponseGenerator.generate_response_template(
            self.current_position,
            user_message
        )
    
    def _analyze_complexity(self, message: str) -> float:
        """Estimate message complexity"""
        words = len(message.split())
        has_code = any(x in message.lower() for x in ['code', 'function', 'algorithm', 'error'])
        has_technical = any(x in message.lower() for x in ['debug', 'implement', 'optimize'])
        
        base = min(1.0, words / 50)
        if has_code: base += 0.3
        if has_technical: base += 0.2
        
        return min(1.0, base)
    
    def _detect_urgency(self, message: str) -> float:
        """Detect urgency in message"""
        urgent_words = ['urgent', 'asap', 'now', 'immediately', 'critical', 'emergency']
        msg_lower = message.lower()
        
        urgency = 0.2
        for word in urgent_words:
            if word in msg_lower:
                urgency += 0.3
        
        if '!' in message:
            urgency += 0.1 * message.count('!')
        if message.isupper():
            urgency += 0.3
            
        return min(1.0, urgency)
    
    def _detect_emotion(self, message: str) -> float:
        """Detect emotional content"""
        emotional_words = ['feel', 'hate', 'love', 'angry', 'sad', 'happy', 
                          'frustrated', 'confused', 'worried', 'excited']
        msg_lower = message.lower()
        
        emotion = 0.1
        for word in emotional_words:
            if word in msg_lower:
                emotion += 0.2
                
        return min(1.0, emotion)
    
    def get_state_summary(self) -> Dict:
        """Get complete state summary"""
        if not self.current_position:
            self.current_position = DimensionalEmergence.hormones_to_position(self.state)
        
        return {
            "user_id": self.user_id,
            "position": self.current_position.to_vector(),
            "valence": self.current_position.valence,
            "arousal": self.current_position.arousal,
            "dominance": self.current_position.dominance,
            "hormones": {
                "dopamine": self.state.dopamine,
                "cortisol": self.state.cortisol,
                "adrenaline": self.state.adrenaline,
                "serotonin": self.state.serotonin,
                "oxytocin": self.state.oxytocin
            },
            "behavioral_params": DimensionalEmergence.position_to_behavior(self.current_position)
        }

class NeurochemicalOrchestrator:
    """
    Manages neurochemical agents for all users
    """
    
    def __init__(self):
        self.agents: Dict[str, NeurochemicalAgent] = {}
    
    def get_agent(self, user_id: str) -> NeurochemicalAgent:
        """Get or create agent for user"""
        if user_id not in self.agents:
            self.agents[user_id] = NeurochemicalAgent(user_id)
        return self.agents[user_id]
    
    def process_interaction(self, user_id: str, message: str) -> Dict:
        """
        Process complete interaction cycle
        Returns neurochemical state and prompt injection
        """
        
        agent = self.get_agent(user_id)
        
        # Process message
        response_params = agent.process_message(message)
        
        # Start generation
        complexity = agent._analyze_complexity(message)
        agent.start_generation(f"gen_{int(time.time())}", complexity)
        
        # Get response template
        template = agent.get_response_template(message)
        
        return {
            "prompt_injection": response_params["prompt_injection"],
            "position": response_params["position"],
            "behaviors": response_params["behaviors"],
            "template": template,
            "state": agent.get_state_summary()
        }
    
    def complete_generation(self, user_id: str, quality: float):
        """Mark generation as complete with quality score"""
        agent = self.get_agent(user_id)
        agent.complete_generation(quality)
        
        return agent.get_state_summary()

# Global orchestrator instance
neurochemical_orchestrator = NeurochemicalOrchestrator()

def create_enhanced_prompt(user_message: str, user_id: str) -> str:
    """
    Create prompt with neurochemical injection
    This is what the AI actually receives
    """
    
    result = neurochemical_orchestrator.process_interaction(user_id, user_message)
    
    # Simple injection - just the position vector
    # AI doesn't know what this means, just acts according to it
    enhanced_prompt = f"{result['prompt_injection']} {user_message}"
    
    return enhanced_prompt, result

def demonstrate_system():
    """Demonstrate the complete integrated system"""
    
    print("="*70)
    print("🧬 INTEGRATED NEUROCHEMICAL SYSTEM DEMONSTRATION")
    print("="*70)
    
    user_id = "test_user"
    
    # Test messages
    messages = [
        "Hello, can you help me understand recursion?",
        "URGENT! Production is DOWN! Database errors everywhere!",
        "I'm so frustrated, nothing is working...",
        "That's brilliant! Thank you so much!",
        "Can you write a complex sorting algorithm?"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n{'─'*60}")
        print(f"Message {i}: {message}")
        
        # Process interaction
        enhanced_prompt, result = create_enhanced_prompt(message, user_id)
        
        print(f"\n📍 Position: {result['position']}")
        print(f"   V={result['state']['valence']:+.2f} "
              f"A={result['state']['arousal']:.2f} "
              f"D={result['state']['dominance']:.2f}")
        
        print(f"\n🎯 Behavioral Changes:")
        behaviors = result['behaviors']
        print(f"   Speed: {behaviors['response_speed']:.2f}")
        print(f"   Directness: {behaviors['directness']:.2f}")
        print(f"   Empathy: {behaviors['empathy']:.2f}")
        print(f"   Patience: {behaviors['patience']:.2f}")
        
        print(f"\n💉 Enhanced Prompt for AI:")
        print(f"   {enhanced_prompt}")
        
        # Simulate quality score
        quality = 0.8 if i % 2 == 0 else 0.9
        
        # Complete generation
        state_after = neurochemical_orchestrator.complete_generation(user_id, quality)
        
        print(f"\n📊 After Generation:")
        print(f"   Dopamine: {state_after['hormones']['dopamine']:.1f}")
        if quality > 0.85:
            print("   ↑ Positive reinforcement")
        else:
            print("   ↓ Lower than expected")

if __name__ == "__main__":
    demonstrate_system()
