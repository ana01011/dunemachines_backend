"""
Continuous neurochemical wave streaming via WebSocket
Hormones flow as continuous waves throughout the connection
"""

import asyncio
import json
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from app.neurochemistry.core.state_v2 import NeurochemicalState, Event
from app.neurochemistry.core.mood_emergence_v2 import MoodEmergence

@dataclass
class WavePacket:
    """Single packet of neurochemical wave data"""
    timestamp: float
    levels: Dict[str, float]
    baselines: Dict[str, float]
    waves: Dict[str, float]  # Distance from baseline
    mood: str
    mood_intensity: float
    triggers: list
    behavioral_tendencies: Dict[str, float]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))

class NeurochemicalWaveStreamer:
    """
    Streams continuous neurochemical updates via WebSocket
    Creates natural wave patterns that affect AI behavior
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state = NeurochemicalState()
        self.streaming = False
        self.update_frequency = 10  # Hz (10 updates per second)
        self.dt = 1.0 / self.update_frequency
        
        # Track generation for reward-prediction
        self.current_generation_id: Optional[str] = None
        self.generation_start_time: Optional[float] = None
        self.expected_quality: float = 0.7
        
    async def start_streaming(self, websocket):
        """Start streaming neurochemical waves"""
        self.streaming = True
        
        while self.streaming:
            try:
                # Update dynamics
                self.state.apply_dynamics(self.dt)
                
                # Create wave packet
                packet = self.create_wave_packet()
                
                # Send via WebSocket
                await websocket.send_json({
                    "type": "neuro_wave",
                    "data": asdict(packet)
                })
                
                # Wait for next update
                await asyncio.sleep(self.dt)
                
            except Exception as e:
                print(f"Streaming error: {e}")
                break
    
    def create_wave_packet(self) -> WavePacket:
        """Create current wave packet"""
        levels = {
            "dopamine": round(self.state.dopamine, 1),
            "cortisol": round(self.state.cortisol, 1),
            "adrenaline": round(self.state.adrenaline, 1),
            "serotonin": round(self.state.serotonin, 1),
            "oxytocin": round(self.state.oxytocin, 1)
        }
        
        baselines = {
            "dopamine": round(self.state.dopamine_baseline, 1),
            "cortisol": round(self.state.cortisol_baseline, 1),
            "adrenaline": round(self.state.adrenaline_baseline, 1),
            "serotonin": round(self.state.serotonin_baseline, 1),
            "oxytocin": round(self.state.oxytocin_baseline, 1)
        }
        
        waves = {
            h: round(levels[h] - baselines[h], 1) 
            for h in levels
        }
        
        mood = MoodEmergence.describe_emergent_state(self.state)
        tendencies = MoodEmergence.get_behavioral_tendencies(self.state)
        triggers = MoodEmergence.get_capability_triggers(self.state)
        
        # Calculate mood intensity (how strong the emotion is)
        mood_intensity = abs(sum(waves.values())) / 250  # Normalized
        
        return WavePacket(
            timestamp=time.time(),
            levels=levels,
            baselines=baselines,
            waves=waves,
            mood=mood,
            mood_intensity=min(1.0, mood_intensity),
            triggers=triggers,
            behavioral_tendencies=tendencies
        )
    
    async def process_user_message(self, message: str):
        """Process incoming user message - affects neurochemistry"""
        # Analyze message properties
        complexity = self._estimate_complexity(message)
        urgency = self._detect_urgency(message)
        emotional_content = self._detect_emotion(message)
        
        # Create event
        event = Event(
            type="user_message",
            complexity=complexity,
            urgency=urgency,
            emotional_content=emotional_content,
            novelty=0.3,  # Could analyze for novelty
            social_interaction=0.8  # User interaction is social
        )
        
        # Apply to state
        self.state.apply_dynamics(self.dt, event)
    
    def start_generation(self, generation_id: str, expected_difficulty: float = 0.5):
        """AI starts generating response - dopamine anticipation"""
        self.current_generation_id = generation_id
        self.generation_start_time = time.time()
        self.expected_quality = 1.0 - expected_difficulty
        
        # Start task (creates anticipatory dopamine rise)
        self.state.start_task(generation_id, expected_difficulty)
    
    def update_generation_progress(self, progress: float):
        """Update progress during generation"""
        if self.current_generation_id:
            self.state.update_progress(progress)
    
    def complete_generation(self, actual_quality: float):
        """
        Generation complete - trigger reward prediction error
        This is where dopamine spikes or crashes
        """
        if self.current_generation_id:
            # Complete task - dopamine responds to prediction error
            self.state.complete_task(actual_quality)
            
            # Reset tracking
            self.current_generation_id = None
            self.generation_start_time = None
    
    def retry_generation(self):
        """Retry after poor quality - adjusted expectations"""
        if self.current_generation_id:
            self.state.retry_task(self.current_generation_id)
    
    def _estimate_complexity(self, message: str) -> float:
        """Estimate message complexity"""
        # Simple heuristic - could use NLP
        word_count = len(message.split())
        has_code = "```" in message or "function" in message or "code" in message
        has_math = any(c in message for c in ["=", "+", "-", "*", "/", "∫", "∑"])
        
        complexity = min(1.0, word_count / 100)
        if has_code:
            complexity += 0.3
        if has_math:
            complexity += 0.2
            
        return min(1.0, complexity)
    
    def _detect_urgency(self, message: str) -> float:
        """Detect urgency in message"""
        urgent_words = ["urgent", "asap", "immediately", "now", "quickly", "hurry"]
        message_lower = message.lower()
        
        urgency = 0.3  # Base urgency
        for word in urgent_words:
            if word in message_lower:
                urgency += 0.2
        
        if "!" in message:
            urgency += 0.1
        if "HELP" in message or "ERROR" in message:
            urgency += 0.3
            
        return min(1.0, urgency)
    
    def _detect_emotion(self, message: str) -> float:
        """Detect emotional content"""
        emotional_words = ["feel", "happy", "sad", "angry", "love", "hate", "worried", "excited"]
        message_lower = message.lower()
        
        emotion = 0.2  # Base
        for word in emotional_words:
            if word in message_lower:
                emotion += 0.15
                
        if "!" in message:
            emotion += 0.1
        if "?" in message:
            emotion += 0.05
            
        return min(1.0, emotion)
    
    def stop_streaming(self):
        """Stop streaming"""
        self.streaming = False
    
    def get_current_prompt_injection(self) -> str:
        """Get current mood for prompt injection"""
        return MoodEmergence.create_natural_prompt(self.state)
    
    def get_current_state_summary(self) -> Dict:
        """Get full state summary"""
        return {
            "user_id": self.user_id,
            "mood": MoodEmergence.describe_emergent_state(self.state),
            "prompt": self.get_current_prompt_injection(),
            "levels": {
                "dopamine": self.state.dopamine,
                "cortisol": self.state.cortisol,
                "adrenaline": self.state.adrenaline,
                "serotonin": self.state.serotonin,
                "oxytocin": self.state.oxytocin
            },
            "triggers": MoodEmergence.get_capability_triggers(self.state),
            "behavioral": self.state.get_behavioral_parameters(),
            "stable": self.state.check_stability()
        }
