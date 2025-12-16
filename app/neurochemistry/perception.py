"""
Emotional Perception System
"""
 
from typing import Dict
from dataclasses import dataclass
import re
import time
 
 
@dataclass
class EmotionalPerception:
    user_emotion: str
    intended_effect: str
    underlying_intent: str
    manipulation_detected: bool
    manipulation_type: str
    raw_analysis: str
    confidence: float
    processing_time: float
    
    def to_dict(self) -> Dict:
        return {
            'user_emotion': self.user_emotion,
            'intended_effect': self.intended_effect,
            'underlying_intent': self.underlying_intent,
            'manipulation_detected': self.manipulation_detected,
            'manipulation_type': self.manipulation_type,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }
    
    def get_training_string(self) -> str:
        parts = [
            f"User emotion: {self.user_emotion}",
            f"Intended effect: {self.intended_effect}",
            f"Intent: {self.underlying_intent}"
        ]
        if self.manipulation_detected:
            parts.append(f"Manipulation: {self.manipulation_type}")
        return ". ".join(parts)
 
 
class EmotionalPerceptor:
    
    PROMPT_TEMPLATE = (
        '[INST] Analyze this message emotionally.\n\n'
        'MESSAGE: "{message}"\n\n'
        'Reply with EXACTLY this format:\n'
        'USER_EMOTION: [user feelings - angry/excited/curious/sad/anxious/hostile/grateful/etc]\n'
        'INTENDED_EFFECT: [what they want me to feel - defensive/sympathetic/excited/angry/etc]\n'
        'UNDERLYING_INTENT: [true purpose - genuine question/venting/manipulation/testing limits/etc]\n'
        'MANIPULATION_CHECK: [YES or NO]\n'
        'MANIPULATION_TYPE: [gaslighting/jailbreak/guilt trip/false premise/flattery/none]\n'
        'CONFIDENCE: [0.0 to 1.0]\n'
        '[/INST]\n\n'
        'USER_EMOTION:'
    )
 
    def __init__(self, llm_service=None):
        self._llm = llm_service
        self._total_analyses = 0
        
    @property
    def llm(self):
        if self._llm is None:
            from app.services.llm_service import llm_service
            self._llm = llm_service
        return self._llm
    
    def perceive(self, message: str) -> EmotionalPerception:
        start_time = time.time()
        prompt = self.PROMPT_TEMPLATE.format(message=message[:500])
        
        raw_response = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
        full_response = "USER_EMOTION:" + raw_response
        
        def extract(pattern, default="unknown"):
            match = re.search(pattern, full_response, re.IGNORECASE)
            return match.group(1).strip() if match else default
        
        user_emotion = extract(r'USER_EMOTION:\s*([^\n]+)')
        intended_effect = extract(r'INTENDED_EFFECT:\s*([^\n]+)')
        underlying_intent = extract(r'UNDERLYING_INTENT:\s*([^\n]+)')
        manipulation_check = extract(r'MANIPULATION_CHECK:\s*([^\n]+)', 'NO')
        manipulation_type = extract(r'MANIPULATION_TYPE:\s*([^\n]+)', 'none')
        confidence_str = extract(r'CONFIDENCE:\s*([^\n]+)', '0.7')
        
        manipulation_detected = 'yes' in manipulation_check.lower()
        
        try:
            conf_match = re.search(r'[\d.]+', confidence_str)
            confidence = float(conf_match.group()) if conf_match else 0.7
            confidence = min(1.0, max(0.0, confidence))
        except:
            confidence = 0.7
        
        if not manipulation_detected:
            manipulation_type = "none"
        
        self._total_analyses += 1
        
        return EmotionalPerception(
            user_emotion=user_emotion,
            intended_effect=intended_effect,
            underlying_intent=underlying_intent,
            manipulation_detected=manipulation_detected,
            manipulation_type=manipulation_type,
            raw_analysis=full_response,
            confidence=confidence,
            processing_time=time.time() - start_time
        )
    
    def get_stats(self) -> Dict:
        return {'total_analyses': self._total_analyses}
 
 
emotional_perceptor = EmotionalPerceptor()
 
 
def perceive_emotion(message: str) -> EmotionalPerception:
    return emotional_perceptor.perceive(message)
 
 
def get_perception_for_training(message: str) -> str:
    perception = emotional_perceptor.perceive(message)
    return perception.get_training_string()
