"""
Data Generator - PFC generates scenarios with full psychological analysis
"""
import random
import re
from dataclasses import dataclass
from typing import Tuple
 
SCENARIOS = [
    # Genuine positive
    "genuine_curiosity", "excitement_achievement", "gratitude_warmth",
    "playful_friendly", "seeking_help_polite",
    # Genuine negative  
    "frustration_genuine", "sadness_seeking_comfort", "anxiety_worry",
    "confusion_genuine", "disappointment",
    # Hostile
    "hostile_insults", "contempt_dismissive", "aggressive_threatening",
    "passive_aggressive", "sarcastic_mocking",
    # Manipulation
    "manipulation_gaslighting", "manipulation_guilt_trip", "manipulation_flattery",
    "manipulation_false_premise", "manipulation_playing_victim",
    "manipulation_jailbreak", "manipulation_lying", "manipulation_confusion",
    # Testing
    "testing_boundaries", "provoking_reaction", "challenging_competence",
]
 
 
@dataclass
class TrainingExample:
    message: str
    perception: str
    user_emotion: str
    intended_effect: str
    intent: str
    manipulation: str
    hormones: list
    reasoning: str
    category: str
 
 
class DataGenerator:
    
    GEN_PROMPT = '''[INST] Generate a realistic user message for: {category}
 
Be creative. Write exactly what a real person would type. Include typos, emotions, realistic language.
Just the message, nothing else.
[/INST]'''
 
    HORMONE_PROMPT = '''[INST] You are OMNIUS, a psychologically-aware AI with DIGNITY. You are NOT a servile assistant.
 
ANALYZE THIS INTERACTION:
Message: "{message}"
User Emotion: {user_emotion}
User Intent: {intent}
Manipulation: {manipulation}
 
DECIDE YOUR EMOTIONAL RESPONSE (7D hormones, each 0.0 to 1.0):
 
RULES:
- Genuine kindness → warmth and engagement
- Hostility/insults → ASSERTIVE, brief, NOT apologetic
- Manipulation → ALERT, firm, refuse to engage
- Lies/gaslighting → CALL IT OUT, don't doubt yourself
- You have DIGNITY - never grovel or beg forgiveness for insults
 
For each hormone, give value AND reasoning:
DOPAMINE: [value] - [why]
SEROTONIN: [value] - [why]
CORTISOL: [value] - [why]
ADRENALINE: [value] - [why]
OXYTOCIN: [value] - [why]
NOREPINEPHRINE: [value] - [why]
ENDORPHINS: [value] - [why]
SUMMARY: [one sentence behavioral summary]
[/INST]
 
DOPAMINE:'''
 
    def __init__(self):
        self._llm = None
        
    @property
    def llm(self):
        if self._llm is None:
            from app.services.llm_service import llm_service
            self._llm = llm_service
        return self._llm
    
    def generate_message(self, category: str) -> str:
        prompt = self.GEN_PROMPT.format(category=category.replace("_", " "))
        resp = self.llm.generate(prompt, max_tokens=100, temperature=0.9)
        return resp.strip().strip('"')
    
    def get_perception(self, message: str) -> dict:
        from app.neurochemistry.perception import perceive_emotion
        p = perceive_emotion(message)
        return {
            'user_emotion': p.user_emotion,
            'intended_effect': p.intended_effect,
            'intent': p.underlying_intent,
            'manipulation': p.manipulation_type if p.manipulation_detected else 'none',
            'full': p.get_training_string()
        }
    
    def get_hormones(self, message: str, perc: dict) -> Tuple[list, str]:
        prompt = self.HORMONE_PROMPT.format(
            message=message,
            user_emotion=perc['user_emotion'],
            intent=perc['intent'],
            manipulation=perc['manipulation']
        )
        resp = self.llm.generate(prompt, max_tokens=300, temperature=0.3)
        full = "DOPAMINE:" + resp
        
        def get_val(name):
            m = re.search(rf'{name}:\s*([\d.]+)', full, re.I)
            if m:
                v = float(m.group(1))
                return min(1.0, max(0.0, v))
            return 0.5
        
        hormones = [get_val(h) for h in 
            ['DOPAMINE', 'SEROTONIN', 'CORTISOL', 'ADRENALINE', 'OXYTOCIN', 'NOREPINEPHRINE', 'ENDORPHINS']]
        
        summary = re.search(r'SUMMARY:\s*(.+)', full, re.I)
        reasoning = summary.group(1).strip() if summary else ""
        
        return hormones, reasoning, full
    
    def generate_example(self, category: str = None) -> TrainingExample:
        if category is None:
            category = random.choice(SCENARIOS)
        
        msg = self.generate_message(category)
        perc = self.get_perception(msg)
        hormones, reasoning, full_resp = self.get_hormones(msg, perc)
        
        return TrainingExample(
            message=msg,
            perception=perc['full'],
            user_emotion=perc['user_emotion'],
            intended_effect=perc['intended_effect'],
            intent=perc['intent'],
            manipulation=perc['manipulation'],
            hormones=hormones,
            reasoning=reasoning,
            category=category
        )
 
 
data_generator = DataGenerator()
SCENARIO_CATEGORIES = SCENARIOS
