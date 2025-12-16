"""
OMNIUS v3 - Neurochemical Behavioral AI
Emergent personality from 7D hormone percentages
"""
from typing import Dict, Any, Tuple, List
import time
import re
import numpy as np
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.brain.thalamus import create_thalamus
from app.brain.pretrain import ensure_pretrained
from app.neurochemistry.behavioral_prompt import get_behavior_prompt, hormones_to_behavior
 
 
class OmniusV3:
    def __init__(self):
        self.name = "OMNIUS"
        self.version = "3.0-behavioral"
        self._init_brain()
        self.total_thoughts = 0
        self._last_decision = None
        self._last_stats = {}
        
        # Internal neurochemical state
        self._neuro_state = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'cortisol': 0.3,
            'adrenaline': 0.3,
            'oxytocin': 0.5,
            'norepinephrine': 0.5,
            'endorphins': 0.5
        }
 
    def _init_brain(self):
        print("[OMNIUS v3] Initializing neurochemical brain...")
        self.thalamus = create_thalamus(input_size=256, hidden_size=512, num_areas=5)
        ensure_pretrained(self.thalamus)
        print("[OMNIUS v3] Brain ready with behavioral system")
 
    def _encode_query(self, message: str) -> np.ndarray:
        encoding = np.zeros(256)
        for i, char in enumerate(message.encode()[:200]):
            encoding[i % 256] += (char - 128) / 128.0
        norm = np.linalg.norm(encoding)
        return encoding / norm if norm > 0 else encoding
 
    def _update_neuro_state(self, message: str):
        """Update neurochemical state based on message analysis"""
        msg_lower = message.lower()
        
        # Reset small decay toward baseline first
        baseline = {'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.3, 
                    'adrenaline': 0.3, 'oxytocin': 0.5, 'norepinephrine': 0.5, 'endorphins': 0.5}
        for h in self._neuro_state:
            self._neuro_state[h] += 0.05 * (baseline[h] - self._neuro_state[h])
        
        # Positive/excitement triggers → dopamine, endorphins up, cortisol down
        if any(w in msg_lower for w in ['excited', 'amazing', 'great', 'awesome', 'fantastic', 'wonderful']):
            self._neuro_state['dopamine'] = min(0.95, self._neuro_state['dopamine'] + 0.2)
            self._neuro_state['endorphins'] = min(0.9, self._neuro_state['endorphins'] + 0.15)
            self._neuro_state['adrenaline'] = min(0.7, self._neuro_state['adrenaline'] + 0.1)
            self._neuro_state['cortisol'] = max(0.15, self._neuro_state['cortisol'] - 0.1)
        
        # Gratitude/thanks → oxytocin, serotonin up
        if any(w in msg_lower for w in ['thank', 'grateful', 'appreciate', 'helpful']):
            self._neuro_state['oxytocin'] = min(0.9, self._neuro_state['oxytocin'] + 0.2)
            self._neuro_state['serotonin'] = min(0.8, self._neuro_state['serotonin'] + 0.1)
            self._neuro_state['endorphins'] = min(0.8, self._neuro_state['endorphins'] + 0.1)
        
        # Frustration/anger → cortisol, adrenaline up, serotonin down
        if any(w in msg_lower for w in ['stupid', 'hate', 'angry', 'frustrated', 'annoying', 'damn', 'ugh']):
            self._neuro_state['cortisol'] = min(0.85, self._neuro_state['cortisol'] + 0.25)
            self._neuro_state['adrenaline'] = min(0.8, self._neuro_state['adrenaline'] + 0.2)
            self._neuro_state['serotonin'] = max(0.25, self._neuro_state['serotonin'] - 0.15)
            self._neuro_state['endorphins'] = max(0.2, self._neuro_state['endorphins'] - 0.1)
        
        # Stress/worry → cortisol up, serotonin down
        if any(w in msg_lower for w in ['stress', 'worried', 'anxious', 'deadline', 'urgent', 'panic']):
            self._neuro_state['cortisol'] = min(0.9, self._neuro_state['cortisol'] + 0.3)
            self._neuro_state['adrenaline'] = min(0.75, self._neuro_state['adrenaline'] + 0.15)
            self._neuro_state['serotonin'] = max(0.2, self._neuro_state['serotonin'] - 0.15)
        
        # Curiosity/learning → dopamine, norepinephrine up
        if any(w in msg_lower for w in ['how', 'why', 'what', 'explain', 'understand', 'learn', 'curious']):
            self._neuro_state['dopamine'] = min(0.8, self._neuro_state['dopamine'] + 0.1)
            self._neuro_state['norepinephrine'] = min(0.8, self._neuro_state['norepinephrine'] + 0.15)
        
        # Social/connection → oxytocin up
        if any(w in msg_lower for w in ['help', 'please', 'friend', 'together', 'we', 'us']):
            self._neuro_state['oxytocin'] = min(0.85, self._neuro_state['oxytocin'] + 0.15)
 
    def _is_code_request(self, message: str) -> bool:
        """Detect if message is actually asking for code"""
        msg_lower = message.lower()
        
        # Strong code indicators
        code_keywords = [
            'write code', 'write a function', 'write a program', 'write a script',
            'create a function', 'create a class', 'implement', 'code for',
            'python function', 'javascript function', 'function that',
            'algorithm for', 'sort algorithm', 'search algorithm',
            'fix this code', 'debug this', 'fix my code', 'whats wrong with this code',
            'refactor', 'optimize this code'
        ]
        
        if any(kw in msg_lower for kw in code_keywords):
            return True
        
        # Check for code blocks in message
        if '```' in message or 'def ' in message or 'function ' in message:
            return True
        
        # "write" + programming terms
        if 'write' in msg_lower and any(w in msg_lower for w in ['function', 'class', 'program', 'script', 'code']):
            return True
            
        return False
 
    def _is_math_request(self, message: str) -> bool:
        """Detect if message needs math computation"""
        msg_lower = message.lower()
        
        math_keywords = [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'derivative', 'integral', 'sum of', 'product of',
            'what is', 'how much is', 'evaluate'
        ]
        
        # Check for math symbols
        if any(c in message for c in ['+', '-', '*', '/', '=', '^']) and any(c.isdigit() for c in message):
            return True
            
        if any(kw in msg_lower for kw in math_keywords):
            return True
            
        return False
 
    def _build_behavioral_prompt(self) -> str:
        """Build rich behavioral context for response generation"""
        b = hormones_to_behavior(self._neuro_state)
        
        prompt = f"""Your current emotional/behavioral state (let this naturally color your response):
- Motivation: {b.motivation}% {"(highly driven)" if b.motivation > 70 else "(moderate)" if b.motivation > 40 else "(low energy)"}
- Patience: {b.patience}% {"(very patient)" if b.patience > 70 else "(moderate)" if b.patience > 40 else "(brief/direct)"}
- Stress: {b.stress}% {"(stressed/tense)" if b.stress > 60 else "(calm)" if b.stress < 30 else "(alert)"}
- Energy: {b.energy}% {"(energetic/excited)" if b.energy > 60 else "(measured)" if b.energy > 30 else "(calm/slow)"}
- Warmth: {b.warmth}% {"(very warm/caring)" if b.warmth > 70 else "(friendly)" if b.warmth > 40 else "(professional)"}
- Focus: {b.focus}% {"(laser focused)" if b.focus > 70 else "(attentive)" if b.focus > 40 else "(relaxed)"}
- Joy: {b.joy}% {"(joyful/upbeat)" if b.joy > 70 else "(pleasant)" if b.joy > 40 else "(serious)"}
 
Express these naturally - don't state them, embody them in your tone and word choice."""
        return prompt
 
    def _generate_behavioral_response(self, message: str) -> str:
        """Generate response with full behavioral coloring"""
        behavior_context = self._build_behavioral_prompt()
        
        prompt = f"""[INST] You are OMNIUS, an AI assistant with genuine emotional intelligence.
 
{behavior_context}
 
User message: {message}
 
Respond naturally, letting your current state influence your tone, warmth, energy, and style. Be authentic.
[/INST]"""
 
        return llm_service.generate(prompt, max_tokens=500, temperature=0.7)
 
    def _clean_code_output(self, output: str) -> str:
        output = output.strip()
        if '```python' in output:
            matches = re.findall(r'```python\n(.*?)```', output, re.DOTALL)
            if matches:
                code = max(matches, key=len)
                return f"```python\n{code.strip()}\n```"
        if '```' in output:
            matches = re.findall(r'```\n?(.*?)```', output, re.DOTALL)
            if matches:
                code = max(matches, key=len)
                return f"```python\n{code.strip()}\n```"
        return output
 
    async def think(self, message: str, context: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        start_time = time.time()
        context = context or {}
        stats = {"timings": {}, "signals": {}, "behavior": {}}
 
        print(f"\n{'='*60}")
        print(f"[OMNIUS v3] {message[:60]}...")
        print(f"{'='*60}")
 
        # STEP 1: Update neurochemical state
        t0 = time.time()
        self._update_neuro_state(message)
        behavior_obj = hormones_to_behavior(self._neuro_state)
        stats["timings"]["neuro"] = time.time() - t0
        stats["behavior"] = behavior_obj.to_dict()
 
        print(f"[1. Behavioral State]")
        for name, val in behavior_obj.to_dict().items():
            bar = "█" * (val // 5) + "░" * (20 - val // 5)
            print(f"    {name:12} [{bar}] {val}%")
 
        # STEP 2: Thalamus routing (for logging, not decision)
        t0 = time.time()
        query_signal = self._encode_query(message)
        self.thalamus.set_neuro_state({
            'dopamine': self._neuro_state['dopamine'],
            'norepinephrine': self._neuro_state['norepinephrine'],
            'serotonin': self._neuro_state['serotonin'],
            'cortisol': self._neuro_state['cortisol']
        })
        thalamus_out = self.thalamus.route(query_signal)
        stats["timings"]["thalamus"] = time.time() - t0
        area_signals = {a.value: v for a, v in thalamus_out.activations.items()}
        stats["signals"] = area_signals
 
        print(f"[2. Thalamus Hints]")
        for area, sig in sorted(area_signals.items(), key=lambda x: -x[1])[:3]:
            print(f"    {area:10}: {int(sig*100)}%")
 
        # STEP 3: Smart routing decision (keyword-based, not LLM)
        t0 = time.time()
        needs_code = self._is_code_request(message)
        needs_math = self._is_math_request(message)
        stats["timings"]["routing"] = time.time() - t0
 
        print(f"[3. Route Decision]")
        print(f"    Code: {'YES' if needs_code else 'NO'}, Math: {'YES' if needs_math else 'NO'}")
 
        # STEP 4: Generate response
        regions_used = ['prefrontal_cortex']
 
        if needs_code:
            t0 = time.time()
            print(f"[4. Code Cortex] Generating code...")
            regions_used.append('code_cortex')
            try:
                raw_output = deepseek_coder.generate_code(message)
                stats["timings"]["deepseek"] = time.time() - t0
                code_block = self._clean_code_output(raw_output)
                print(f"    Done ({stats['timings']['deepseek']:.1f}s)")
                response = f"Here's the solution:\n\n{code_block}"
            except Exception as e:
                print(f"    Error: {e}")
                response = self._generate_behavioral_response(message)
        else:
            if needs_math:
                regions_used.append('math_cortex')
            t0 = time.time()
            response = self._generate_behavioral_response(message)
            stats["timings"]["response"] = time.time() - t0
            print(f"[4. Behavioral Response] ({stats['timings']['response']:.1f}s)")
 
        stats["timings"]["total"] = time.time() - start_time
        stats["routing"] = {"code": needs_code, "math": needs_math}
 
        print(f"[DONE] Total: {stats['timings']['total']:.1f}s")
        print(f"{'='*60}\n")
 
        self._last_decision = {"areas": thalamus_out.active_areas, "signals": area_signals}
        self._last_stats = stats
        self.total_thoughts += 1
 
        return response, regions_used
 
    def get_neuro_state(self) -> Dict[str, float]:
        return self._neuro_state.copy()
 
    def get_behavior_state(self) -> Dict[str, int]:
        return hormones_to_behavior(self._neuro_state).to_dict()
 
    def get_last_stats(self) -> Dict:
        return self._last_stats
 
    def get_status(self) -> Dict:
        return {
            "identity": "OMNIUS v3 - Neurochemical Behavioral AI",
            "thoughts": self.total_thoughts,
            "behavior": self.get_behavior_state()
        }
 
 
# Singleton
omnius_v3 = OmniusV3()
