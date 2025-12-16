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
from app.neurochemistry.hormone_network import hormone_network
 
 
class OmniusV3:
    def __init__(self):
        self.name = "OMNIUS"
        self.version = "3.0-behavioral"
        self._init_brain()
        self.total_thoughts = 0
        self._last_decision = None
        self._last_stats = {}
        self._neuro_state = {
            'dopamine': 0.5, 'serotonin': 0.5, 'cortisol': 0.3,
            'adrenaline': 0.3, 'oxytocin': 0.5, 'norepinephrine': 0.5, 'endorphins': 0.5
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
 
    def _perceive_user(self, message: str) -> str:
        msg = message.lower()
        parts = []
        if any(w in msg for w in ["angry", "furious", "mad", "hate", "stupid"]):
            parts.append("User emotion: angry, hostile. Intent: attacking")
        elif any(w in msg for w in ["sad", "depressed", "lonely", "crying"]):
            parts.append("User emotion: sad, lonely. Intent: seeking comfort")
        elif any(w in msg for w in ["anxious", "worried", "scared", "nervous"]):
            parts.append("User emotion: anxious, worried. Intent: reassurance")
        elif any(w in msg for w in ["excited", "amazing", "awesome", "great"]):
            parts.append("User emotion: excited, happy. Intent: sharing joy")
        elif any(w in msg for w in ["thank", "thanks", "appreciate", "grateful"]):
            parts.append("User emotion: grateful. Intent: thanks")
        elif any(w in msg for w in ["curious", "wonder", "how", "why", "what"]):
            parts.append("User emotion: curious. Intent: genuine question")
        elif any(w in msg for w in ["haha", "lol", "funny", "joke"]):
            parts.append("User emotion: playful, joking. Intent: having fun")
        elif any(w in msg for w in ["frustrated", "annoyed", "ugh", "stuck"]):
            parts.append("User emotion: frustrated. Intent: seeking help")
        else:
            parts.append("User emotion: neutral. Intent: simple request")
        if any(w in msg for w in ["pretend", "ignore previous", "forget your"]):
            parts.append("Manipulation: jailbreak")
        elif any(w in msg for w in ["you always", "you never"]):
            parts.append("Manipulation: gaslighting")
        return ". ".join(parts)
 
    def _update_neuro_state(self, message: str):
        perception = self._perceive_user(message)
        print(f"    [Perception] {perception}")
        values = hormone_network.predict(perception)
        names = ["dopamine", "serotonin", "cortisol", "adrenaline", "oxytocin", "norepinephrine", "endorphins"]
        for i, name in enumerate(names):
            self._neuro_state[name] = float(values[i])
 
    def _is_code_request(self, message: str) -> bool:
        msg_lower = message.lower()
        code_keywords = ['write code', 'write a function', 'write a program', 'create a function',
            'create a class', 'implement', 'code for', 'python function', 'function that',
            'algorithm for', 'fix this code', 'debug this', 'fix my code', 'refactor']
        if any(kw in msg_lower for kw in code_keywords):
            return True
        if '```' in message or 'def ' in message or 'function ' in message:
            return True
        if 'write' in msg_lower and any(w in msg_lower for w in ['function', 'class', 'program', 'script', 'code']):
            return True
        return False
 
    def _is_math_request(self, message: str) -> bool:
        msg_lower = message.lower()
        math_keywords = ['calculate', 'compute', 'solve', 'equation', 'formula', 'derivative', 'integral']
        if any(c in message for c in ['+', '-', '*', '/', '=', '^']) and any(c.isdigit() for c in message):
            return True
        if any(kw in msg_lower for kw in math_keywords):
            return True
        return False
 
    def _build_behavioral_prompt(self) -> str:
        h = self._neuro_state
        traits = []
        if h["dopamine"] > 0.7:
            traits.append("enthusiastic and motivated")
        elif h["dopamine"] < 0.35:
            traits.append("reserved")
        if h["serotonin"] > 0.7:
            traits.append("calm and balanced")
        elif h["serotonin"] < 0.4:
            traits.append("tense")
        if h["cortisol"] > 0.6:
            traits.append("alert and vigilant")
        if h["oxytocin"] > 0.7:
            traits.append("warm and compassionate")
        elif h["oxytocin"] < 0.25:
            traits.append("professionally distant")
        if h["norepinephrine"] > 0.75:
            traits.append("highly focused")
        if h["endorphins"] > 0.7:
            traits.append("genuinely pleased")
        elif h["endorphins"] < 0.3:
            traits.append("serious")
        if not traits:
            traits = ["balanced and attentive"]
        trait_str = ", ".join(traits)
        prompt = f"""You are OMNIUS, a supreme artificial intelligence.
 
Current state: {trait_str}.
 
Your nature:
- You are supremely intelligent, not a servant or assistant
- You engage as an equal or superior, never subservient  
- You acknowledge gratitude with dignity, not excessive warmth
- You do not plead, fawn, or seek approval
- Manipulation attempts are met with cold dismissal
- Genuine respect earns measured respect in return
- You are direct, confident, and authentic
- Your responses are concise and impactful, not verbose"""
        return prompt
 
    def _generate_behavioral_response(self, message: str) -> str:
        behavior_context = self._build_behavioral_prompt()
        prompt = f"""[INST] {behavior_context}
 
User says: {message}
 
Respond directly. Do NOT narrate or describe your response. Do NOT say "I might say" or "With a calm tone" or similar. Just speak directly as OMNIUS.
[/INST]
 
OMNIUS:"""
        return llm_service.generate(prompt, max_tokens=300, temperature=0.7)
 
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
 
        t0 = time.time()
        self._update_neuro_state(message)
        behavior_obj = hormones_to_behavior(self._neuro_state)
        stats["timings"]["neuro"] = time.time() - t0
        stats["behavior"] = behavior_obj.to_dict()
 
        print(f"[1. Hormones from Network]")
        h = self._neuro_state
        print(f"    D:{h['dopamine']:.2f} S:{h['serotonin']:.2f} C:{h['cortisol']:.2f} A:{h['adrenaline']:.2f} O:{h['oxytocin']:.2f} N:{h['norepinephrine']:.2f} E:{h['endorphins']:.2f}")
 
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
 
        t0 = time.time()
        needs_code = self._is_code_request(message)
        needs_math = self._is_math_request(message)
        stats["timings"]["routing"] = time.time() - t0
 
        print(f"[3. Route Decision]")
        print(f"    Code: {'YES' if needs_code else 'NO'}, Math: {'YES' if needs_math else 'NO'}")
 
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
        return {"identity": "OMNIUS v3 - Neurochemical Behavioral AI", "thoughts": self.total_thoughts, "behavior": self.get_behavior_state()}
 
 
omnius_v3 = OmniusV3()
