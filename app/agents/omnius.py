"""
OMNIUS v2 - PFC Thinks About Signals
Flow: Query → Thalamus → Signals → PFC THINKS → Decision
"""
from typing import Dict, Any, Tuple, List
import time
import re
import numpy as np
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.brain.thalamus import create_thalamus
from app.brain.areas import CodeArea, MathArea, MemoryArea
from app.brain.pretrain import ensure_pretrained


class OmniusOrchestrator:
    def __init__(self):
        self.name = "OMNIUS"
        self.version = "2.0"
        self._init_brain()
        self.total_thoughts = 0
        self._last_decision = None
        self._last_stats = {}

    def _init_brain(self):
        print("[OMNIUS] Initializing brain...")
        self.thalamus = create_thalamus(input_size=256, hidden_size=512, num_areas=5)
        ensure_pretrained(self.thalamus)
        self.brain_areas = {"code": CodeArea(), "math": MathArea(), "memory": MemoryArea()}
        print("[OMNIUS] Brain ready")

    def _encode_query(self, message: str) -> np.ndarray:
        encoding = np.zeros(256)
        for i, char in enumerate(message.encode()[:200]):
            encoding[i % 256] += (char - 128) / 128.0
        norm = np.linalg.norm(encoding)
        return encoding / norm if norm > 0 else encoding

    def _pfc_decide_from_signals(self, message: str, signals: Dict[str, float]) -> Dict[str, Any]:
        """PFC looks at signals and THINKS about what to do"""
        
        code_pct = int(signals.get("code", 0) * 100)
        math_pct = int(signals.get("math", 0) * 100)
        mem_pct = int(signals.get("memory", 0) * 100)
        
        prompt = f"""[INST] You are OMNIUS brain. Your thalamus sent these neural signals:
- Code area: {code_pct}% activation
- Math area: {math_pct}% activation  
- Memory area: {mem_pct}% activation

User request: "{message}"

Based on these brain signals and the request:
1. Should you use Code Cortex (DeepSeek) to generate code? Answer USE_CODE:YES or USE_CODE:NO
2. Should you use Math region for calculations? Answer USE_MATH:YES or USE_MATH:NO

Think step by step, then give your answers.
[/INST]"""

        response = llm_service.generate(prompt, max_tokens=150, temperature=0.3)
        
        # Parse PFC's decision
        resp_upper = response.upper()
        use_code = "USE_CODE:YES" in resp_upper or "USE_CODE: YES" in resp_upper
        use_math = "USE_MATH:YES" in resp_upper or "USE_MATH: YES" in resp_upper
        
        # If PFC didn't give clear answer, check for YES after CODE/MATH mention
        if "CODE" in resp_upper and "YES" in resp_upper and not "NO" in resp_upper.split("CODE")[1][:20]:
            use_code = True
        if "MATH" in resp_upper and "YES" in resp_upper and not "NO" in resp_upper.split("MATH")[1][:20]:
            use_math = True
            
        return {
            "use_code": use_code,
            "use_math": use_math,
            "pfc_reasoning": response.strip()
        }

    def _clean_code_output(self, output: str) -> str:
        """Extract clean code from DeepSeek output"""
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
        
        lines = output.split('\n')
        code_indicators = ('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return ')
        code_lines = [l for l in lines if l.strip().startswith(code_indicators)]
        
        if len(code_lines) > 2:
            return f"```python\n{output}\n```"
        
        return output

    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        start_time = time.time()
        stats = {"timings": {}, "signals": {}}
        
        print(f"\n{'='*60}")
        print(f"[OMNIUS] {message[:80]}...")
        print(f"{'='*60}")
        
        # STEP 1: Thalamus processes query → outputs signals
        t0 = time.time()
        query_signal = self._encode_query(message)
        thalamus_out = self.thalamus.route(query_signal)
        stats["timings"]["thalamus"] = time.time() - t0
        
        area_signals = {a.value: v for a, v in thalamus_out.activations.items()}
        code_sig = area_signals.get("code", 0)
        math_sig = area_signals.get("math", 0)
        mem_sig = area_signals.get("memory", 0)
        
        print(f"[1. Thalamus Signals]")
        print(f"    Code:   {int(code_sig*100)}%")
        print(f"    Math:   {int(math_sig*100)}%")
        print(f"    Memory: {int(mem_sig*100)}%")
        stats["signals"] = {"code": code_sig, "math": math_sig, "memory": mem_sig}
        
        # STEP 2: PFC receives signals and THINKS about what to do
        t0 = time.time()
        pfc_decision = self._pfc_decide_from_signals(message, area_signals)
        stats["timings"]["pfc_thinking"] = time.time() - t0
        
        print(f"[2. PFC Thinking] ({stats['timings']['pfc_thinking']:.1f}s)")
        print(f"    Reasoning: {pfc_decision['pfc_reasoning'][:100]}...")
        print(f"    Decision: code={pfc_decision['use_code']} math={pfc_decision['use_math']}")
        
        # STEP 3: Execute based on PFC decision
        regions_used = ['prefrontal_cortex']
        
        if pfc_decision["use_code"]:
            t0 = time.time()
            print(f"[3. Code Cortex] DeepSeek generating...")
            regions_used.append('code_cortex')
            
            try:
                raw_output = deepseek_coder.generate_code(message)
                stats["timings"]["deepseek"] = time.time() - t0
                code_block = self._clean_code_output(raw_output)
                print(f"    Done in {stats['timings']['deepseek']:.1f}s")
                response = f"Here's the solution:\n\n{code_block}"
            except Exception as e:
                print(f"    Error: {e}")
                response = llm_service.generate(f"[INST] {message} [/INST]", max_tokens=600)
        else:
            if pfc_decision["use_math"]:
                regions_used.append('math_region')
            
            t0 = time.time()
            response = llm_service.generate(f"[INST] {message} [/INST]", max_tokens=600, temperature=0.7)
            stats["timings"]["pfc_response"] = time.time() - t0
            print(f"[3. PFC Response] Direct ({stats['timings']['pfc_response']:.1f}s)")
        
        stats["timings"]["total"] = time.time() - start_time
        stats["pfc_decision"] = pfc_decision
        
        print(f"[DONE] Total: {stats['timings']['total']:.1f}s")
        print(f"{'='*60}\n")
        
        self._last_decision = {"areas": thalamus_out.active_areas, "signals": area_signals}
        self._last_stats = stats
        self.total_thoughts += 1
        
        return response, regions_used

    def learn(self, reward: float) -> Dict:
        if not self._last_decision:
            return {"error": "Nothing to learn"}
        result = self.thalamus.learn(reward, self._last_decision["areas"])
        from app.brain.pretrain import save_weights
        save_weights(self.thalamus)
        return {"reward": reward, "saved": True}

    def get_last_stats(self) -> Dict:
        return self._last_stats

    def get_status(self) -> Dict:
        return {
            "identity": "OMNIUS v2 - PFC Thinks",
            "thoughts": self.total_thoughts,
            "status": "operational"
        }


omnius = OmniusOrchestrator()
