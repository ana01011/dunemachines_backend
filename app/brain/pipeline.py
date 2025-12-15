"""
OMNIUS v3 - PFC-Planned Pipeline
"""
import asyncio
import time
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
 
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
 
 
@dataclass
class PFCPlan:
    areas_needed: List[str]
    execution_order: str
    reasoning: str
    direct_answer: Optional[str] = None
 
 
@dataclass
class AreaResult:
    area_name: str
    output: str
    processing_time: float = 0.0
    success: bool = True
 
 
@dataclass
class PipelineResult:
    query: str
    final_response: str
    pfc_plan: PFCPlan
    area_results: List[AreaResult]
    total_time: float
    thalamus_signals: Dict[str, float]
 
 
class BrainPipeline:
    def __init__(self):
        self.total_pipelines = 0
        self.area_usage = {}
    
    async def process(self, query: str, signals: Dict[str, float], context: Dict[str, Any] = None) -> PipelineResult:
        start_time = time.time()
        context = context or {}
        
        print(f"\n{'='*60}")
        print(f"[Pipeline] {query[:60]}...")
        print(f"{'='*60}")
        
        print(f"\n[Thalamus Hints]")
        for area, sig in sorted(signals.items(), key=lambda x: x[1], reverse=True):
            print(f"    {area:10}: {int(sig*100):3}%")
        
        print(f"\n[PFC Planning]...", end=" ", flush=True)
        t0 = time.time()
        plan = await self._pfc_plan(query, signals)
        print(f"({time.time()-t0:.1f}s)")
        print(f"    Decision: {plan.execution_order} -> {plan.areas_needed}")
        print(f"    Reason: {plan.reasoning[:60]}...")
        
        area_results = []
        final_response = ""
        
        if plan.execution_order == "direct":
            print(f"\n[Direct Answer]")
            final_response = plan.direct_answer if plan.direct_answer else await self._language(query)
            
        elif plan.execution_order == "single":
            area = plan.areas_needed[0]
            print(f"\n[Single: {area}]...", end=" ", flush=True)
            result = await self._process_area(area, query, context)
            area_results.append(result)
            print(f"done ({result.processing_time:.1f}s)")
            
            if area == "code":
                final_response = f"Here's the solution:\n\n{result.output}"
            else:
                final_response = result.output
                
        elif plan.execution_order == "sequential":
            print(f"\n[Sequential: {' -> '.join(plan.areas_needed)}]")
            ctx = dict(context)
            
            for area in plan.areas_needed:
                print(f"    -> {area}...", end=" ", flush=True)
                result = await self._process_area(area, query, ctx)
                area_results.append(result)
                if result.success:
                    ctx[f"{area}_output"] = result.output
                print(f"done ({result.processing_time:.1f}s)")
            
            final_response = self._combine(area_results, ctx)
        
        total_time = time.time() - start_time
        self.total_pipelines += 1
        
        print(f"\n[Done] {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return PipelineResult(
            query=query,
            final_response=final_response,
            pfc_plan=plan,
            area_results=area_results,
            total_time=total_time,
            thalamus_signals=signals
        )
    
    async def _pfc_plan(self, query: str, signals: Dict[str, float]) -> PFCPlan:
        top_area = max(signals.items(), key=lambda x: x[1])
        hints = ", ".join([f"{a}:{int(s*100)}%" for a, s in sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]])
        
        prompt = f"""[INST] Decide what brain areas to use. Be minimal - don't use areas unnecessarily.
 
Query: "{query}"
Signals: {hints}
 
Areas: code (programming), math (calculations), physics (science), language (explanations)
 
Rules:
- "Write/create/implement code/function/class" -> just code
- "Calculate/solve/compute" with no code needed -> just math  
- "Explain/describe/what is" -> direct answer or language
- Only use multiple areas if TRULY needed (e.g., "derive formula AND code it")
 
Reply with ONLY one line in format:
ORDER:AREA1,AREA2:reason
 
Examples:
- single:code:user wants code written
- single:math:calculation needed
- direct::simple question I can answer
- sequential:math,code:need formula then implement it
 
Your answer:
[/INST]"""
 
        response = llm_service.generate(prompt, max_tokens=50, temperature=0.1)
        
        try:
            line = response.strip().split('\n')[0].lower()
            parts = line.split(':')
            
            order = parts[0].strip()
            areas = [a.strip() for a in parts[1].split(',') if a.strip()] if len(parts) > 1 else []
            reason = parts[2] if len(parts) > 2 else ""
            
            if order == "direct":
                return PFCPlan([], "direct", reason)
            elif order == "single" and areas:
                return PFCPlan([areas[0]], "single", reason)
            elif order == "sequential" and areas:
                return PFCPlan(areas, "sequential", reason)
        except:
            pass
        
        if top_area[1] > 0.65:
            return PFCPlan([top_area[0]], "single", "fallback to top signal")
        return PFCPlan([], "direct", "fallback")
    
    async def _process_area(self, area: str, query: str, context: Dict[str, Any]) -> AreaResult:
        start = time.time()
        try:
            if area == "code":
                output = await self._code(query, context)
            elif area == "math":
                output = await self._math(query, context)
            elif area == "physics":
                output = await self._physics(query, context)
            else:
                output = await self._language(query)
            
            return AreaResult(area, output, time.time() - start, True)
        except Exception as e:
            return AreaResult(area, str(e), time.time() - start, False)
    
    async def _code(self, query: str, ctx: Dict) -> str:
        enhanced = query
        if ctx.get("math_output"):
            enhanced += f"\n\nUse these formulas:\n{ctx['math_output'][:500]}"
        raw = deepseek_coder.generate_code(enhanced)
        return self._clean_code(raw)
    
    async def _math(self, query: str, ctx: Dict) -> str:
        prompt = f"[INST] Solve mathematically with equations and steps:\n{query}\n[/INST]"
        return llm_service.generate(prompt, max_tokens=400, temperature=0.2)
    
    async def _physics(self, query: str, ctx: Dict) -> str:
        prompt = f"[INST] Explain the physics with laws, constants, formulas:\n{query}\n[/INST]"
        return llm_service.generate(prompt, max_tokens=350, temperature=0.2)
    
    async def _language(self, query: str) -> str:
        return llm_service.generate(f"[INST] {query} [/INST]", max_tokens=600, temperature=0.7)
    
    def _clean_code(self, output: str) -> str:
        output = output.strip()
        if '```python' in output:
            m = re.findall(r'```python\n(.*?)```', output, re.DOTALL)
            if m:
                return f"```python\n{max(m, key=len).strip()}\n```"
        if '```' in output:
            m = re.findall(r'```\n?(.*?)```', output, re.DOTALL)
            if m:
                return f"```python\n{max(m, key=len).strip()}\n```"
        if output.startswith(('def ', 'class ', 'import ')):
            return f"```python\n{output}\n```"
        return output
    
    def _combine(self, results: List[AreaResult], ctx: Dict) -> str:
        parts = []
        if ctx.get("math_output"):
            parts.append(f"**Math:**\n\n{ctx['math_output']}")
        if ctx.get("physics_output"):
            parts.append(f"**Physics:**\n\n{ctx['physics_output']}")
        if ctx.get("code_output"):
            parts.append(f"**Code:**\n\n{ctx['code_output']}")
        if parts:
            return "\n\n---\n\n".join(parts)
        if results:
            return results[0].output
        return ""
    
    def get_stats(self):
        return {"total": self.total_pipelines}
 
 
brain_pipeline = BrainPipeline()
