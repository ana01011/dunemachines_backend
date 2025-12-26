"""
PFC Planner - Deep thinking and execution planning
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from app.services.llm_service import llm_service
 
 
@dataclass
class ExecutionPlan:
    needs_code: bool
    needs_math: bool
    areas_needed: List[str]
    execution_order: List[str]
    reasoning: str
    complexity: str
    deep_analysis: str
 
 
class PFCPlanner:
    def __init__(self):
        self.last_plan = None
    
    def analyze_and_plan(self, message: str, thalamus_signals: Dict[str, float]) -> ExecutionPlan:
        """Deep analysis of user request and create execution plan"""
        
        # Sort areas by activation signal
        sorted_areas = sorted(thalamus_signals.items(), key=lambda x: -x[1])
        top_areas = [a[0] for a in sorted_areas if a[1] > 0.4]
        
        # Quick detection
        msg_lower = message.lower()
        code_keywords = ['write code', 'write a function', 'create a function', 
                        'implement', 'code for', 'python', 'javascript',
                        'fix this code', 'debug', 'algorithm', 'program',
                        'write a program', 'create a class', 'script']
        needs_code = any(kw in msg_lower for kw in code_keywords)
        if '```' in message or 'def ' in message:
            needs_code = True
        
        math_keywords = ['calculate', 'compute', 'solve', 'equation', 'integral', 'derivative', 'sum of', 'product of']
        needs_math = any(kw in msg_lower for kw in math_keywords)
        if any(c in message for c in ['*', '/', '+']) and any(c.isdigit() for c in message):
            needs_math = True
        
        # Determine complexity
        if needs_code and needs_math:
            complexity = 'high'
        elif needs_code or needs_math or len(message) > 200:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # Deep thinking for medium/high complexity
        deep_analysis = ""
        if complexity in ['medium', 'high']:
            deep_analysis = self._deep_think(message)
        
        # Build execution order
        areas_needed = ['prefrontal_cortex']
        if needs_code:
            areas_needed.append('code_cortex')
        if needs_math:
            areas_needed.append('math_cortex')
        for area in top_areas[:2]:
            if area not in areas_needed:
                areas_needed.append(area)
        
        # Reasoning
        reasoning = f"Complexity: {complexity}. Code: {needs_code}, Math: {needs_math}. "
        reasoning += f"Signals: {', '.join([f'{a}:{int(s*100)}%' for a,s in sorted_areas[:3]])}."
        
        plan = ExecutionPlan(
            needs_code=needs_code,
            needs_math=needs_math,
            areas_needed=areas_needed,
            execution_order=areas_needed,
            reasoning=reasoning,
            complexity=complexity,
            deep_analysis=deep_analysis
        )
        
        self.last_plan = plan
        return plan
    
    def _deep_think(self, message: str) -> str:
        """PFC deep analysis of the request"""
        prompt = f"""[INST] Analyze this request in 2-3 sentences:
 
"{message}"
 
What does the user want? What's the best approach? Any potential issues?
[/INST]
 
Analysis:"""
        
        try:
            analysis = llm_service.generate(prompt, max_tokens=100, temperature=0.3)
            return analysis.strip().split('\n')[0]
        except:
            return "Quick processing mode."
 
 
pfc_planner = PFCPlanner()
