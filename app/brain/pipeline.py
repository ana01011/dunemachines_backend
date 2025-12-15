"""
OMNIUS v3 - Hybrid Brain Pipeline
Primary area processes first, then parallel secondary areas, then PFC synthesizes.

Flow:
    Query → Thalamus (signals) → Rank Areas → Primary Process → Parallel Secondary → PFC Synthesize
"""
import asyncio
import time
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder


class AreaRole(str, Enum):
    PRIMARY = "primary"      # >60% - processes first, output feeds others
    SECONDARY = "secondary"  # 45-60% - runs in parallel after primary
    SUPPORTING = "supporting"  # 35-45% - lightweight check
    SKIP = "skip"           # <35% - not activated


@dataclass
class AreaResult:
    """Result from a brain area processing"""
    area_name: str
    role: AreaRole
    activation: float
    output: str
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Final result from the brain pipeline"""
    query: str
    final_response: str
    area_results: List[AreaResult]
    pipeline_order: List[str]
    total_time: float
    thalamus_signals: Dict[str, float]
    synthesis_method: str


class AreaProcessor:
    """Base class for specialized area processors"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        """Override in subclasses"""
        raise NotImplementedError


class MathProcessor(AreaProcessor):
    """Processes math/physics queries - extracts equations, formulas, calculations"""
    
    def __init__(self):
        super().__init__("math")
        self.math_patterns = [
            r'(\d+\s*[\+\-\*\/\^]\s*\d+)',  # Basic arithmetic
            r'([a-zA-Z]\s*=\s*[^,\n]+)',     # Variable assignments
            r'(∫|∑|∏|√|π|θ)',                # Math symbols
            r'(\d+\.?\d*\s*%)',              # Percentages
            r'(derivative|integral|limit|sum|equation|formula|calculate|solve)',
        ]
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        start_time = time.time()
        
        try:
            # Use LLM to extract mathematical content
            extract_prompt = f"""[INST] You are a mathematical analyst. Extract the key mathematical elements from this query.

Query: "{query}"

Provide:
1. EQUATIONS: Any mathematical equations or formulas relevant to answering this
2. CONSTANTS: Physical or mathematical constants needed (with values)
3. STEPS: Brief calculation steps if computation is needed
4. UNITS: Any units involved

Be concise and precise. If no math is needed, say "NO_MATH_REQUIRED".
[/INST]"""

            math_output = llm_service.generate(extract_prompt, max_tokens=300, temperature=0.3)
            
            # Extract structured data
            extracted = {
                "equations": self._extract_equations(math_output),
                "constants": self._extract_constants(math_output),
                "has_math": "NO_MATH_REQUIRED" not in math_output.upper()
            }
            
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output=math_output,
                extracted_data=extracted,
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equation-like patterns from text"""
        equations = []
        # Look for lines with = signs or common math operators
        for line in text.split('\n'):
            if '=' in line or any(op in line for op in ['+', '-', '*', '/', '^', '²', '³']):
                cleaned = line.strip().strip('-').strip('*').strip()
                if cleaned and len(cleaned) > 3:
                    equations.append(cleaned)
        return equations[:5]  # Limit to 5 equations
    
    def _extract_constants(self, text: str) -> Dict[str, str]:
        """Extract physical/mathematical constants"""
        constants = {}
        # Common patterns: G = 6.674, π ≈ 3.14, c = 3×10^8
        patterns = [
            r'([A-Za-z_]+)\s*[=≈]\s*([\d\.]+(?:\s*[×x]\s*10\^?[\d\-]+)?)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for name, value in matches:
                constants[name.strip()] = value.strip()
        return constants


class CodeProcessor(AreaProcessor):
    """Processes code generation requests using DeepSeek"""
    
    def __init__(self):
        super().__init__("code")
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        start_time = time.time()
        
        try:
            # Check if we have math context to incorporate
            math_context = context.get("math_output", "")
            
            if math_context and "NO_MATH_REQUIRED" not in math_context.upper():
                # Enhanced prompt with math context
                enhanced_query = f"""{query}

Use these mathematical foundations:
{math_context}

Implement the code based on these equations/formulas."""
            else:
                enhanced_query = query
            
            # Generate code with DeepSeek
            raw_code = deepseek_coder.generate_code(enhanced_query)
            clean_code = self._clean_code(raw_code)
            
            extracted = {
                "language": self._detect_language(clean_code),
                "has_functions": "def " in clean_code or "function " in clean_code,
                "has_classes": "class " in clean_code,
                "line_count": len(clean_code.split('\n'))
            }
            
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output=clean_code,
                extracted_data=extracted,
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _clean_code(self, output: str) -> str:
        """Extract clean code from output"""
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
        
        # Check if it looks like code
        code_indicators = ('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return ')
        lines = output.split('\n')
        code_lines = [l for l in lines if l.strip().startswith(code_indicators)]
        
        if len(code_lines) > 2:
            return f"```python\n{output}\n```"
        
        return output
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language"""
        if 'def ' in code or 'import ' in code:
            return "python"
        elif 'function ' in code or 'const ' in code or 'let ' in code:
            return "javascript"
        elif '#include' in code:
            return "c/c++"
        return "unknown"


class PhysicsProcessor(AreaProcessor):
    """Processes physics-related queries - extracts laws, constants, principles"""
    
    def __init__(self):
        super().__init__("physics")
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        start_time = time.time()
        
        try:
            extract_prompt = f"""[INST] You are a physics expert. Identify the physical principles relevant to this query.

Query: "{query}"

Provide:
1. LAWS: Physical laws that apply (e.g., Newton's laws, conservation laws)
2. CONSTANTS: Physical constants with SI values
3. FORMULAS: Key physics formulas needed
4. CONCEPTS: Core physics concepts involved

Be concise and precise.
[/INST]"""

            physics_output = llm_service.generate(extract_prompt, max_tokens=250, temperature=0.3)
            
            extracted = {
                "laws": self._extract_laws(physics_output),
                "concepts": self._extract_concepts(query)
            }
            
            return AreaResult(
                area_name=self.name,
                role=AreaRole.SECONDARY,
                activation=context.get("activation", 0.5),
                output=physics_output,
                extracted_data=extracted,
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            return AreaResult(
                area_name=self.name,
                role=AreaRole.SECONDARY,
                activation=context.get("activation", 0.5),
                output="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _extract_laws(self, text: str) -> List[str]:
        laws = []
        law_keywords = ['law', 'principle', 'theorem', 'conservation', 'equation']
        for line in text.split('\n'):
            if any(kw in line.lower() for kw in law_keywords):
                laws.append(line.strip())
        return laws[:5]
    
    def _extract_concepts(self, query: str) -> List[str]:
        concepts = []
        physics_terms = ['force', 'energy', 'momentum', 'velocity', 'acceleration', 
                        'mass', 'gravity', 'electric', 'magnetic', 'wave', 'quantum',
                        'relativity', 'thermodynamic', 'entropy', 'pressure', 'temperature']
        query_lower = query.lower()
        for term in physics_terms:
            if term in query_lower:
                concepts.append(term)
        return concepts


class MemoryProcessor(AreaProcessor):
    """Processes memory/context queries - retrieves relevant past information"""
    
    def __init__(self):
        super().__init__("memory")
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        start_time = time.time()
        
        # For now, return context from the conversation
        # In future, this would query the Hippocampus/ChromaDB
        
        user_context = context.get("user_context", {})
        conversation_history = context.get("conversation_history", [])
        
        extracted = {
            "has_history": len(conversation_history) > 0,
            "user_info": user_context,
            "relevant_memories": []  # Would come from Hippocampus
        }
        
        memory_summary = "No previous context available."
        if conversation_history:
            memory_summary = f"Previous conversation context: {len(conversation_history)} messages"
        
        return AreaResult(
            area_name=self.name,
            role=AreaRole.SUPPORTING,
            activation=context.get("activation", 0.5),
            output=memory_summary,
            extracted_data=extracted,
            processing_time=time.time() - start_time,
            success=True
        )


class LanguageProcessor(AreaProcessor):
    """Processes general language queries - explanations, descriptions"""
    
    def __init__(self):
        super().__init__("language")
    
    async def process(self, query: str, context: Dict[str, Any]) -> AreaResult:
        start_time = time.time()
        
        try:
            # Direct PFC response for language tasks
            response = llm_service.generate(
                f"[INST] {query} [/INST]",
                max_tokens=600,
                temperature=0.7
            )
            
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output=response,
                extracted_data={"type": "explanation"},
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            return AreaResult(
                area_name=self.name,
                role=AreaRole.PRIMARY,
                activation=context.get("activation", 0.5),
                output="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )


class BrainPipeline:
    """
    Hybrid Brain Pipeline Orchestrator
    
    Coordinates multiple brain areas based on thalamus activation signals.
    Primary area runs first, secondary areas run in parallel, then PFC synthesizes.
    """
    
    # Thresholds for area activation
    PRIMARY_THRESHOLD = 0.60    # >60% = primary processor
    SECONDARY_THRESHOLD = 0.45  # 45-60% = secondary (parallel)
    SUPPORTING_THRESHOLD = 0.35 # 35-45% = supporting (lightweight)
    
    def __init__(self):
        # Initialize area processors
        self.processors = {
            "math": MathProcessor(),
            "code": CodeProcessor(),
            "physics": PhysicsProcessor(),
            "memory": MemoryProcessor(),
            "language": LanguageProcessor()
        }
        
        # Stats
        self.total_pipelines = 0
        self.area_usage_count = {name: 0 for name in self.processors}
    
    def _classify_areas(self, signals: Dict[str, float]) -> Dict[str, AreaRole]:
        """Classify each area's role based on activation signal"""
        classifications = {}
        
        for area, activation in signals.items():
            if activation >= self.PRIMARY_THRESHOLD:
                classifications[area] = AreaRole.PRIMARY
            elif activation >= self.SECONDARY_THRESHOLD:
                classifications[area] = AreaRole.SECONDARY
            elif activation >= self.SUPPORTING_THRESHOLD:
                classifications[area] = AreaRole.SUPPORTING
            else:
                classifications[area] = AreaRole.SKIP
        
        return classifications
    
    def _rank_areas(self, signals: Dict[str, float]) -> List[Tuple[str, float, AreaRole]]:
        """Rank areas by activation, assign roles"""
        classifications = self._classify_areas(signals)
        
        ranked = []
        for area, activation in sorted(signals.items(), key=lambda x: x[1], reverse=True):
            role = classifications.get(area, AreaRole.SKIP)
            ranked.append((area, activation, role))
        
        return ranked
    
    async def process(self, query: str, signals: Dict[str, float], context: Dict[str, Any] = None) -> PipelineResult:
        """
        Main pipeline processing
        
        Args:
            query: User's query
            signals: Thalamus activation signals {area_name: activation_level}
            context: Additional context (user info, history, etc.)
        
        Returns:
            PipelineResult with final response and all area outputs
        """
        start_time = time.time()
        context = context or {}
        
        # Rank and classify areas
        ranked_areas = self._rank_areas(signals)
        
        print(f"\n[Pipeline] Processing: {query[:50]}...")
        print(f"[Pipeline] Area Rankings:")
        for area, activation, role in ranked_areas:
            print(f"    {area}: {activation*100:.0f}% → {role.value}")
        
        area_results = []
        pipeline_order = []
        accumulated_context = dict(context)
        
        # PHASE 1: Process PRIMARY areas (sequentially, highest first)
        primary_areas = [(a, act) for a, act, role in ranked_areas if role == AreaRole.PRIMARY]
        
        if primary_areas:
            print(f"\n[Phase 1] PRIMARY areas: {[a for a, _ in primary_areas]}")
            
            for area_name, activation in primary_areas:
                if area_name in self.processors:
                    processor = self.processors[area_name]
                    accumulated_context["activation"] = activation
                    
                    print(f"    Processing {area_name}...")
                    result = await processor.process(query, accumulated_context)
                    result.role = AreaRole.PRIMARY
                    result.activation = activation
                    
                    area_results.append(result)
                    pipeline_order.append(area_name)
                    self.area_usage_count[area_name] += 1
                    
                    # Add output to context for next areas
                    if result.success:
                        accumulated_context[f"{area_name}_output"] = result.output
                        accumulated_context[f"{area_name}_data"] = result.extracted_data
                    
                    print(f"    {area_name} done in {result.processing_time:.1f}s")
        
        # PHASE 2: Process SECONDARY areas (in parallel)
        secondary_areas = [(a, act) for a, act, role in ranked_areas if role == AreaRole.SECONDARY]
        
        if secondary_areas:
            print(f"\n[Phase 2] SECONDARY areas (parallel): {[a for a, _ in secondary_areas]}")
            
            async def process_secondary(area_name: str, activation: float) -> AreaResult:
                if area_name in self.processors:
                    processor = self.processors[area_name]
                    ctx = dict(accumulated_context)
                    ctx["activation"] = activation
                    result = await processor.process(query, ctx)
                    result.role = AreaRole.SECONDARY
                    result.activation = activation
                    return result
                return None
            
            # Run secondary areas in parallel
            tasks = [process_secondary(a, act) for a, act in secondary_areas]
            secondary_results = await asyncio.gather(*tasks)
            
            for result in secondary_results:
                if result:
                    area_results.append(result)
                    pipeline_order.append(result.area_name)
                    self.area_usage_count[result.area_name] += 1
                    
                    if result.success:
                        accumulated_context[f"{result.area_name}_output"] = result.output
                        accumulated_context[f"{result.area_name}_data"] = result.extracted_data
                    
                    print(f"    {result.area_name} done in {result.processing_time:.1f}s")
        
        # PHASE 3: Synthesize final response
        print(f"\n[Phase 3] Synthesizing response...")
        final_response, synthesis_method = await self._synthesize(query, area_results, accumulated_context)
        
        total_time = time.time() - start_time
        self.total_pipelines += 1
        
        print(f"\n[Pipeline] Complete in {total_time:.1f}s using {synthesis_method}")
        
        return PipelineResult(
            query=query,
            final_response=final_response,
            area_results=area_results,
            pipeline_order=pipeline_order,
            total_time=total_time,
            thalamus_signals=signals,
            synthesis_method=synthesis_method
        )
    
    async def _synthesize(self, query: str, area_results: List[AreaResult], context: Dict[str, Any]) -> Tuple[str, str]:
        """
        Synthesize final response from all area outputs
        
        Returns:
            Tuple of (response, synthesis_method)
        """
        successful_results = [r for r in area_results if r.success and r.output]
        
        if not successful_results:
            # Fallback to direct PFC response
            response = llm_service.generate(f"[INST] {query} [/INST]", max_tokens=600)
            return response, "direct_pfc_fallback"
        
        # Single primary area - use its output directly
        if len(successful_results) == 1:
            result = successful_results[0]
            if result.area_name == "code":
                return f"Here's the solution:\n\n{result.output}", "single_code"
            elif result.area_name == "language":
                return result.output, "single_language"
            else:
                return result.output, f"single_{result.area_name}"
        
        # Multiple areas - synthesize with PFC
        synthesis_parts = []
        
        # Collect outputs by type
        code_output = context.get("code_output", "")
        math_output = context.get("math_output", "")
        physics_output = context.get("physics_output", "")
        language_output = context.get("language_output", "")
        
        # Build synthesis prompt
        synthesis_prompt = f"""[INST] You are OMNIUS, an advanced AI with multiple specialized brain regions. 
Synthesize the following outputs from your brain regions into a coherent, comprehensive response.

USER QUERY: "{query}"

"""
        
        if math_output and "NO_MATH" not in math_output.upper():
            synthesis_prompt += f"""MATH REGION OUTPUT:
{math_output}

"""
        
        if physics_output:
            synthesis_prompt += f"""PHYSICS REGION OUTPUT:
{physics_output}

"""
        
        if code_output:
            synthesis_prompt += f"""CODE REGION OUTPUT:
{code_output}

"""
        
        if language_output:
            synthesis_prompt += f"""LANGUAGE REGION OUTPUT:
{language_output}

"""
        
        synthesis_prompt += """Create a unified response that:
1. Integrates insights from all active brain regions
2. Presents information in a logical flow
3. Includes any code or equations from the specialized regions
4. Is comprehensive but not repetitive

Your synthesized response:
[/INST]"""
        
        synthesized = llm_service.generate(synthesis_prompt, max_tokens=800, temperature=0.5)
        
        # If code was generated, make sure it's preserved
        if code_output and "```" in code_output and "```" not in synthesized:
            synthesized = f"{synthesized}\n\n{code_output}"
        
        return synthesized, "multi_area_synthesis"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "total_pipelines": self.total_pipelines,
            "area_usage": self.area_usage_count,
            "thresholds": {
                "primary": self.PRIMARY_THRESHOLD,
                "secondary": self.SECONDARY_THRESHOLD,
                "supporting": self.SUPPORTING_THRESHOLD
            }
        }


# Global pipeline instance
brain_pipeline = BrainPipeline()
