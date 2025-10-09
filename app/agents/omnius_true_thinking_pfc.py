"""
Omnius with TRUE THINKING PFC - Self-planning, prediction, and validation
"""
from typing import Dict, Any, Tuple, List, Optional
import json
import re
import time

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.core.database import db


class OmniusTrueThinkingPFC:
    """Omnius with true thinking PFC that plans, predicts, and validates"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.prefrontal_cortex = llm_service
        self.code_cortex = deepseek_coder
        
    async def initialize(self, db_pool):
        """Initialize"""
        print("🧠 Omnius TRUE THINKING PFC initialized")
        
    async def check_tokens(self, user_id: str) -> Dict:
        """Check tokens"""
        try:
            result = await db.fetchrow("""
                SELECT * FROM user_token_status WHERE user_id = $1::uuid
            """, user_id)
            
            if not result:
                await db.execute("""
                    INSERT INTO user_tokens (user_id) VALUES ($1::uuid)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                return {'has_tokens': True, 'tokens_remaining': 1000}
            
            return {
                'has_tokens': result.get('tokens_remaining', 0) > 0,
                'tokens_remaining': result.get('tokens_remaining', 0)
            }
        except:
            return {'has_tokens': True, 'tokens_remaining': 1000}
    
    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        TRUE THINKING with prediction, planning, execution, and validation
        """
        
        print("\n" + "="*70)
        print("🧠 [PFC TRUE THINKING PROCESS INITIATED]")
        print("="*70)
        print(f"📥 Input: {message[:100]}...")
        
        # ========================================
        # PHASE 1: TRUE THINKING & ANALYSIS
        # ========================================
        print("\n🤔 [PHASE 1: PFC THINKING & ANALYSIS]")
        thought_process = await self._true_think(message)
        print(f"💭 Thought process complete")
        
        # ========================================
        # PHASE 2: PREDICTION & PLANNING
        # ========================================
        print("\n📊 [PHASE 2: PREDICTION & PLANNING]")
        execution_plan = await self._create_execution_plan(message, thought_process)
        print(f"📋 Plan created with {len(execution_plan['steps'])} steps")
        print(f"🎯 Predicted quality scores:")
        for criterion, score in execution_plan['predicted_scores'].items():
            print(f"   - {criterion}: {score:.2f}")
        
        # ========================================
        # PHASE 3: ITERATIVE EXECUTION WITH VALIDATION
        # ========================================
        print("\n⚡ [PHASE 3: EXECUTION WITH VALIDATION LOOP]")
        
        max_iterations = 3
        best_output = None
        best_score = 0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Execute plan
            output = await self._execute_plan(message, execution_plan)
            
            # Validate against predictions
            actual_scores = await self._validate_output(output, execution_plan)
            
            print(f"📈 Validation scores:")
            total_score = 0
            for criterion, score in actual_scores.items():
                predicted = execution_plan['predicted_scores'].get(criterion, 0.5)
                print(f"   - {criterion}: {score:.2f} (predicted: {predicted:.2f})")
                total_score += score
            
            avg_score = total_score / len(actual_scores) if actual_scores else 0
            print(f"   📊 Average score: {avg_score:.2f}")
            
            # Check if we meet quality threshold
            if avg_score >= 0.8:
                print(f"   ✅ Quality threshold met!")
                best_output = output
                best_score = avg_score
                break
            elif avg_score > best_score:
                best_output = output
                best_score = avg_score
            
            # If not good enough, adjust plan
            if iteration < max_iterations - 1:
                print(f"   🔧 Adjusting plan for better quality...")
                execution_plan = await self._adjust_plan(execution_plan, actual_scores)
        
        # ========================================
        # PHASE 4: FINAL SYNTHESIS
        # ========================================
        print("\n🎨 [PHASE 4: FINAL SYNTHESIS]")
        final_response = await self._synthesize_final(best_output, execution_plan)
        print(f"✨ Final response prepared: {len(final_response)} chars")
        print(f"🏆 Final quality score: {best_score:.2f}")
        
        print("\n✅ [PFC THINKING COMPLETE]")
        print("="*70 + "\n")
        
        return final_response, {
            'consciousness_used': execution_plan.get('regions_used', ['prefrontal_cortex']),
            'neurochemistry_active': False,
            'thinking_process': 'True PFC thinking with validation',
            'quality_score': best_score,
            'iterations': iteration + 1,
            'plan': execution_plan
        }
    
    async def _true_think(self, message: str) -> Dict:
        """
        PFC truly thinks about the problem
        """
        thinking_prompt = f"""[INST]You are the Prefrontal Cortex. Think deeply about this request.

Request: {message}

Think step by step and answer:
1. What is the user REALLY asking for? What do they want to achieve?
2. What type of response would be MOST helpful?
3. What components should the response have? (explanation, code, examples, math, diagrams, etc.)
4. What level of detail is appropriate?
5. What would make this a PERFECT response?
6. What could go wrong? What mistakes should we avoid?

Provide your thinking in JSON format:
{{
    "true_goal": "what user really wants",
    "response_type": "type of response needed",
    "required_components": ["list", "of", "components"],
    "detail_level": "simple/moderate/comprehensive",
    "success_criteria": ["what", "makes", "perfect", "response"],
    "potential_pitfalls": ["things", "to", "avoid"],
    "confidence": 0.0 to 1.0
}}[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            thinking_prompt,
            temperature=0.4,  # Lower temp for analytical thinking
            max_tokens=500
        )
        
        # Parse thinking
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback thinking
        return {
            "true_goal": "Answer the query",
            "response_type": "explanation",
            "required_components": ["explanation"],
            "detail_level": "moderate",
            "success_criteria": ["clear", "helpful", "complete"],
            "potential_pitfalls": ["too complex", "incomplete"],
            "confidence": 0.5
        }
    
    async def _create_execution_plan(self, message: str, thought_process: Dict) -> Dict:
        """
        Create detailed execution plan with predictions
        """
        components = thought_process.get('required_components', [])
        
        planning_prompt = f"""[INST]Based on this analysis, create an execution plan.

User request: {message}
Analysis: {json.dumps(thought_process, indent=2)}

Create a detailed plan with:
1. Specific steps to execute
2. Which brain regions to use
3. Quality predictions for each component

Respond in JSON:
{{
    "steps": [
        {{"action": "what to do", "specialist": "which brain area", "details": "specifics"}}
    ],
    "predicted_scores": {{
        "completeness": 0.0-1.0,
        "correctness": 0.0-1.0,
        "clarity": 0.0-1.0,
        "usefulness": 0.0-1.0
    }},
    "expected_output_characteristics": "description of ideal output"
}}[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            planning_prompt,
            temperature=0.3,
            max_tokens=600
        )
        
        # Parse plan
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                
                # Determine regions needed
                regions = ['prefrontal_cortex']
                for step in plan.get('steps', []):
                    if 'code' in step.get('specialist', '').lower():
                        regions.append('code_cortex')
                
                plan['regions_used'] = list(set(regions))
                return plan
        except:
            pass
        
        # Fallback plan
        needs_code = 'code' in ' '.join(components).lower()
        return {
            "steps": [
                {"action": "analyze", "specialist": "prefrontal", "details": "understand request"},
                {"action": "execute", "specialist": "code_cortex" if needs_code else "prefrontal", "details": "generate response"}
            ],
            "predicted_scores": {
                "completeness": 0.7,
                "correctness": 0.7,
                "clarity": 0.7,
                "usefulness": 0.7
            },
            "expected_output_characteristics": "Clear and helpful response",
            "regions_used": ['prefrontal_cortex', 'code_cortex'] if needs_code else ['prefrontal_cortex']
        }
    
    async def _execute_plan(self, message: str, plan: Dict) -> Dict:
        """
        Execute the plan step by step
        """
        outputs = {}
        
        for i, step in enumerate(plan.get('steps', [])):
            specialist = step.get('specialist', '').lower()
            action = step.get('action', '')
            
            if 'code' in specialist:
                # Execute with Code Cortex
                output = deepseek_coder.generate_code(message)
                if '```' not in output:
                    output = f"```python\n{output}\n```"
                outputs[f'step_{i}'] = output
            else:
                # Execute with PFC
                prompt = f"[INST]{action}: {message}[/INST]"
                output = self.prefrontal_cortex.generate(
                    prompt,
                    temperature=0.6,
                    max_tokens=800
                )
                outputs[f'step_{i}'] = output
        
        return outputs
    
    async def _validate_output(self, output: Dict, plan: Dict) -> Dict:
        """
        Validate output against predicted scores
        """
        # Combine all outputs
        combined_output = '\n\n'.join(output.values())
        
        validation_prompt = f"""[INST]Evaluate this output against quality criteria.

Output to evaluate:
{combined_output[:1000]}...

Expected characteristics: {plan.get('expected_output_characteristics', 'Good quality')}

Score each criterion from 0.0 to 1.0:
{{
    "completeness": "score and reason",
    "correctness": "score and reason",
    "clarity": "score and reason",
    "usefulness": "score and reason"
}}

Respond with just the scores as JSON:
{{
    "completeness": 0.0-1.0,
    "correctness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "usefulness": 0.0-1.0
}}[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            validation_prompt,
            temperature=0.2,  # Low temp for consistent scoring
            max_tokens=200
        )
        
        # Parse scores
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                # Ensure scores are floats
                return {k: float(v) if isinstance(v, (int, float)) else 0.5 
                       for k, v in scores.items()}
        except:
            pass
        
        # Default scores
        return {
            "completeness": 0.5,
            "correctness": 0.5,
            "clarity": 0.5,
            "usefulness": 0.5
        }
    
    async def _adjust_plan(self, plan: Dict, scores: Dict) -> Dict:
        """
        Adjust plan based on validation scores
        """
        # Find weakest area
        weakest = min(scores.items(), key=lambda x: x[1])
        
        adjustment_prompt = f"""[INST]The output scored low on {weakest[0]} ({weakest[1]:.2f}).
        
Adjust the plan to improve {weakest[0]}.

Current plan: {json.dumps(plan['steps'], indent=2)}

Create adjusted plan focusing on improving {weakest[0]}:
{{
    "steps": [...],
    "predicted_scores": {{...}},
    "expected_output_characteristics": "..."
}}[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            adjustment_prompt,
            temperature=0.4,
            max_tokens=400
        )
        
        # Parse adjusted plan
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                adjusted = json.loads(json_match.group())
                adjusted['regions_used'] = plan['regions_used']
                return adjusted
        except:
            pass
        
        return plan
    
    async def _synthesize_final(self, output: Dict, plan: Dict) -> str:
        """
        Synthesize final clean response
        """
        # Combine outputs cleanly
        parts = []
        for key, value in output.items():
            if value and len(value.strip()) > 0:
                parts.append(value)
        
        return '\n\n'.join(parts)
    
    def get_status(self) -> Dict:
        """Get status"""
        return {
            'prefrontal_cortex': 'active',
            'code_cortex': 'active',
            'math_region': 'not_installed',
            'creative_center': 'not_installed',
            'neurochemistry': 'ready_but_disabled',
            'total_parameters': '~14B',
            'active_regions': 2,
            'processing_power': 0.7
        }
    
    async def shutdown(self):
        """Shutdown"""
        print("👋 Omnius TRUE THINKING PFC shutdown")


# Create instance
omnius_neurochemical = OmniusTrueThinkingPFC()
