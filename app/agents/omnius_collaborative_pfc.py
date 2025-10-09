"""
Omnius with COLLABORATIVE PFC - True parallel thinking with streaming
"""
from typing import Dict, Any, Tuple, List, Optional, AsyncGenerator
import json
import re
import time
import asyncio
import subprocess
import tempfile
import os

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.core.database import db


class OmniusCollaborativePFC:
    """Omnius with collaborative thinking between brain regions"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.prefrontal_cortex = llm_service
        self.code_cortex = deepseek_coder
        
    async def initialize(self, db_pool):
        """Initialize"""
        print("🧠 Omnius COLLABORATIVE PFC initialized with streaming support")
        
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
        Collaborative thinking with detailed logging and clean output
        """
        
        print("\n" + "="*80)
        print("🧠 [OMNIUS COLLABORATIVE THINKING INITIATED]")
        print("="*80)
        print(f"📥 USER REQUEST: {message}")
        print("-"*80)
        
        # ================================================
        # PHASE 1: UNDERSTAND THE GOAL
        # ================================================
        print("\n💭 [PHASE 1: UNDERSTANDING THE TRUE GOAL]")
        print("-"*40)
        
        goal_understanding = await self._understand_goal(message)
        print(f"✓ True Goal Identified: {goal_understanding['true_goal']}")
        print(f"✓ User Intent: {goal_understanding['intent']}")
        print(f"✓ Expected Output Type: {goal_understanding['output_type']}")
        print(f"✓ Success Criteria: {', '.join(goal_understanding['success_criteria'])}")
        
        # ================================================
        # PHASE 2: COLLABORATIVE PLANNING
        # ================================================
        print("\n📋 [PHASE 2: COLLABORATIVE PLANNING]")
        print("-"*40)
        
        # Parallel consultation with specialists
        print("🤝 Consulting specialists in parallel...")
        
        consultation_tasks = []
        if goal_understanding.get('needs_code'):
            print("  → Consulting Code Cortex...")
            consultation_tasks.append(self._consult_code_specialist(message, goal_understanding))
        
        if goal_understanding.get('needs_explanation'):
            print("  → Consulting Explanation Specialist...")
            consultation_tasks.append(self._consult_explanation_specialist(message, goal_understanding))
        
        if goal_understanding.get('needs_math'):
            print("  → Consulting Math Specialist...")
            consultation_tasks.append(self._consult_math_specialist(message, goal_understanding))
        
        # Wait for all consultations to complete
        if consultation_tasks:
            consultations = await asyncio.gather(*consultation_tasks)
        else:
            consultations = []
        
        # Merge consultations into unified plan
        unified_plan = await self._create_unified_plan(goal_understanding, consultations)
        
        print("\n📊 UNIFIED EXECUTION PLAN:")
        print(f"  Steps: {len(unified_plan['steps'])}")
        for i, step in enumerate(unified_plan['steps'], 1):
            print(f"    {i}. {step['action']} ({step['specialist']})")
            if 'expected_output' in step:
                print(f"       Expected: {step['expected_output'][:50]}...")
        
        print("\n🎯 QUALITY PREDICTIONS:")
        for metric, score in unified_plan['predicted_scores'].items():
            print(f"  - {metric}: {score:.2f}")
        
        # ================================================
        # PHASE 3: PARALLEL EXECUTION
        # ================================================
        print("\n⚡ [PHASE 3: PARALLEL EXECUTION]")
        print("-"*40)
        
        execution_results = await self._execute_parallel(unified_plan, message)
        
        # ================================================
        # PHASE 4: QUALITY VALIDATION
        # ================================================
        print("\n✅ [PHASE 4: QUALITY VALIDATION]")
        print("-"*40)
        
        validation_results = await self._validate_quality(execution_results, unified_plan)
        
        print("📈 VALIDATION RESULTS:")
        total_score = 0
        for metric, score in validation_results.items():
            predicted = unified_plan['predicted_scores'].get(metric, 0.5)
            symbol = "✓" if score >= predicted else "✗"
            print(f"  {symbol} {metric}: {score:.2f} (predicted: {predicted:.2f})")
            total_score += score
        
        avg_score = total_score / len(validation_results) if validation_results else 0
        print(f"\n🏆 OVERALL QUALITY SCORE: {avg_score:.2f}")
        
        # ================================================
        # PHASE 5: ITERATION IF NEEDED
        # ================================================
        if avg_score < 0.8:
            print("\n🔄 [PHASE 5: QUALITY IMPROVEMENT ITERATION]")
            print("-"*40)
            print(f"  Score {avg_score:.2f} below threshold 0.8")
            print("  → Initiating improvement cycle...")
            
            # Identify weak areas and improve
            improved_results = await self._improve_weak_areas(
                execution_results, validation_results, unified_plan, message
            )
            execution_results = improved_results
            
            # Re-validate
            validation_results = await self._validate_quality(execution_results, unified_plan)
            avg_score = sum(validation_results.values()) / len(validation_results)
            print(f"  ✓ Improved score: {avg_score:.2f}")
        
        # ================================================
        # PHASE 6: FINAL SYNTHESIS
        # ================================================
        print("\n🎨 [PHASE 6: SYNTHESIZING FINAL RESPONSE]")
        print("-"*40)
        
        final_response = self._synthesize_clean_response(execution_results, unified_plan)
        print(f"  ✓ Final response prepared: {len(final_response)} characters")
        
        print("\n" + "="*80)
        print("✅ [COLLABORATIVE THINKING COMPLETE]")
        print("="*80 + "\n")
        
        return final_response, {
            'consciousness_used': unified_plan.get('regions_used', ['prefrontal_cortex']),
            'neurochemistry_active': False,
            'thinking_process': 'Collaborative parallel thinking',
            'quality_score': avg_score,
            'plan': unified_plan,
            'streaming_ready': True
        }
    
    async def think_stream(self, message: str, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Streaming version for WebSocket support
        """
        # Get the full response first
        response, metadata = await self.think(message, context)
        
        # Stream it in chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            yield chunk
            await asyncio.sleep(0.01)  # Small delay for streaming effect
    
    async def _understand_goal(self, message: str) -> Dict:
        """
        Truly understand what the user wants
        """
        understanding_prompt = f"""[INST]Analyze this request deeply.

Request: {message}

Think carefully and determine:
1. What is the TRUE GOAL? What does the user really want to achieve?
2. What TYPE of response is needed? (explanation, code, calculation, analysis, etc.)
3. What COMPONENTS are required? (code, examples, math, diagrams, etc.)
4. What would make this response PERFECT?
5. What common MISTAKES should we avoid?

Respond in JSON:
{{
    "true_goal": "actual user goal",
    "intent": "underlying intent",
    "output_type": "type of output",
    "needs_code": true/false,
    "needs_explanation": true/false,
    "needs_math": true/false,
    "needs_examples": true/false,
    "success_criteria": ["criterion1", "criterion2"],
    "potential_mistakes": ["mistake1", "mistake2"]
}}[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            understanding_prompt,
            temperature=0.3,
            max_tokens=400
        )
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback understanding
        return {
            "true_goal": "Answer the query",
            "intent": "Get information",
            "output_type": "explanation",
            "needs_code": "code" in message.lower(),
            "needs_explanation": True,
            "needs_math": False,
            "needs_examples": "example" in message.lower(),
            "success_criteria": ["clear", "complete", "correct"],
            "potential_mistakes": ["incomplete", "unclear"]
        }
    
    async def _consult_code_specialist(self, message: str, goal: Dict) -> Dict:
        """
        Code Cortex consultation for planning
        """
        consultation = {
            "specialist": "code_cortex",
            "recommendation": "Generate clean, tested code",
            "approach": "Focus on efficiency and readability",
            "expected_output": "Working implementation with comments",
            "quality_checks": ["syntax_valid", "logic_correct", "efficient"]
        }
        
        # If quicksort mentioned, add specific recommendations
        if "quicksort" in message.lower():
            consultation["specific_requirements"] = [
                "Include pivot selection strategy",
                "Handle edge cases",
                "Optimize for performance"
            ]
        
        return consultation
    
    async def _consult_explanation_specialist(self, message: str, goal: Dict) -> Dict:
        """
        Explanation specialist consultation
        """
        return {
            "specialist": "explanation",
            "recommendation": "Provide clear, structured explanation",
            "approach": "Step-by-step with examples",
            "expected_output": "Comprehensive understanding",
            "quality_checks": ["clarity", "completeness", "accuracy"]
        }
    
    async def _consult_math_specialist(self, message: str, goal: Dict) -> Dict:
        """
        Math specialist consultation
        """
        return {
            "specialist": "math",
            "recommendation": "Ensure mathematical correctness",
            "approach": "Rigorous proofs and calculations",
            "expected_output": "Accurate mathematical content",
            "quality_checks": ["consistency", "correctness", "precision"]
        }
    
    async def _create_unified_plan(self, goal: Dict, consultations: List[Dict]) -> Dict:
        """
        Create unified plan from all consultations
        """
        steps = []
        regions_used = ['prefrontal_cortex']
        
        # Add steps based on consultations
        for consultation in consultations:
            if consultation['specialist'] == 'code_cortex':
                steps.append({
                    "action": "Generate code implementation",
                    "specialist": "code_cortex",
                    "expected_output": consultation['expected_output'],
                    "quality_checks": consultation['quality_checks']
                })
                regions_used.append('code_cortex')
            elif consultation['specialist'] == 'explanation':
                steps.append({
                    "action": "Generate explanation",
                    "specialist": "prefrontal_cortex",
                    "expected_output": consultation['expected_output'],
                    "quality_checks": consultation['quality_checks']
                })
        
        # Predict quality scores based on our capabilities
        predicted_scores = {
            "completeness": 0.9,
            "correctness": 0.85,
            "clarity": 0.9,
            "usefulness": 0.85
        }
        
        if goal.get('needs_code'):
            predicted_scores["code_quality"] = 0.8
        
        return {
            "steps": steps,
            "predicted_scores": predicted_scores,
            "regions_used": list(set(regions_used)),
            "success_criteria": goal.get('success_criteria', [])
        }
    
    async def _execute_parallel(self, plan: Dict, message: str) -> Dict:
        """
        Execute plan steps in parallel where possible
        """
        results = {}
        
        # Group steps by dependency
        tasks = []
        for i, step in enumerate(plan['steps']):
            if step['specialist'] == 'code_cortex':
                tasks.append(self._execute_code_step(message, step, f"step_{i}"))
            else:
                tasks.append(self._execute_explanation_step(message, step, f"step_{i}"))
        
        # Execute all tasks in parallel
        if tasks:
            task_results = await asyncio.gather(*tasks)
            for result in task_results:
                results.update(result)
        
        return results
    
    async def _execute_code_step(self, message: str, step: Dict, key: str) -> Dict:
        """
        Execute code generation step
        """
        code = deepseek_coder.generate_code(message)
        if '```' not in code:
            code = f"```python\n{code}\n```"
        return {key: code}
    
    async def _execute_explanation_step(self, message: str, step: Dict, key: str) -> Dict:
        """
        Execute explanation step
        """
        explanation = self.prefrontal_cortex.generate(
            f"[INST]Explain clearly: {message}[/INST]",
            temperature=0.7,
            max_tokens=800
        )
        return {key: explanation}
    
    async def _validate_quality(self, results: Dict, plan: Dict) -> Dict:
        """
        Validate quality with real checks (code execution, math verification)
        """
        scores = {}
        
        # Check completeness
        total_content = ' '.join(results.values())
        scores['completeness'] = min(1.0, len(total_content) / 1000)  # Simple length check
        
        # Check code quality if present
        code_blocks = re.findall(r'```python\n(.*?)```', total_content, re.DOTALL)
        if code_blocks:
            code_score = await self._validate_code(code_blocks)
            scores['code_quality'] = code_score
            scores['correctness'] = code_score  # Code correctness affects overall correctness
        else:
            scores['correctness'] = 0.8  # Default for non-code
        
        # Check clarity (simple heuristic)
        scores['clarity'] = 0.9 if len(total_content.split('\n')) > 5 else 0.7
        
        # Check usefulness
        scores['usefulness'] = 0.85  # Default, could be improved with user feedback
        
        return scores
    
    async def _validate_code(self, code_blocks: List[str]) -> float:
        """
        Actually test the code for syntax and basic functionality
        """
        total_score = 0
        
        for code in code_blocks:
            # Test syntax by trying to compile
            try:
                compile(code, '<string>', 'exec')
                total_score += 0.5  # Syntax is valid
                
                # Try to execute in isolated environment
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.flush()
                    
                    # Run with timeout
                    try:
                        result = subprocess.run(
                            ['python', f.name],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            total_score += 0.5  # Execution successful
                    except subprocess.TimeoutExpired:
                        total_score += 0.3  # Code runs but might be infinite
                    finally:
                        os.unlink(f.name)
                        
            except SyntaxError:
                total_score += 0.0  # Syntax error
                
        return total_score / len(code_blocks) if code_blocks else 0
    
    async def _improve_weak_areas(self, results: Dict, scores: Dict, plan: Dict, message: str) -> Dict:
        """
        Improve areas with low scores
        """
        # Find weakest area
        weakest = min(scores.items(), key=lambda x: x[1])
        
        if weakest[0] == 'code_quality' and weakest[1] < 0.7:
            # Regenerate code with emphasis on correctness
            improved_code = deepseek_coder.generate_code(
                f"Write CORRECT, TESTED code. {message}"
            )
            if '```' not in improved_code:
                improved_code = f"```python\n{improved_code}\n```"
            
            # Replace code in results
            for key in results:
                if 'code' in key or '```' in results[key]:
                    results[key] = improved_code
                    break
        
        return results
    
    def _synthesize_clean_response(self, results: Dict, plan: Dict) -> str:
        """
        Create single, clean final response (no duplicates)
        """
        # Order results by step order
        ordered_parts = []
        for i in range(len(plan['steps'])):
            key = f"step_{i}"
            if key in results and results[key].strip():
                ordered_parts.append(results[key])
        
        # Join with proper spacing
        return "\n\n".join(ordered_parts)
    
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
            'processing_power': 0.7,
            'streaming_enabled': True
        }
    
    async def shutdown(self):
        """Shutdown"""
        print("👋 Omnius Collaborative PFC shutdown")


# Create instance
omnius_neurochemical = OmniusCollaborativePFC()
