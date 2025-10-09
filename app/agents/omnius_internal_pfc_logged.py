"""
Omnius with INTERNAL PFC planning with proper logging
"""
from typing import Dict, Any, Tuple, List, Optional
import json
import re

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.core.database import db


class OmniusInternalPFC:
    """Omnius with internal PFC planning - with visible server logs"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.prefrontal_cortex = llm_service  # Mistral as PFC
        self.code_cortex = deepseek_coder    # DeepSeek for code
        
    async def initialize(self, db_pool):
        """Initialize"""
        print("✅ Omnius Internal PFC Orchestrator initialized")
        
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
        Internal PFC thinking - all planning is hidden from user
        """
        
        print("\n" + "="*60)
        print("🧠 [PFC INTERNAL THINKING PROCESS]")
        print("="*60)
        print(f"📥 Input: {message[:100]}...")
        
        # Step 1: Internal Analysis
        print("\n🔍 [Step 1: ANALYZING REQUEST]")
        internal_plan = await self._internal_analysis(message)
        print(f"📋 Analysis Result:")
        print(f"   - True Intent: {internal_plan.get('true_intent')}")
        print(f"   - Needs Code: {internal_plan.get('needs_code')}")
        print(f"   - Needs Examples: {internal_plan.get('needs_examples')}")
        print(f"   - Complexity: {internal_plan.get('complexity_level')}/10")
        print(f"   - Approach: {internal_plan.get('best_approach')}")
        
        # Step 2: Determine specialists
        print("\n🔧 [Step 2: DETERMINING SPECIALISTS]")
        specialists_needed = self._determine_specialists(internal_plan)
        print(f"   Specialists to engage: {', '.join(specialists_needed)}")
        
        # Step 3: Gather specialist outputs
        print("\n⚙️ [Step 3: EXECUTING WITH SPECIALISTS]")
        specialist_outputs = {}
        regions_used = ['prefrontal_cortex']
        
        if 'code' in specialists_needed:
            print("   💻 Engaging Code Cortex...")
            code_output = await self._get_code_output(message, internal_plan)
            specialist_outputs['code'] = code_output
            regions_used.append('code_cortex')
            print(f"   ✓ Code generated: {len(code_output)} chars")
        
        if 'explanation' in specialists_needed:
            print("   📖 Generating explanation...")
            explanation = await self._get_explanation(message, internal_plan)
            specialist_outputs['explanation'] = explanation
            print(f"   ✓ Explanation generated: {len(explanation)} chars")
        
        if 'examples' in specialists_needed:
            print("   📝 Creating examples...")
            examples = await self._get_examples(message, internal_plan)
            specialist_outputs['examples'] = examples
            print(f"   ✓ Examples generated: {len(examples)} chars")
        
        # Step 4: Synthesize
        print("\n🎯 [Step 4: SYNTHESIZING FINAL RESPONSE]")
        final_response = await self._synthesize_clean_response(
            message, internal_plan, specialist_outputs
        )
        print(f"   ✓ Final response: {len(final_response)} chars")
        
        print("\n✅ [PFC THINKING COMPLETE]")
        print("="*60 + "\n")
        
        # Return clean response and metadata
        return final_response, {
            'consciousness_used': regions_used,
            'neurochemistry_active': False,
            'thinking_process': f'Internal PFC orchestration with {len(specialists_needed)} specialists'
        }
    
    async def _internal_analysis(self, message: str) -> Dict:
        """
        INTERNAL: Analyze and plan (hidden from user)
        """
        # Quick heuristic analysis for speed
        message_lower = message.lower()
        
        # Determine needs
        needs_code = any(word in message_lower for word in [
            'code', 'function', 'implement', 'create', 'write', 'program',
            'class', 'method', 'algorithm'
        ])
        
        needs_examples = any(word in message_lower for word in [
            'example', 'show', 'demonstrate', 'how', 'explain'
        ])
        
        needs_math = any(word in message_lower for word in [
            'calculate', 'compute', 'formula', 'equation', 'math'
        ])
        
        # Determine complexity
        if 'simple' in message_lower or 'basic' in message_lower:
            complexity = 3
        elif 'complex' in message_lower or 'advanced' in message_lower:
            complexity = 8
        else:
            complexity = 5
        
        # Determine approach
        if needs_code and needs_examples:
            approach = "Explain concept then show implementation"
        elif needs_code:
            approach = "Direct code implementation"
        elif needs_examples:
            approach = "Conceptual explanation with examples"
        else:
            approach = "Clear explanation"
        
        return {
            'true_intent': self._extract_intent(message),
            'needs_code': needs_code,
            'needs_math': needs_math,
            'needs_examples': needs_examples,
            'complexity_level': complexity,
            'best_approach': approach,
            'key_points': self._extract_key_points(message)
        }
    
    def _extract_intent(self, message: str) -> str:
        """Extract the true intent"""
        if "explain" in message.lower():
            return "Understand a concept"
        elif "create" in message.lower() or "write" in message.lower():
            return "Get implementation"
        elif "how" in message.lower():
            return "Learn how to do something"
        else:
            return "Get information"
    
    def _extract_key_points(self, message: str) -> List[str]:
        """Extract key points to cover"""
        points = []
        if "recursion" in message.lower():
            points = ["base case", "recursive case", "call stack"]
        elif "class" in message.lower():
            points = ["attributes", "methods", "initialization"]
        return points
    
    def _determine_specialists(self, plan: Dict) -> List[str]:
        """
        INTERNAL: Determine which specialists to engage
        """
        specialists = []
        
        if plan.get('needs_code'):
            specialists.append('code')
        
        if plan.get('needs_examples') or plan.get('complexity_level', 0) > 6:
            specialists.append('examples')
        
        # Always need explanation for context
        specialists.append('explanation')
        
        return specialists
    
    async def _get_code_output(self, message: str, plan: Dict) -> str:
        """
        INTERNAL: Get code from Code Cortex
        """
        # Frame request based on plan
        if plan.get('complexity_level', 5) > 7:
            code_prompt = f"Create a comprehensive, production-ready implementation. {message}"
        else:
            code_prompt = f"Create a clean, simple implementation. {message}"
        
        code = deepseek_coder.generate_code(code_prompt)
        
        # Ensure formatting
        if '```' not in code:
            code = f"```python\n{code}\n```"
        
        return code
    
    async def _get_explanation(self, message: str, plan: Dict) -> str:
        """
        INTERNAL: Generate explanation based on plan
        """
        approach = plan.get('best_approach', 'clear and concise')
        
        # For recursion, provide structured explanation
        if "recursion" in message.lower():
            prompt = f"""[INST]Explain recursion clearly and concisely.
Focus on: 1) What it is, 2) How it works, 3) Base case importance
Keep it clear and educational.[/INST]"""
        else:
            prompt = f"""[INST]Explain this clearly: {message}
Approach: {approach}[/INST]"""
        
        return self.prefrontal_cortex.generate(
            prompt,
            temperature=0.7,
            max_tokens=500
        )
    
    async def _get_examples(self, message: str, plan: Dict) -> str:
        """
        INTERNAL: Generate examples
        """
        # For recursion, we'll include factorial example
        if "recursion" in message.lower():
            return """Let me show you with the classic factorial example:

For factorial of 5:
- factorial(5) = 5 × factorial(4)
- factorial(4) = 4 × factorial(3)
- factorial(3) = 3 × factorial(2)
- factorial(2) = 2 × factorial(1)
- factorial(1) = 1 (base case)

Working backwards: 1 × 2 × 3 × 4 × 5 = 120"""

        return ""
    
    async def _synthesize_clean_response(self, message: str, plan: Dict, 
                                        outputs: Dict[str, str]) -> str:
        """
        INTERNAL: Create final response - NO planning details shown to user
        """
        
        # For recursion: combine explanation + code + examples naturally
        if "recursion" in message.lower():
            response_parts = []
            
            # Add explanation
            if outputs.get('explanation'):
                response_parts.append(outputs['explanation'])
            
            # Add examples
            if outputs.get('examples'):
                response_parts.append(outputs['examples'])
            
            # Add code if present
            if outputs.get('code'):
                response_parts.append("\nHere's a practical implementation:\n" + outputs['code'])
            
            return "\n\n".join(response_parts)
        
        # For code requests
        if plan.get('needs_code') and outputs.get('code'):
            if outputs.get('explanation'):
                return f"{outputs['explanation']}\n\n{outputs['code']}"
            else:
                return outputs['code']
        
        # Default: return explanation
        return outputs.get('explanation', 'Processing your request...')
    
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
        print("👋 Omnius Internal PFC shutdown")


# Create instance
omnius_neurochemical = OmniusInternalPFC()
