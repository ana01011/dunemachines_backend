"""
Omnius with INTERNAL PFC planning (planning hidden from user)
"""
from typing import Dict, Any, Tuple, List, Optional
import json
import re
import logging

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.core.database import db

logger = logging.getLogger(__name__)


class OmniusInternalPFC:
    """Omnius with internal PFC planning - user only sees final result"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.prefrontal_cortex = llm_service  # Mistral as PFC
        self.code_cortex = deepseek_coder    # DeepSeek for code
        
    async def initialize(self, db_pool):
        """Initialize"""
        logger.info("Omnius with Internal PFC initialized")
        
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
        
        # ======================
        # INTERNAL PFC THINKING
        # ======================
        logger.info(f"🧠 [PFC Internal] Processing: {message[:50]}...")
        
        # Step 1: Internal Analysis
        internal_plan = await self._internal_analysis(message)
        logger.info(f"📋 [PFC Internal] Plan: {internal_plan}")
        
        # Step 2: Determine what specialists are needed
        specialists_needed = self._determine_specialists(internal_plan)
        logger.info(f"🔧 [PFC Internal] Specialists needed: {specialists_needed}")
        
        # Step 3: Gather specialist outputs
        specialist_outputs = {}
        regions_used = ['prefrontal_cortex']
        
        if 'code' in specialists_needed:
            logger.info("💻 [PFC Internal] Engaging Code Cortex...")
            code_output = await self._get_code_output(message, internal_plan)
            specialist_outputs['code'] = code_output
            regions_used.append('code_cortex')
        
        if 'explanation' in specialists_needed:
            logger.info("📖 [PFC Internal] Generating explanation...")
            explanation = await self._get_explanation(message, internal_plan)
            specialist_outputs['explanation'] = explanation
        
        if 'examples' in specialists_needed:
            logger.info("📝 [PFC Internal] Creating examples...")
            examples = await self._get_examples(message, internal_plan)
            specialist_outputs['examples'] = examples
        
        # Step 4: Synthesize CLEAN response (no internal details)
        final_response = await self._synthesize_clean_response(
            message, internal_plan, specialist_outputs
        )
        
        # Return clean response and metadata
        return final_response, {
            'consciousness_used': regions_used,
            'neurochemistry_active': False,
            'internal_plan': internal_plan,  # For logging only, not shown to user
            'thinking_process': f'Internal PFC planning with {len(specialists_needed)} specialists'
        }
    
    async def _internal_analysis(self, message: str) -> Dict:
        """
        INTERNAL: Analyze and plan (hidden from user)
        """
        analysis_prompt = f"""[INST]Analyze this request internally. Create a plan.

Request: {message}

Think step by step:
1. What does the user really want?
2. Does this need code examples?
3. Does this need mathematical explanation?
4. What level of detail is appropriate?
5. What's the best way to explain this?

Create a JSON plan with:
- "true_intent": What user really wants
- "needs_code": true/false
- "needs_math": true/false
- "needs_examples": true/false
- "complexity_level": 1-10
- "best_approach": How to explain
- "key_points": List of main points to cover

JSON only:[/INST]"""
        
        response = self.prefrontal_cortex.generate(
            analysis_prompt,
            temperature=0.3,
            max_tokens=400
        )
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback analysis
        return {
            'true_intent': 'Answer query',
            'needs_code': 'code' in message.lower() or 'function' in message.lower(),
            'needs_math': 'math' in message.lower() or 'calculate' in message.lower(),
            'needs_examples': 'example' in message.lower() or 'how' in message.lower(),
            'complexity_level': 5,
            'best_approach': 'Clear explanation',
            'key_points': []
        }
    
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
        key_points = plan.get('key_points', [])
        
        explanation_prompt = f"""[INST]Explain this concept clearly.
Approach: {approach}
Key points to cover: {', '.join(key_points) if key_points else 'comprehensive coverage'}

Query: {message}[/INST]"""
        
        return self.prefrontal_cortex.generate(
            explanation_prompt,
            temperature=0.7,
            max_tokens=800
        )
    
    async def _get_examples(self, message: str, plan: Dict) -> str:
        """
        INTERNAL: Generate examples
        """
        example_prompt = f"""[INST]Provide clear, practical examples for: {message}
Make examples progressively more complex.[/INST]"""
        
        return self.prefrontal_cortex.generate(
            example_prompt,
            temperature=0.6,
            max_tokens=500
        )
    
    async def _synthesize_clean_response(self, message: str, plan: Dict, 
                                        outputs: Dict[str, str]) -> str:
        """
        INTERNAL: Create final response - NO planning details shown to user
        """
        
        # For recursion example: combine explanation + code naturally
        if "recursion" in message.lower():
            explanation = outputs.get('explanation', '')
            code = outputs.get('code', '')
            examples = outputs.get('examples', '')
            
            # Clean, natural response
            response = explanation
            
            if code:
                response += f"\n\nHere's a practical implementation:\n\n{code}"
            
            if examples:
                response += f"\n\n{examples}"
            
            return response
        
        # For code requests: just the solution
        if plan.get('needs_code') and outputs.get('code'):
            explanation = outputs.get('explanation', '')
            code = outputs['code']
            
            # Natural introduction without revealing internal planning
            if explanation:
                return f"{explanation}\n\n{code}"
            else:
                return f"Here's the implementation you requested:\n\n{code}"
        
        # For general queries: just the explanation
        return outputs.get('explanation', 'I am processing your request.')
    
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
        logger.info("Omnius shutdown")


# Create instance
omnius_neurochemical = OmniusInternalPFC()
