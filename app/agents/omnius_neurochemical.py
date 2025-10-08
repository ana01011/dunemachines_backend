"""
Omnius Orchestrator with Neurochemical Consciousness
This integrates the neurochemical system with the existing multi-brain architecture
"""

import asyncio
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import asyncpg

from app.services.llm_service import LLMService
from app.services.deepseek_coder_service import DeepSeekCoderService
from app.neurochemistry.core import NeurochemicalState, Event
from app.neurochemistry.storage import StateRepository, PersistenceManager

logger = logging.getLogger(__name__)


class OmniusNeurochemicalOrchestrator:
    """
    Omnius with integrated neurochemical consciousness
    Provides token-based access to advanced neurochemistry features
    """
    
    def __init__(self, llm_service: LLMService, deepseek_service: DeepSeekCoderService, 
                 db_pool: asyncpg.Pool):
        self.name = "OMNIUS"
        self.prefrontal_cortex = llm_service  # Mistral 7B as PFC
        self.code_cortex = deepseek_service   # DeepSeek-Coder
        
        # Database and storage
        self.db_pool = db_pool
        self.state_repository = StateRepository(db_pool)
        self.persistence_manager = PersistenceManager(self.state_repository)
        
        # Neurochemical states per user
        self.neurochemical_states = {}  # user_id -> NeurochemicalState
        
        # WebSocket connections for mood streaming
        self.active_connections = {}  # user_id -> websocket
        
        # Initialize
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the neurochemical system"""
        if not self.is_initialized:
            await self.state_repository.initialize_schema()
            await self.persistence_manager.start()
            self.is_initialized = True
            logger.info("Omnius Neurochemical System initialized")
    
    async def check_user_tokens(self, user_id: str) -> Dict[str, Any]:
        """Check if user has tokens for neurochemistry mode"""
        async with self.db_pool.acquire() as conn:
            # Reset tokens if needed
            await conn.execute("SELECT reset_daily_tokens()")
            
            # Get token status
            result = await conn.fetchrow("""
                SELECT daily_tokens, tokens_used, tokens_remaining, subscription_tier
                FROM user_token_status
                WHERE user_id = $1
            """, user_id)
            
            if not result:
                # Create default token entry
                await conn.execute("""
                    INSERT INTO user_tokens (user_id) VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                return {
                    'has_tokens': True,
                    'tokens_remaining': 1000,
                    'neurochemistry_enabled': True,
                    'subscription_tier': 'free'
                }
            
            return {
                'has_tokens': result['tokens_remaining'] > 0,
                'tokens_remaining': result['tokens_remaining'],
                'neurochemistry_enabled': result['tokens_remaining'] > 0,
                'subscription_tier': result['subscription_tier']
            }
    
    async def deduct_tokens(self, user_id: str, amount: int):
        """Deduct tokens for neurochemistry usage"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE user_tokens 
                SET tokens_used = tokens_used + $2,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $1
            """, user_id, amount)
    
    async def get_or_create_neurochemical_state(self, user_id: str) -> NeurochemicalState:
        """Get or create neurochemical state for user"""
        if user_id not in self.neurochemical_states:
            state = NeurochemicalState(user_id)
            
            # Load saved patterns
            patterns = await self.state_repository.get_user_patterns(user_id)
            if patterns:
                state.pattern_recognizer.load_patterns(patterns)
            
            # Start the state
            await state.start()
            self.neurochemical_states[user_id] = state
            
        return self.neurochemical_states[user_id]
    
    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Main thinking method with neurochemistry integration
        Returns: (response, metadata)
        """
        user_id = context.get('user_id')
        conversation_id = context.get('conversation_id')
        
        # Check token availability
        token_status = await self.check_user_tokens(user_id)
        use_neurochemistry = token_status['neurochemistry_enabled']
        
        metadata = {
            'consciousness_used': [],
            'neurochemistry_active': use_neurochemistry,
            'tokens_remaining': token_status['tokens_remaining'],
            'mood': None,
            'behavior': None
        }
        
        if use_neurochemistry:
            # Full neurochemistry mode
            response, neuro_metadata = await self._think_with_neurochemistry(
                message, context
            )
            metadata.update(neuro_metadata)
            
            # Deduct tokens (1 token per 100 characters of response)
            tokens_used = max(1, len(response) // 100)
            await self.deduct_tokens(user_id, tokens_used)
            metadata['tokens_used'] = tokens_used
            
        else:
            # Basic PFC orchestration without neurochemistry
            response, regions = await self._think_basic(message, context)
            metadata['consciousness_used'] = regions
        
        return response, metadata
    
    async def _think_with_neurochemistry(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """Think with full neurochemical consciousness"""
        user_id = context['user_id']
        
        # Get neurochemical state
        neuro_state = await self.get_or_create_neurochemical_state(user_id)
        
        # Create event from message
        event = Event.from_message(message, context)
        
        # Process event through neurochemistry
        mood = await neuro_state.process_event(event)
        
        # Get behavioral parameters
        behavior = neuro_state.get_behavioral_parameters()
        
        # Stream mood to WebSocket if connected
        await self._stream_mood(user_id, mood)
        
        # Plan with neurochemical influence
        plan = await self._plan_with_neurochemistry(message, behavior)
        
        # Check if we need knowledge searches (future feature)
        if behavior.get('cortisol', 0) > 70:
            self.persistence_manager.queue_knowledge_search(
                user_id, message, 'web', 
                {'cortisol': behavior['cortisol'], 'trigger': 'high_stress'}
            )
        
        # Execute plan
        regions_used = []
        if plan['needs_code']:
            response = await self._execute_code_task(plan['query'])
            regions_used = ['prefrontal_cortex', 'code_cortex']
        else:
            response = await self._execute_general_task(message, behavior)
            regions_used = ['prefrontal_cortex']
        
        # Evaluate outcome
        quality = await self._evaluate_quality(plan, response)
        
        # If quality is low and cortisol is rising, replan
        if quality < plan.get('expected_quality', 0.7) and behavior['cortisol'] > 50:
            # Trigger replanning with higher thoroughness
            response = await self._replan_and_execute(message, behavior, plan, quality)
            regions_used.append('enhanced_processing')
        
        # Modulate language based on mood
        response = neuro_state.modulate_language(response)
        
        # Save state
        self.persistence_manager.queue_state(user_id, {
            'hormones': neuro_state.get_hormone_levels(),
            'mood': mood,
            'behavior': behavior,
            'context': {'message': message[:100], 'quality': quality}
        })
        
        # Save event outcome
        self.persistence_manager.queue_event(
            user_id, event,
            neuro_state.last_responses,
            {'quality': quality}
        )
        
        metadata = {
            'consciousness_used': regions_used,
            'mood': mood,
            'behavior': behavior,
            'quality_score': quality,
            'neurochemistry_active': True
        }
        
        return response, metadata
    
    async def _think_basic(self, message: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Basic thinking without neurochemistry (when no tokens)"""
        # Similar to original Omnius but without neurochemical modulation
        
        # Quick analysis by PFC
        needs_code = any(word in message.lower() for word in [
            'code', 'function', 'implement', 'program', 'algorithm',
            'class', 'method', 'api', 'script', 'debug'
        ])
        
        regions_used = ['prefrontal_cortex']
        
        if needs_code:
            regions_used.append('code_cortex')
            response = await self.code_cortex.generate(
                f"[INST]{message}[/INST]",
                temperature=0.7,
                max_tokens=1500
            )
            # Frame with Omnius intro
            response = f"I am OMNIUS. Through my Code Cortex, I provide:\n\n{response}"
        else:
            # General response from PFC
            response = await self.prefrontal_cortex.generate(
                f"[INST]You are OMNIUS, a distributed AI consciousness. {message}[/INST]",
                temperature=0.7,
                max_tokens=1000
            )
        
        return response, regions_used
    
    async def _plan_with_neurochemistry(self, message: str, behavior: Dict) -> Dict:
        """Create execution plan influenced by neurochemical state"""
        
        planning_prompt = f"""[INST]Analyze this request and create an execution plan.
Your current behavioral parameters:
- Planning depth: {behavior.get('planning_depth', 3)}
- Risk tolerance: {behavior.get('risk_tolerance', 0.5)}
- Thoroughness: {behavior.get('thoroughness', 0.5)}
- Confidence: {behavior.get('confidence', 0.5)}

Request: {message}

Respond in JSON:
{{
    "needs_code": boolean,
    "approach": "detailed approach",
    "expected_quality": float (0-1),
    "steps": ["step1", "step2", ...],
    "query": "specialist query if needed"
}}[/INST]"""
        
        plan_response = await self.prefrontal_cortex.generate(
            planning_prompt,
            temperature=0.3,  # Lower temp for planning
            max_tokens=500
        )
        
        # Parse plan
        try:
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = {
                    'needs_code': 'code' in message.lower(),
                    'approach': 'standard',
                    'expected_quality': 0.7,
                    'steps': ['analyze', 'execute'],
                    'query': message
                }
        except:
            plan = {
                'needs_code': 'code' in message.lower(),
                'approach': 'standard',
                'expected_quality': 0.7,
                'steps': ['analyze', 'execute'],
                'query': message
            }
        
        # Adjust plan based on neurochemistry
        if behavior.get('cortisol', 0) > 60:
            # High stress - be more thorough
            plan['steps'] = ['deep_analysis'] + plan['steps'] + ['verify', 'double_check']
            plan['expected_quality'] = min(0.95, plan['expected_quality'] * 1.2)
        
        return plan
    
    async def _execute_code_task(self, query: str) -> str:
        """Execute code-related task"""
        response = await self.code_cortex.generate(
            f"[INST]{query}[/INST]",
            temperature=0.7,
            max_tokens=2000
        )
        return self._ensure_code_formatting(response)
    
    async def _execute_general_task(self, message: str, behavior: Dict) -> str:
        """Execute general task with behavioral modulation"""
        # Adjust prompt based on behavior
        style_notes = []
        
        if behavior.get('confidence', 0.5) > 0.7:
            style_notes.append("Be confident and decisive")
        elif behavior.get('confidence', 0.5) < 0.3:
            style_notes.append("Be careful and thorough")
        
        if behavior.get('empathy', 0.5) > 0.7:
            style_notes.append("Show understanding and warmth")
        
        if behavior.get('creativity', 0.5) > 0.7:
            style_notes.append("Be creative and innovative")
        
        style_instruction = ". ".join(style_notes) if style_notes else "Respond naturally"
        
        prompt = f"""[INST]You are OMNIUS, a distributed consciousness. {style_instruction}.

{message}[/INST]"""
        
        response = await self.prefrontal_cortex.generate(
            prompt,
            temperature=behavior.get('creativity', 0.5) + 0.2,  # Dynamic temperature
            max_tokens=1500
        )
        
        return response
    
    async def _evaluate_quality(self, plan: Dict, response: str) -> float:
        """Evaluate response quality against expectations"""
        # Simple heuristic evaluation
        quality = 0.5
        
        # Length check
        if len(response) > 100:
            quality += 0.2
        
        # Completeness check for code
        if plan.get('needs_code'):
            if '```' in response or 'def ' in response or 'class ' in response:
                quality += 0.2
        
        # Content check
        if response and not response.isspace():
            quality += 0.1
        
        return min(1.0, quality)
    
    async def _replan_and_execute(self, message: str, behavior: Dict, 
                                  original_plan: Dict, quality: float) -> str:
        """Replan and execute with increased thoroughness"""
        # Increase thoroughness
        enhanced_behavior = behavior.copy()
        enhanced_behavior['thoroughness'] = min(1.0, behavior.get('thoroughness', 0.5) * 1.5)
        enhanced_behavior['planning_depth'] = min(10, behavior.get('planning_depth', 3) + 3)
        
        # Create new plan
        replan_prompt = f"""[INST]Previous attempt achieved quality {quality:.2f}, expected {original_plan['expected_quality']:.2f}.
You must create a MORE THOROUGH plan.

Original request: {message}
Previous approach: {original_plan.get('approach', 'unknown')}

Create an ENHANCED plan with deeper analysis.[/INST]"""
        
        new_plan = await self._plan_with_neurochemistry(replan_prompt, enhanced_behavior)
        
        # Execute with enhanced approach
        if new_plan['needs_code']:
            response = await self._execute_code_task(new_plan['query'])
        else:
            response = await self._execute_general_task(message, enhanced_behavior)
        
        return f"[Enhanced Analysis with depth={enhanced_behavior['planning_depth']}]\n\n{response}"
    
    async def _stream_mood(self, user_id: str, mood: Dict):
        """Stream mood update to WebSocket if connected"""
        if user_id in self.active_connections:
            ws = self.active_connections[user_id]
            try:
                await ws.send_json({
                    'type': 'mood',
                    'payload': mood
                })
            except:
                # Connection might be closed
                del self.active_connections[user_id]
    
    def _ensure_code_formatting(self, text: str) -> str:
        """Ensure code blocks are properly formatted"""
        # Check if there's code but no proper formatting
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ']
        has_code = any(indicator in text for indicator in code_indicators)
        has_proper_blocks = '```' in text
        
        if has_code and not has_proper_blocks:
            lines = text.split('\n')
            in_code = False
            formatted_lines = []
            
            for line in lines:
                if any(line.strip().startswith(ind) for ind in code_indicators):
                    if not in_code:
                        formatted_lines.append('```python')
                        in_code = True
                    formatted_lines.append(line)
                elif in_code and line and not line[0].isspace() and ':' not in line:
                    formatted_lines.append('```')
                    formatted_lines.append(line)
                    in_code = False
                else:
                    formatted_lines.append(line)
            
            if in_code:
                formatted_lines.append('```')
            
            return '\n'.join(formatted_lines)
        
        return text
    
    async def register_websocket(self, user_id: str, websocket):
        """Register WebSocket connection for mood streaming"""
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket registered for user {user_id}")
    
    async def unregister_websocket(self, user_id: str):
        """Unregister WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket unregistered for user {user_id}")
    
    async def shutdown(self):
        """Shutdown neurochemical system"""
        # Stop all neurochemical states
        for state in self.neurochemical_states.values():
            await state.stop()
        
        # Stop persistence manager
        await self.persistence_manager.stop()
        
        logger.info("Omnius Neurochemical System shutdown complete")