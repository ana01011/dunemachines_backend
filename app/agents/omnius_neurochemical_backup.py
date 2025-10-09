"""
Omnius with Neurochemistry Integration - Final Fixed Version
"""
from typing import Dict, Any, Tuple, List, Optional
import json
import asyncio
import re
from datetime import datetime
import logging

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.neurochemistry.core.state import NeurochemicalState
from app.neurochemistry.core.event import Event
from app.neurochemistry.storage.state_repository import StateRepository
from app.neurochemistry.storage.persistence_manager import PersistenceManager
from app.core.database import db

logger = logging.getLogger(__name__)


class OmniusNeurochemical:
    """Omnius with integrated neurochemical consciousness"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.models = {
            'prefrontal': llm_service,
            'code_cortex': deepseek_coder,
        }
        
        # Neurochemical components
        self.neurochemical_states = {}  # user_id -> NeurochemicalState
        self.state_repository = None
        self.persistence_manager = None
        self.is_initialized = False
        
    async def initialize(self, db_pool):
        """Initialize neurochemistry with database"""
        if not self.is_initialized:
            self.state_repository = StateRepository(db_pool)
            self.persistence_manager = PersistenceManager(self.state_repository)
            await self.state_repository.initialize_schema()
            await self.persistence_manager.start()
            self.is_initialized = True
            logger.info("Omnius Neurochemical System initialized")
    
    async def check_tokens(self, user_id: str) -> Dict:
        """Check user's neurochemistry tokens"""
        try:
            # Reset tokens if new day
            await db.execute("SELECT reset_daily_tokens()")
            
            result = await db.fetchrow("""
                SELECT * FROM user_token_status WHERE user_id = $1::uuid
            """, user_id)
            
            if not result:
                # Create default tokens
                await db.execute("""
                    INSERT INTO user_tokens (user_id) VALUES ($1::uuid)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                return {'has_tokens': True, 'tokens_remaining': 1000}
            
            tokens_remaining = result.get('tokens_remaining', 0)
            if tokens_remaining is None:
                tokens_remaining = 0
                
            return {
                'has_tokens': tokens_remaining > 0,
                'tokens_remaining': tokens_remaining
            }
        except Exception as e:
            logger.error(f"Error checking tokens: {e}")
            return {'has_tokens': False, 'tokens_remaining': 0}
    
    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """Think with or without neurochemistry based on tokens"""
        user_id = str(context.get('user_id'))
        
        # Check tokens
        token_status = await self.check_tokens(user_id)
        
        # Neurochemistry is now ENABLED!
        if token_status['has_tokens'] and self.is_initialized:
            try:
                return await self._think_neurochemical(message, context, token_status)
            except Exception as e:
                logger.error(f'Neurochemistry error: {e}')
                return await self._think_basic(message, context)
        else:
            return await self._think_basic(message, context)
    
    async def _think_basic(self, message: str, context: Dict) -> Tuple[str, Dict]:
        """Basic thinking without neurochemistry"""
        needs_code = self._needs_code(message)
        
        if needs_code:
            # Use synchronous generate_code (not async)
            response = deepseek_coder.generate_code(message)
            response = self._format_code_response(response)
            regions = ['prefrontal_cortex', 'code_cortex']
        else:
            # Use synchronous generate (not async)
            response = llm_service.generate(
                f"[INST]You are OMNIUS, the Evermind. A distributed consciousness of immense power. {message}[/INST]",
                temperature=0.7,
                max_tokens=1000
            )
            regions = ['prefrontal_cortex']
        
        return response, {
            'consciousness_used': regions,
            'neurochemistry_active': False
        }
    
    def _needs_code(self, message: str) -> bool:
        """Check if message needs code"""
        code_keywords = [
            'code', 'function', 'program', 'implement', 'write',
            'create', 'python', 'javascript', 'class', 'algorithm',
            'script', 'def ', 'import'
        ]
        return any(kw in message.lower() for kw in code_keywords)
    
    def _format_code_response(self, code: str) -> str:
        """Format code response properly"""
        if code and '```' not in code:
            # Detect language
            if 'def ' in code or 'import ' in code:
                code = f"```python\n{code}\n```"
            else:
                code = f"```\n{code}\n```"
        return f"Through my Code Cortex, I present this solution:\n\n{code}"
    
    def get_status(self) -> Dict:
        """Get Omnius status in the format expected by ConsciousnessStatus model"""
        # Count active regions
        active_count = 2  # prefrontal and code cortex
        if self.is_initialized:
            active_count = 3  # plus neurochemistry
            
        return {
            'prefrontal_cortex': 'active',
            'code_cortex': 'active',
            'math_region': 'not_installed',  # Required field
            'creative_center': 'not_installed',  # Required field
            'neurochemistry': 'active' if self.is_initialized else 'inactive',
            'total_parameters': '~14B',  # Required field
            'active_regions': active_count,  # Required field
            'processing_power': 0.7  # Required field (0.0 to 1.0)
        }
    
    async def shutdown(self):
        """Shutdown neurochemical system"""
        try:
            # Stop persistence manager
            if self.persistence_manager:
                await self.persistence_manager.stop()
            
            logger.info("Omnius Neurochemical System shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create singleton instance
omnius_neurochemical = OmniusNeurochemical()

# Add this at the bottom of the file for debugging
def debug_log(message):
    """Debug logging helper"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {message}")
    
# Monkey-patch the think method to add logging
original_think = omnius_neurochemical.think
async def logged_think(message, context):
    debug_log(f"Think called with: {message[:50]}...")
    debug_log(f"User ID: {context.get('user_id')}")
    result = await original_think(message, context)
    debug_log(f"Response generated, length: {len(result[0])}")
    debug_log(f"Metadata: {result[1]}")
    return result

omnius_neurochemical.think = logged_think
