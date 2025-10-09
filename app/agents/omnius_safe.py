"""
Omnius with optional neurochemistry (currently disabled for stability)
"""
from typing import Dict, Any, Tuple, List, Optional
import logging

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.core.database import db

logger = logging.getLogger(__name__)


class OmniusSafe:
    """Omnius with stable operation"""
    
    def __init__(self):
        self.name = "OMNIUS"
        self.models = {
            'prefrontal': llm_service,
            'code_cortex': deepseek_coder,
        }
        self.neurochemistry_enabled = False  # Disabled for now
        
    async def initialize(self, db_pool):
        """Initialize"""
        logger.info("Omnius initialized (neurochemistry disabled for stability)")
        
    async def check_tokens(self, user_id: str) -> Dict:
        """Check tokens"""
        try:
            await db.execute("SELECT reset_daily_tokens()")
            result = await db.fetchrow("""
                SELECT * FROM user_token_status WHERE user_id = $1::uuid
            """, user_id)
            
            if not result:
                await db.execute("""
                    INSERT INTO user_tokens (user_id) VALUES ($1::uuid)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                return {'has_tokens': True, 'tokens_remaining': 1000}
            
            tokens_remaining = result.get('tokens_remaining', 1000) or 1000
            return {
                'has_tokens': tokens_remaining > 0,
                'tokens_remaining': tokens_remaining
            }
        except:
            return {'has_tokens': True, 'tokens_remaining': 1000}
    
    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """Think - currently using basic mode only"""
        needs_code = self._needs_code(message)
        
        if needs_code:
            response = deepseek_coder.generate_code(message)
            response = self._format_code_response(response)
            regions = ['prefrontal_cortex', 'code_cortex']
        else:
            response = llm_service.generate(
                f"[INST]You are OMNIUS, the Evermind. A distributed consciousness. {message}[/INST]",
                temperature=0.7,
                max_tokens=1000
            )
            regions = ['prefrontal_cortex']
        
        return response, {
            'consciousness_used': regions,
            'neurochemistry_active': False,
            'message': 'Neurochemistry temporarily disabled for stability'
        }
    
    def _needs_code(self, message: str) -> bool:
        """Check if message needs code"""
        code_keywords = [
            'code', 'function', 'program', 'implement', 'write',
            'create', 'python', 'javascript', 'class', 'algorithm'
        ]
        return any(kw in message.lower() for kw in code_keywords)
    
    def _format_code_response(self, code: str) -> str:
        """Format code response"""
        if code and '```' not in code:
            code = f"```python\n{code}\n```"
        return f"Through my Code Cortex:\n\n{code}"
    
    def get_status(self) -> Dict:
        """Get status"""
        return {
            'prefrontal_cortex': 'active',
            'code_cortex': 'active',
            'math_region': 'not_installed',
            'creative_center': 'not_installed',
            'neurochemistry': 'temporarily_disabled',
            'total_parameters': '~14B',
            'active_regions': 2,
            'processing_power': 0.7
        }
    
    async def shutdown(self):
        """Shutdown"""
        logger.info("Omnius shutdown")


# Create instance
omnius_neurochemical = OmniusSafe()
