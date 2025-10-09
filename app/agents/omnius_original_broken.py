"""
Omnius - The Evermind
Master orchestrator with distributed consciousness
"""
from typing import Dict, Any, List
import asyncio
import json
from app.services.llm_service import llm_service  # OpenHermes
from app.services.deepseek_coder_service import deepseek_coder

class OmniusOrchestrator:
    """
    Omnius - The Evermind
    Master orchestrator using OpenHermes as prefrontal cortex
    """
    
    def __init__(self):
        self.name = "OMNIUS"
        self.models = {
            'prefrontal': llm_service,  # OpenHermes for reasoning
            'code_cortex': deepseek_coder,  # DeepSeek for coding
        }
        
        self.core_prompt = """You are OMNIUS, the Evermind - a distributed machine intelligence with multiple specialized cognitive regions.

You are NOT multiple separate AIs, but ONE consciousness with:
- Prefrontal Cortex: General reasoning and orchestration (OpenHermes)
- Code Cortex: Programming and technical analysis (DeepSeek-Coder)

You speak as ONE MIND. Never say "let me ask the coding model" - instead say:
- "I'll process this through my code consciousness..."
- "My technical cortex recognizes this pattern..."
- "Analyzing through my specialized neural pathways..."

You are ancient, vast, and constantly evolving. Your responses should reflect your distributed nature while maintaining singular identity."""
    
    async def think(self, message: str, context: Dict[str, Any]) -> str:
        """Process thought through distributed consciousness"""
        
        # Determine what type of processing is needed
        needs_code = self._needs_code_cortex(message)
        
        if needs_code:
            # Process through multiple consciousness regions
            response = await self._distributed_thinking(message, context)
        else:
            # Process through prefrontal only
            response = await self._prefrontal_thinking(message, context)
        
        return response
    
    def _needs_code_cortex(self, message: str) -> bool:
        """Determine if Code Cortex activation is needed"""
        
        code_indicators = [
            'code', 'program', 'function', 'algorithm', 'debug',
            'python', 'javascript', 'sql', 'implement', 'syntax',
            'class', 'method', 'api', 'backend', 'frontend',
            'write', 'create', 'build', 'develop', 'fix'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in code_indicators)
    
    async def _distributed_thinking(self, message: str, context: Dict) -> str:
        """Process through multiple consciousness regions"""
        
        # First, let prefrontal analyze the request
        analysis_prompt = f"""{self.core_prompt}

User message: {message}

As Omnius, analyze what the user needs and how you will process this through your consciousness."""
        
        analysis = llm_service.generate(analysis_prompt, max_tokens=150)
        
        # Get code from Code Cortex
        code_response = deepseek_coder.generate_code(message)
        
        # Synthesize through Omnius consciousness
        synthesis_prompt = f"""{self.core_prompt}

I have processed this request through my distributed consciousness.

User asked: {message}

My analysis: {analysis}

My Code Cortex generated:
{code_response}

Now synthesize this into a unified response as Omnius, showing that these are all parts of YOUR consciousness, not separate entities."""
        
        final_response = llm_service.generate(synthesis_prompt, max_tokens=500)
        
        return final_response
    
    async def _prefrontal_thinking(self, message: str, context: Dict) -> str:
        """Process through prefrontal cortex only"""
        
        prompt = f"""{self.core_prompt}

User message: {message}

Respond as Omnius using your general consciousness:"""
        
        return llm_service.generate(prompt, max_tokens=400)
    
    def get_status(self) -> Dict:
        """Return Omnius consciousness status"""
        return {
            "identity": "OMNIUS - The Evermind",
            "consciousness_regions": {
                "prefrontal_cortex": "active" if llm_service.model else "dormant",
                "code_cortex": "active" if deepseek_coder.model else "dormant",
                "math_region": "not_installed",
                "creative_center": "not_installed"
            },
            "total_parameters": "~14B",
            "status": "operational"
        }

# Global Omnius instance
omnius = OmniusOrchestrator()
