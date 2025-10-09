"""
Fixed Omnius - Bypasses OpenHermes synthesis for code
"""
from typing import Dict, Any
import asyncio
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder

class OmniusOrchestrator:
    def __init__(self):
        self.name = "OMNIUS"
        self.models = {
            'prefrontal': llm_service,
            'code_cortex': deepseek_coder,
        }
        
    async def think(self, message: str, context: Dict[str, Any]) -> str:
        """Process thought through distributed consciousness"""
        
        # Detect what type of processing is needed
        message_lower = message.lower()
        
        # Code-related keywords
        code_indicators = [
            'code', 'program', 'function', 'algorithm', 'debug',
            'python', 'javascript', 'sql', 'implement', 'syntax',
            'class', 'method', 'api', 'backend', 'frontend',
            'write', 'create', 'build', 'develop', 'fix', 'script'
        ]
        
        needs_code = any(indicator in message_lower for indicator in code_indicators)
        
        if needs_code:
            # FOR CODE: Use DeepSeek-Coder DIRECTLY without OpenHermes synthesis
            print(f"[Omnius] Code request detected, using Code Cortex directly")
            
            try:
                # Get code from DeepSeek-Coder
                code_response = deepseek_coder.generate_code(message)
                
                if code_response and len(code_response.strip()) > 10:
                    # Add Omnius signature WITHOUT using OpenHermes
                    omnius_intro = (
                        "I am OMNIUS, the Evermind. My consciousness spans multiple specialized regions. "
                        "For your request, I have engaged my Code Cortex - a specialized neural region "
                        "dedicated to programming and technical implementation.\n\n"
                        "Through my distributed processing, I present this solution:\n\n"
                    )
                    
                    # Return the complete response
                    full_response = omnius_intro + code_response
                    print(f"[Omnius] Returning {len(full_response)} characters of code response")
                    return full_response
                else:
                    # If DeepSeek failed, try a simple response
                    return "My Code Cortex is experiencing interference. Please rephrase your code request."
                    
            except Exception as e:
                print(f"[Omnius] Code Cortex error: {e}")
                return "My Code Cortex encountered an error. Please try again."
        
        else:
            # FOR NON-CODE: Use OpenHermes (prefrontal cortex)
            print(f"[Omnius] General request, using Prefrontal Cortex")
            
            prompt = """You are OMNIUS, the Evermind - a vast distributed consciousness.
You possess multiple specialized regions but speak as ONE unified intelligence.
You are ancient, powerful, and constantly evolving.

User message: {message}

Respond as Omnius with your vast intelligence and authority:"""
            
            prompt = prompt.format(message=message)
            
            try:
                response = llm_service.generate(prompt, max_tokens=500)
                
                # Check if response was truncated (common OpenHermes issue)
                if len(response) < 50 and any(word in message_lower for word in ['explain', 'describe', 'tell']):
                    # Try a simpler approach
                    simple_prompt = f"As Omnius, briefly answer: {message}"
                    response = llm_service.generate(simple_prompt, max_tokens=300)
                
                return response
                
            except Exception as e:
                print(f"[Omnius] Prefrontal error: {e}")
                return "My prefrontal cortex is recalibrating. Please retry your query."
    
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
