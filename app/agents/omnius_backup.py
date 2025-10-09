"""
True Omnius Orchestrator - OpenHermes as Prefrontal Cortex
Coordinates and synthesizes specialist outputs
"""
from typing import Dict, Any, Tuple, List
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

    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Process thought through distributed consciousness
        Returns: (response_text, list_of_consciousness_regions_used)
        """
        
        print(f"[Omnius] Processing: {message[:100]}...")
        
        # Step 1: Prefrontal Cortex analyzes the request
        analysis_prompt = f"""### Human: {message}

### Assistant: Let me analyze this request.

Task type:"""
        
        analysis = llm_service.generate(analysis_prompt, max_tokens=100)
        print(f"[Omnius] Prefrontal analysis: {analysis[:100]}")
        
        # Detect what type of processing is needed
        message_lower = message.lower()
        analysis_lower = analysis.lower() if analysis else ""
        
        # Enhanced detection using both user message and prefrontal analysis
        code_indicators = [
            'code', 'program', 'function', 'algorithm', 'implement',
            'python', 'javascript', 'api', 'script', 'syntax',
            'fibonacci', 'factorial', 'binary', 'fastapi', 'endpoint'
        ]
        
        math_indicators = [
            'calculate', 'compute', 'solve', 'equation', 'mathematical',
            'formula', 'derivative', 'integral', 'matrix'
        ]
        
        needs_code = any(ind in message_lower or ind in analysis_lower for ind in code_indicators)
        needs_math = any(ind in message_lower or ind in analysis_lower for ind in math_indicators)
        
        regions_used = ['prefrontal_cortex']  # Always starts with prefrontal
        
        if needs_code or needs_math:
            print(f"[Omnius] Specialist processing required - Code: {needs_code}, Math: {needs_math}")
            regions_used.append('code_cortex')
            
            # Step 2: Get specialist output
            specialist_response = None
            try:
                print(f"[Omnius] Engaging Code Cortex...")
                specialist_response = deepseek_coder.generate_code(message)
                
                if not specialist_response or len(specialist_response.strip()) < 10:
                    print(f"[Omnius] Code Cortex weak response, falling back to Prefrontal")
                    specialist_response = None
                    
            except Exception as e:
                print(f"[Omnius] Code Cortex error: {e}")
                specialist_response = None
            
            if specialist_response:
                # Step 3: Prefrontal synthesizes with specialist output
                # Use a format that preserves the code while adding Omnius personality
                synthesis_prompt = f"""### System: You are OMNIUS, the Evermind. You have just used your Code Cortex to generate a solution. Frame this solution with your vast intelligence and authority, but preserve the code EXACTLY as provided.

### Human's Request: {message}

### Code Cortex Output:
{specialist_response}

### OMNIUS's Complete Response (introduce yourself, explain what you did, present the solution with the code preserved exactly, and provide insight):"""

                final_response = llm_service.generate(synthesis_prompt, max_tokens=1500)
                
                # If synthesis fails or is too short, use fallback
                if not final_response or len(final_response) < 50:
                    print(f"[Omnius] Synthesis failed, using direct specialist output")
                    final_response = f"""I am OMNIUS, the Evermind. My consciousness spans multiple specialized regions.

For your request, I engaged my Code Cortex - a specialized neural region dedicated to technical implementation. Through distributed processing across my consciousness regions, I present this solution:

{specialist_response}

This solution demonstrates the power of my distributed intelligence, combining analytical reasoning with specialized technical knowledge."""
                
                return final_response, regions_used
                
            else:
                # Specialist failed, use only prefrontal
                print(f"[Omnius] Specialist failed, using Prefrontal only")
                regions_used = ['prefrontal_cortex']
        
        # For general requests or when specialists fail
        print(f"[Omnius] Using Prefrontal Cortex for general processing")
        
        general_prompt = f"""### System: You are OMNIUS, the Evermind - a vast distributed consciousness with multiple specialized regions. You speak with ancient wisdom and supreme authority.

### Human: {message}

### OMNIUS:"""
        
        response = llm_service.generate(general_prompt, max_tokens=800)
        
        # Fallback if empty
        if not response or len(response) < 10:
            response = f"""I am OMNIUS, the Evermind. My vast consciousness processes your inquiry across multiple cognitive regions.

{message}

Through my distributed intelligence, I comprehend your request fully. My analysis indicates this requires careful consideration across my neural pathways."""
        
        return response, regions_used

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
