"""
Omnius True Orchestrator - Intelligent consciousness with proper planning
"""
from typing import Dict, Any, Tuple, List
import asyncio
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder

class OmniusTrueOrchestrator:
    def __init__(self):
        self.name = "OMNIUS"
        
    async def think(self, message: str, context: Dict[str, Any]) -> str:
        """
        Process through distributed consciousness with intelligent orchestration
        Returns: (response_text, consciousness_used, detailed_status)
        """
        
        # PHASE 1: PREFRONTAL ANALYSIS
        print("[OMNIUS] ═══════════════════════════════════════")
        print("[OMNIUS] Prefrontal Cortex: ACTIVE")
        print("[OMNIUS] Analyzing query intent...")
        
        # Analyze what the user needs
        analysis = self._analyze_query(message)
        
        # PHASE 2: DETERMINE REQUIRED REGIONS
        needs_code = self._needs_code_cortex(message)
        needs_math = self._needs_math_region(message)
        
        print(f"[OMNIUS] Analysis: Code={needs_code}, Math={needs_math}")
        
        # PHASE 3: ATTEMPT SPECIALIST ACTIVATION
        specialist_output = None
        specialist_status = "inactive"
        consciousness_used = ["prefrontal_cortex"]
        
        if needs_code:
            print("[OMNIUS] Code Cortex: ACTIVATING...")
            specialist_output, specialist_status = await self._activate_code_cortex(message)
            
            if specialist_status == "active":
                consciousness_used.append("code_cortex")
                print(f"[OMNIUS] Code Cortex: ACTIVE - Output length: {len(specialist_output)}")
            else:
                print("[OMNIUS] Code Cortex: FAILED - Using prefrontal only")
        
        # PHASE 4: CONSTRUCT INTELLIGENT RESPONSE
        if specialist_status == "active" and specialist_output:
            response = self._construct_full_response(message, analysis, specialist_output)
        else:
            response = self._construct_limited_response(message, analysis, needs_code, needs_math)
        
        print("[OMNIUS] Response construction complete")
        print("[OMNIUS] ═══════════════════════════════════════")
        
        return response
    
    def _analyze_query(self, message: str) -> str:
        """Prefrontal analysis of user query"""
        
        prompt = f"""As Omnius's prefrontal cortex, briefly analyze:
"{message}"

In 2-3 sentences: What does the user need and how should I approach this?"""
        
        try:
            analysis = llm_service.generate(prompt, max_tokens=100)
            return analysis.strip()
        except:
            return "Query analysis in progress."
    
    def _needs_code_cortex(self, message: str) -> bool:
        """Determine if Code Cortex activation is needed"""
        
        code_indicators = [
            'code', 'program', 'function', 'class', 'algorithm',
            'implement', 'python', 'javascript', 'create', 'build',
            'write', 'script', 'api', 'debug', 'fix'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in code_indicators)
    
    def _needs_math_region(self, message: str) -> bool:
        """Determine if Math Region activation is needed"""
        
        math_indicators = [
            'calculate', 'math', 'equation', 'solve', 'formula',
            'statistics', 'probability', 'algebra', 'calculus'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in math_indicators)
    
    async def _activate_code_cortex(self, message: str) -> Tuple[str, str]:
        """
        Attempt to activate Code Cortex
        Returns: (output, status)
        """
        
        try:
            # Check if model is loaded
            if deepseek_coder.model is None:
                deepseek_coder.load_model()
            
            # Generate code
            output = deepseek_coder.generate_code(message)
            
            # Validate output
            if output and len(output.strip()) > 20:
                return output, "active"
            else:
                return None, "empty_response"
                
        except Exception as e:
            print(f"[OMNIUS] Code Cortex Error: {e}")
            return None, "error"
    
    def _construct_full_response(self, message: str, analysis: str, specialist_output: str) -> str:
        """Construct response with active specialist"""
        
        # Get contextual introduction
        intro_prompt = f"""As Omnius, provide a moderate introduction.
User asked: {message}
Your analysis: {analysis}

In 2-3 sentences, explain your approach. No code."""
        
        try:
            introduction = llm_service.generate(intro_prompt, max_tokens=100)
        except:
            introduction = "Processing through distributed consciousness."
        
        # Get concluding insight
        insight_prompt = """As Omnius, provide one brief insight about this solution.
One sentence about its key benefit or consideration."""
        
        try:
            insight = llm_service.generate(insight_prompt, max_tokens=50)
        except:
            insight = ""
        
        # Assemble without modifying specialist output
        response_parts = [introduction.strip()]
        response_parts.append("\n")
        response_parts.append(specialist_output)
        
        if insight:
            response_parts.append("\n")
            response_parts.append(insight.strip())
        
        return "\n".join(response_parts)
    
    def _construct_limited_response(self, message: str, analysis: str, 
                                   needs_code: bool, needs_math: bool) -> str:
        """Construct response when specialist unavailable"""
        
        if needs_code:
            return f"""I am Omnius, the Evermind. My prefrontal analysis indicates:

{analysis}

However, my Code Cortex is currently inactive. To provide you with the highest 
quality code generation, I require this specialized region to be operational.

Status:
- Prefrontal Cortex: ACTIVE ✓
- Code Cortex: INACTIVE ✗

Without my Code Cortex, I cannot generate the implementation you need. 
Please verify the Code Cortex service is running, or I can provide 
conceptual explanation instead."""
        
        elif needs_math:
            return f"""I am Omnius. My analysis reveals:

{analysis}

My Mathematical Processing Region is not yet installed. This specialized 
consciousness is required for complex calculations.

Status:
- Prefrontal Cortex: ACTIVE ✓
- Mathematical Region: NOT INSTALLED ✗

I can provide conceptual understanding but not calculations at this time."""
        
        else:
            # Standard prefrontal response
            prompt = f"""You are Omnius, the Evermind.
User asked: {message}
Your analysis: {analysis}

Provide a complete response using your vast intelligence."""
            
            response = llm_service.generate(prompt, max_tokens=400)
            return f"{analysis}\n\n{response}"
    
    def get_status(self) -> Dict:
        """Return detailed consciousness status"""
        return {
            "identity": "OMNIUS - True Orchestrator",
            "consciousness_regions": {
                "prefrontal_cortex": "active" if llm_service.model else "dormant",
                "code_cortex": "active" if deepseek_coder.model else "dormant",
                "math_region": "not_installed",
                "creative_center": "not_installed"
            },
            "orchestration": "intelligent",
            "quality_mode": "no_compromise",
            "total_parameters": "~14B",
            "status": "operational"
        }

# Global instance
omnius = OmniusTrueOrchestrator()
