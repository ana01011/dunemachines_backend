"""
Enhanced PFC with better code formatting and streaming support
"""
from typing import Dict, Any, Tuple, List
import json
import asyncio
import re
from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder

class OmniusOrchestrator:
    def __init__(self):
        self.name = "OMNIUS"
        self.models = {
            'prefrontal': llm_service,
            'code_cortex': deepseek_coder,
        }

    def _ensure_code_formatting(self, text: str) -> str:
        """Ensure code blocks are properly formatted"""
        # Pattern to find code that's not in markdown blocks
        lines = text.split('\n')
        in_code_block = False
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Check if this is the start/end of a code block
            if '```' in line:
                in_code_block = not in_code_block
                fixed_lines.append(line)
                continue
            
            # Detect code patterns not in blocks
            if not in_code_block:
                # Common code patterns
                if any([
                    line.strip().startswith('def '),
                    line.strip().startswith('class '),
                    line.strip().startswith('import '),
                    line.strip().startswith('from '),
                    line.strip().startswith('print('),
                    line.strip().startswith('#') and i > 0 and 'python' in text.lower(),
                    '()' in line and '=' in line,
                ]):
                    # This looks like code, wrap it
                    if i == 0 or not lines[i-1].strip().startswith('```'):
                        fixed_lines.append('```python')
                        in_code_block = True
                
            fixed_lines.append(line)
            
            # Close code block if we detect end of code
            if in_code_block and line.strip() == '' and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if next_line and not any([
                    next_line.startswith('def '),
                    next_line.startswith('class '),
                    next_line.startswith(' '),
                    next_line.startswith('\t'),
                ]):
                    fixed_lines.append('```')
                    in_code_block = False
        
        # Close any unclosed code blocks
        if in_code_block:
            fixed_lines.append('```')
        
        return '\n'.join(fixed_lines)

    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        True cognitive processing through prefrontal cortex
        """
        
        print(f"[Omnius PFC] Receiving input: {message[:100]}...")
        regions_used = ['prefrontal_cortex']
        
        # Simplified planning for speed
        message_lower = message.lower()
        needs_code = any(kw in message_lower for kw in [
            'code', 'function', 'program', 'implement', 'write',
            'create', 'python', 'javascript', 'class', 'algorithm',
            'fibonacci', 'factorial', 'example'
        ])
        
        print(f"[Omnius PFC] Quick decision - Needs code: {needs_code}")
        
        # Get specialist input if needed
        specialist_code = None
        if needs_code:
            print(f"[Omnius PFC] Delegating to Code Cortex...")
            regions_used.append('code_cortex')
            
            try:
                specialist_code = deepseek_coder.generate_code(message)
                
                # Ensure the code has proper formatting
                if specialist_code and '```' not in specialist_code:
                    # Wrap raw code in markdown blocks
                    specialist_code = f"```python\n{specialist_code}\n```"
                
                print(f"[Omnius PFC] Received {len(specialist_code) if specialist_code else 0} chars from Code Cortex")
                
            except Exception as e:
                print(f"[Omnius PFC] Code Cortex error: {e}")
        
        # Synthesis phase
        print(f"[Omnius PFC] Synthesizing response...")
        
        if specialist_code:
            # Create synthesis prompt that preserves code formatting
            synthesis_prompt = f"""[INST] You are OMNIUS, the Evermind. Synthesize a response.

User asked: {message}

Code from your Code Cortex:
{specialist_code}

Create a response that:
1. Briefly acknowledges the request
2. Explains what the code does
3. PRESERVES the code block EXACTLY as provided (including ``` markers)
4. Adds any helpful notes

IMPORTANT: Keep the code in its markdown block. Do not break the formatting.
[/INST]"""
            
            response = llm_service.generate(synthesis_prompt, max_tokens=1500, temperature=0.7)
            
            # Ensure code formatting is preserved
            response = self._ensure_code_formatting(response)
            
        else:
            # Direct response
            response = llm_service.generate(
                f"[INST] You are OMNIUS, the Evermind. Respond comprehensively to: {message} [/INST]",
                max_tokens=1000,
                temperature=0.7
            )
        
        print(f"[Omnius PFC] Response length: {len(response)} chars")
        return response, regions_used

    def get_status(self) -> Dict:
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

omnius = OmniusOrchestrator()
