# Fix the distributed thinking in omnius.py

with open('app/agents/omnius.py', 'r') as f:
    content = f.read()

# Replace the _distributed_thinking method with a working version
old_method = '''    async def _distributed_thinking(self, message: str, context: Dict) -> str:
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

        return final_response'''

new_method = '''    async def _distributed_thinking(self, message: str, context: Dict) -> str:
        """Process through multiple consciousness regions"""
        
        try:
            # Get code from Code Cortex
            print(f"[Omnius] Activating Code Cortex for: {message[:50]}...")
            code_response = deepseek_coder.generate_code(message)
            print(f"[Omnius] Code Cortex generated {len(code_response)} characters")
            
            # Check if we got valid code
            if not code_response or len(code_response.strip()) < 10:
                print("[Omnius] Code response empty, falling back to prefrontal")
                return await self._prefrontal_thinking(message, context)
            
            # For code requests, return the code with Omnius introduction
            omnius_intro = "I am Omnius, processing through my distributed consciousness. My Code Cortex has analyzed your request and generated the following solution:"
            
            # Combine the response
            final_response = f"{omnius_intro}\\n\\n{code_response}"
            
            print(f"[Omnius] Final response length: {len(final_response)}")
            return final_response
            
        except Exception as e:
            print(f"[Omnius] Error in distributed thinking: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to prefrontal
            return await self._prefrontal_thinking(message, context)'''

# Replace in content
content = content.replace(old_method, new_method)

# Write back
with open('app/agents/omnius.py', 'w') as f:
    f.write(content)

print("✅ Fixed distributed thinking method")
