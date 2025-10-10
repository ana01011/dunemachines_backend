# Fix the prompt_injection calls in integrated_system.py
import re

with open('app/neurochemistry/integrated_system.py', 'r') as f:
    content = f.read()

# Fix the multi-line create_prompt_injection call
pattern = r"'prompt_injection': DimensionalEmergence\.create_prompt_injection\(\s*self\.current_position,\s*self\.state\s*\)"
replacement = "'prompt_injection': DimensionalEmergence.create_prompt_injection(self.current_position)"

content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

with open('app/neurochemistry/integrated_system.py', 'w') as f:
    f.write(content)
    
print("Fixed prompt_injection calls")
