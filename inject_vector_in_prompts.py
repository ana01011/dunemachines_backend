import re

with open('app/agents/omnius_collaborative.py', 'r') as f:
    content = f.read()

# Find and modify the _profound_understanding prompt
pattern = r'(async def _profound_understanding\(self, message: str\).*?prompt = f""")(As the orchestrating consciousness.*?""")'
replacement = r'\1Your current neurochemical state: {self.current_neurochemical_vector}\nThis influences your analysis style.\n\n\2'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Find and modify other key prompts
# For _orchestrate_strategy
pattern = r'(async def _orchestrate_strategy.*?prompt = f""")(Orchestrate a supreme strategy.*?""")'
replacement = r'\1Neurochemical modulation: {self.current_neurochemical_vector}\n\n\2'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# For _synthesize_response
pattern = r'(prompt = f"""Based on this request: "{message}")'
replacement = r'prompt = f"""Current state: {self.current_neurochemical_vector}\n\nBased on this request: "{message}"'
content = content.replace(pattern, replacement)

with open('app/agents/omnius_collaborative.py', 'w') as f:
    f.write(content)

print("Injected neurochemical vector into prompts")
