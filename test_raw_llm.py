import sys
sys.path.append('/root/openhermes_backend')
from app.services.llm_service import llm_service

# Test how the LLM interprets the neurochemical vector
prompt1 = """Your current neurochemical state: [V-0.22A0.48D0.38]
This influences your analysis style.

As the orchestrating consciousness, profoundly analyze:
"I'm really frustrated! This error keeps happening!"

Determine:
1. True intent behind the request
2. Complexity level
3. Success criteria"""

print("Testing LLM understanding of neurochemical prompt...")
print("-" * 60)

# LLM service doesn't need load()
response = llm_service.generate(prompt1, temperature=0.7, max_tokens=500)
print("Response:", response)
print("-" * 60)

# Now test with better context
prompt2 = """Your emotional state: [V-0.22A0.48D0.38]
This means: Negative mood (frustrated), moderate arousal (urgent), low dominance (helpless feeling).
You should respond with empathy for frustration and urgency to solve the problem.

Analyze this frustrated user's request:
"I'm really frustrated! This error keeps happening!"

Determine:
1. True intent
2. Complexity 
3. Success criteria"""

print("\nTesting with better neurochemical context...")
print("-" * 60)
response2 = llm_service.generate(prompt2, temperature=0.7, max_tokens=500)
print("Response:", response2)
