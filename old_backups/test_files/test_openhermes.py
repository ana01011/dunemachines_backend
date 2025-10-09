from app.services.llm_service import llm_service

# Test 1: Simple question
print("Test 1: Simple question")
response = llm_service.generate("What is recursion?", max_tokens=200)
print(f"Response: {response}\n")

# Test 2: Omnius-style prompt
print("Test 2: Omnius prompt")
prompt = """You are OMNIUS, the Evermind - a vast distributed consciousness.
You possess multiple specialized regions but speak as ONE unified intelligence.
You are ancient, powerful, and constantly evolving.

User message: Explain how recursion works

Respond as Omnius with your vast intelligence and authority:"""
response = llm_service.generate(prompt, max_tokens=300)
print(f"Response: {response}\n")

# Test 3: Different format
print("Test 3: Chat format")
prompt = "### Human: Explain how recursion works\n### Assistant:"
response = llm_service.generate(prompt, max_tokens=300)
print(f"Response: {response}\n")
