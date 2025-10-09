from app.services.llm_service import llm_service

print("Testing OpenHermes with code generation:")
print("=" * 50)

# Test with simple code request
prompt = "Write a Python function to calculate factorial"
response = llm_service.generate(prompt, max_tokens=500)
print(f"Simple code prompt response length: {len(response)}")
print(f"Response:\n{response}\n")

# Test with chat format
prompt2 = "### Human: Write a Python function to calculate factorial\n### Assistant:"
response2 = llm_service.generate(prompt2, max_tokens=500)
print(f"Chat format response length: {len(response2)}")
print(f"Response:\n{response2}\n")
