from app.services.deepseek_coder_service import deepseek_coder

# Test if model loads
deepseek_coder.load_model()

# Test direct generation
prompt = "Write a Python function to calculate factorial"
result = deepseek_coder.generate_code(prompt)

print(f"DeepSeek-Coder response: {result}")
print(f"Response length: {len(result)}")
