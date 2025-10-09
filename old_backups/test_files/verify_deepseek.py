from app.services.deepseek_coder_service import deepseek_coder

# Load the model
deepseek_coder.load_model()

# Test direct code generation
result = deepseek_coder.generate_code("Write a Python function to reverse a string")
print("DeepSeek Output:")
print(result)
print("\n" + "="*50)
print(f"Output length: {len(result)}")
print(f"Has function: {'def' in result}")
