import fileinput

# Fix the prompt format and context in llm_service.py
with open('app/services/llm_service.py', 'r') as file:
    content = file.read()

# Update for Mistral format
content = content.replace(
    'stop=["User:", "Human:", "\\n\\n"]',
    'stop=["</s>", "[INST]", "Human:", "User:"]'
)

# Increase context if not already done
if 'n_ctx=1024' in content:
    content = content.replace('n_ctx=1024', 'n_ctx=4096')

with open('app/services/llm_service.py', 'w') as file:
    file.write(content)

print("Updated prompt format for Mistral")
