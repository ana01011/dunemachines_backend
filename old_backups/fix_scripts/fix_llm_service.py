# Fix the context window size in llm_service.py
import fileinput
import sys

with open('app/services/llm_service.py', 'r') as file:
    content = file.read()

# Replace n_ctx=1024 with n_ctx=4096 for better context
content = content.replace('n_ctx=1024', 'n_ctx=4096')

# Write back
with open('app/services/llm_service.py', 'w') as file:
    file.write(content)

print("Fixed context window to 4096")
