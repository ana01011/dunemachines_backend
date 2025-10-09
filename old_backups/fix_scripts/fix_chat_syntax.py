# Read the file
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Fix line 505 - looks like it's missing closing parenthesis
if len(lines) > 504:
    # Check if line 505 has the broken syntax
    if 'tokens_used=len(response_text.split(,' in lines[504]:
        # Fix it - should be split()
        lines[504] = '            tokens_used=len(response_text.split()),\n'
    elif 'tokens_used=len(response_text.split(' in lines[504]:
        # Alternative fix
        lines[504] = '            tokens_used=len(response_text.split()),\n'

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed syntax error")
