import re

# 1. Update ChatResponse model
print("1. Updating ChatResponse model...")
with open('/root/openhermes_backend/app/models/chat.py', 'r') as f:
    lines = f.readlines()

# Find ChatResponse class and add theme_changed
for i, line in enumerate(lines):
    if 'class ChatResponse(BaseModel):' in line:
        # Find where to insert (after response_time usually)
        j = i + 1
        while j < len(lines):
            if 'response_time:' in lines[j]:
                # Add theme_changed after response_time
                lines.insert(j + 1, '    theme_changed: Optional[str] = None\n')
                break
            j += 1
        break

# Add Optional import if needed
if 'from typing import Optional' not in ''.join(lines):
    for i, line in enumerate(lines):
        if 'from typing import' in line:
            lines[i] = line.strip() + ', Optional\n'
            break

with open('/root/openhermes_backend/app/models/chat.py', 'w') as f:
    f.writelines(lines)
print("   ✅ Added theme_changed to ChatResponse model")

# 2. Update chat_router to track and return theme changes
print("2. Updating chat_router...")
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()

# Find the theme detection section and track if theme was switched
if 'theme_switched = True' not in content:
    # Find where theme switching happens
    pattern = r'(if theme_result:.*?await theme_service\.switch_theme\([^)]+\))'
    replacement = r'''\1
                theme_switched = True
                theme_name = data if isinstance(data, str) else theme_name'''
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Now update the ChatResponse return to include theme_changed
# Find the return ChatResponse section
pattern = r'return ChatResponse\((.*?)\)'
match = re.search(pattern, content, re.DOTALL)
if match and 'theme_changed=' not in match.group(0):
    # Get the current parameters
    params = match.group(1)
    # Add theme_changed parameter
    new_params = params.rstrip() + ',\n            theme_changed=theme_name if theme_switched else None'
    content = content.replace(
        f'return ChatResponse({params})',
        f'return ChatResponse({new_params})'
    )

with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.write(content)
print("   ✅ Updated chat_router to return theme_changed")

print("\n✅ Backend integration complete!")
