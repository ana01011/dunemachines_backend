# Read the file
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Fix line 506 - remove extra ))
for i, line in enumerate(lines):
    if 'theme_changed=theme_name if theme_switched else None)),' in line:
        lines[i] = line.replace('))),', '),')
        print(f"Fixed line {i+1}: removed extra parentheses")
        break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed extra parentheses")
