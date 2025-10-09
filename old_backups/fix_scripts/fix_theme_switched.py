# Read the file
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Find where we need to define theme_switched
theme_switched_defined = False
for line in lines:
    if 'theme_switched = ' in line:
        theme_switched_defined = True
        break

if not theme_switched_defined:
    # Find where to add it (before theme detection)
    for i, line in enumerate(lines):
        if 'Theme detection and switching' in line or 'detect_theme_command' in line:
            # Add theme_switched = False before theme detection
            lines.insert(i, '        theme_switched = False\n')
            lines.insert(i+1, '        theme_name = None\n')
            print("Added theme_switched and theme_name initialization")
            break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed theme_switched variable")
