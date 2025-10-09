# Read chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Add debug print statements
for i, line in enumerate(lines):
    # Find where theme detection happens
    if 'theme_action = await theme_service.detect_theme_intent' in line:
        # Add debug print after
        lines.insert(i+1, '        print(f"DEBUG: detect_theme_intent result: {theme_action}")\n')
        
    if 'theme_action = theme_service.detect_theme_command' in line:
        # Add debug print after
        lines.insert(i+1, '        print(f"DEBUG: detect_theme_command result: {theme_action}")\n')
        
    if "if action_type == 'switch_theme':" in line:
        # Add debug print
        lines.insert(i+1, '                print(f"DEBUG: Switching to theme: {data}")\n')
        
    if 'theme_switched = True' in line:
        lines.insert(i+1, '                    print(f"DEBUG: Theme switched successfully to {theme_name}")\n')

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Added debug logging")
