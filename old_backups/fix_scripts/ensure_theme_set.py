# Read chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Find where theme switching happens and ensure we set the variables
for i, line in enumerate(lines):
    if "action_type == 'switch_theme':" in line:
        # Look for the success check
        j = i + 1
        while j < min(i + 15, len(lines)):
            if 'if success:' in lines[j]:
                # Add theme_switched = True and theme_name = data
                if 'theme_switched = True' not in lines[j+1]:
                    lines.insert(j + 1, '                    theme_switched = True\n')
                    lines.insert(j + 2, '                    theme_name = data\n')
                    print(f"Added theme tracking at line {j+2}")
                break
            j += 1
        break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Ensured theme variables are set on switch")
