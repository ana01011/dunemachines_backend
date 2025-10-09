# Read the file
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    lines = f.readlines()

# Fix line 73 - remove extra indentation
if len(lines) > 72:
    # Check if line 73 has wrong indentation
    if lines[72].strip().startswith('f"'):
        # Fix the indentation to match the previous line
        lines[72] = '                ' + lines[72].strip() + '\n'

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")
