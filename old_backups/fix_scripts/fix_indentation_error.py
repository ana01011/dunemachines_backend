# Read theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    lines = f.readlines()

# Fix line 72 indentation - should have 4 spaces (one indent level)
for i, line in enumerate(lines):
    if 'def detect_theme_command' in line:
        # Ensure proper indentation (4 spaces for class method)
        lines[i] = '    def detect_theme_command(self, message: str) -> Optional[tuple]:\n'
        print(f"Fixed indentation at line {i+1}")
        break

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed indentation error")
