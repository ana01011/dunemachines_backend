# Backup the current broken version
import shutil
shutil.copy('/root/openhermes_backend/app/services/theme_service.py', 
            '/root/openhermes_backend/app/services/theme_service_broken.py')

# Read the file
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    lines = f.readlines()

# Fix the file line by line
fixed_lines = []
in_class = False
class_indent = 0

for i, line in enumerate(lines):
    # Detect class definition
    if 'class ThemeService' in line:
        in_class = True
        class_indent = len(line) - len(line.lstrip())
        fixed_lines.append(line)
        continue
    
    # Fix @classmethod decorators - remove them
    if '@classmethod' in line and in_class:
        continue  # Skip the decorator
    
    # Fix method definitions
    if in_class and 'async def ' in line:
        # Check if it's a method (indented under class)
        current_indent = len(line) - len(line.lstrip())
        if current_indent > class_indent:
            # Fix the method signature
            if '(cls,' in line:
                line = line.replace('(cls,', '(self,')
            elif '(cls)' in line:
                line = line.replace('(cls)', '(self)')
            # Make sure it has self if it doesn't
            elif 'def ' in line and '(self' not in line:
                if '()' in line:
                    line = line.replace('()', '(self)')
                elif '(' in line:
                    line = line.replace('(', '(self, ', 1)
    
    # Fix cls references to self
    if in_class and 'cls.' in line:
        line = line.replace('cls.', 'self.')
    
    fixed_lines.append(line)

# Write the fixed file
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ Fixed theme_service.py")

# Verify the fixes
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()
    
# Check specific problem lines
print("\nChecking fixes:")
print("- 'self' references:", content.count('self.'))
print("- 'cls' references:", content.count('cls.'))
print("- '@classmethod' decorators:", content.count('@classmethod'))

# Make sure singleton exists
if 'theme_service = ThemeService()' not in content:
    with open('/root/openhermes_backend/app/services/theme_service.py', 'a') as f:
        f.write('\n\n# Create singleton instance\ntheme_service = ThemeService()\n')
    print("- Added singleton instance")
