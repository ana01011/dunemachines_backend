# Read theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()

# Fix the async method definitions (they should be instance methods, not class methods)
content = content.replace('@classmethod\n    async def', 'async def')
content = content.replace('cls.', 'self.')

# Make sure theme_service is instantiated
if 'theme_service = ThemeService()' not in content:
    content += '\n\n# Singleton instance\ntheme_service = ThemeService()\n'

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed theme service methods")
