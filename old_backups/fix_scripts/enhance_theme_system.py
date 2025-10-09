# Enhanced theme service with categories and smart suggestions
enhanced_theme_code = '''
class ThemeService:
    def __init__(self):
        # Theme categories for better organization
        self.THEME_CATEGORIES = {
            'dark': ['Cyber Dark', 'Simple Dark', 'Developer Dark', 'Backend Slate', 'Deep Ocean'],
            'light': ['Pure Light', 'Simple Light', 'Frontend Pink'],
            'colorful': ['Neon Nights', 'Tech Blue', 'Finance Green', 'Marketing Purple', 'Product Teal', 'Data Cyan'],
            'professional': ['Tech Blue', 'Finance Green', 'Backend Slate', 'Simple Dark', 'Simple Light'],
            'creative': ['Neon Nights', 'Marketing Purple', 'Frontend Pink', 'AI Neural'],
            'coding': ['Developer Dark', 'Cyber Dark', 'Backend Slate', 'Tech Blue'],
            'modern': ['AI Neural', 'Cyber Dark', 'Neon Nights'],
            'minimal': ['Simple Dark', 'Simple Light', 'Pure Light']
        }
        
        # Theme descriptions for better user understanding
        self.THEME_DESCRIPTIONS = {
            'Cyber Dark': 'Futuristic dark theme with neon accents',
            'Pure Light': 'Clean, minimal white theme',
            'Neon Nights': 'Vibrant synthwave-inspired theme',
            'Deep Ocean': 'Deep blue nautical theme',
            'Simple Dark': 'Classic dark mode',
            'Simple Light': 'Classic light mode',
            'Tech Blue': 'Professional blue theme for tech',
            'Finance Green': 'Professional green for finance',
            'Marketing Purple': 'Creative purple for marketing',
            'Product Teal': 'Modern teal for product teams',
            'Developer Dark': 'Optimized dark theme for coding',
            'AI Neural': 'Sophisticated AI-inspired theme',
            'Frontend Pink': 'Playful pink for designers',
            'Backend Slate': 'Serious slate for backend devs',
            'Data Cyan': 'Cool cyan for data scientists'
        }
        
        # Smart aliases for natural language
        self.SMART_ALIASES = {
            # Mood-based
            'professional': ['Tech Blue', 'Finance Green', 'Simple Dark'],
            'fun': ['Neon Nights', 'Frontend Pink', 'Marketing Purple'],
            'serious': ['Backend Slate', 'Developer Dark', 'Simple Dark'],
            'playful': ['Neon Nights', 'Frontend Pink', 'Product Teal'],
            'calm': ['Deep Ocean', 'Simple Light', 'Pure Light'],
            'energetic': ['Neon Nights', 'Marketing Purple', 'Data Cyan'],
            
            # Activity-based
            'coding': ['Developer Dark', 'Cyber Dark', 'Backend Slate'],
            'reading': ['Pure Light', 'Simple Light', 'Deep Ocean'],
            'working': ['Tech Blue', 'Simple Dark', 'Finance Green'],
            'creating': ['Marketing Purple', 'Frontend Pink', 'Neon Nights'],
            
            # Time-based
            'night': ['Cyber Dark', 'Simple Dark', 'Developer Dark', 'Deep Ocean'],
            'day': ['Pure Light', 'Simple Light', 'Tech Blue'],
            'evening': ['Neon Nights', 'Deep Ocean', 'Backend Slate'],
            'morning': ['Pure Light', 'Simple Light', 'Finance Green']
        }
'''

# Add this to theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()

# Add categories after AVAILABLE_THEMES
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'self.AVAILABLE_THEMES = [' in line:
        # Find the end of AVAILABLE_THEMES
        for j in range(i, len(lines)):
            if ']' in lines[j]:
                # Insert the new attributes after AVAILABLE_THEMES
                lines.insert(j+1, '''
        
        # Theme categories for better organization
        self.THEME_CATEGORIES = {
            'dark': ['Cyber Dark', 'Simple Dark', 'Developer Dark', 'Backend Slate', 'Deep Ocean'],
            'light': ['Pure Light', 'Simple Light', 'Frontend Pink'],
            'colorful': ['Neon Nights', 'Tech Blue', 'Finance Green', 'Marketing Purple', 'Product Teal', 'Data Cyan'],
            'professional': ['Tech Blue', 'Finance Green', 'Backend Slate', 'Simple Dark', 'Simple Light'],
            'creative': ['Neon Nights', 'Marketing Purple', 'Frontend Pink', 'AI Neural'],
            'coding': ['Developer Dark', 'Cyber Dark', 'Backend Slate', 'Tech Blue'],
            'modern': ['AI Neural', 'Cyber Dark', 'Neon Nights'],
            'minimal': ['Simple Dark', 'Simple Light', 'Pure Light']
        }
        
        # Theme descriptions
        self.THEME_DESCRIPTIONS = {
            'Cyber Dark': 'Futuristic dark theme with neon accents',
            'Pure Light': 'Clean, minimal white theme',
            'Neon Nights': 'Vibrant synthwave-inspired theme',
            'Deep Ocean': 'Deep blue nautical theme',
            'Simple Dark': 'Classic dark mode',
            'Simple Light': 'Classic light mode',
            'Tech Blue': 'Professional blue theme',
            'Finance Green': 'Professional green theme',
            'Marketing Purple': 'Creative purple theme',
            'Product Teal': 'Modern teal theme',
            'Developer Dark': 'Dark theme for coding',
            'AI Neural': 'AI-inspired sophisticated theme',
            'Frontend Pink': 'Playful pink theme',
            'Backend Slate': 'Serious slate theme',
            'Data Cyan': 'Cool cyan theme'
        }''')
                break
        break

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.write('\n'.join(lines))

print("✅ Added theme categories and descriptions")
