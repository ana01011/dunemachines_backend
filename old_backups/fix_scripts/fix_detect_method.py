# Read the enhanced file
with open('/root/openhermes_backend/theme_service_enhanced.py', 'r') as f:
    content = f.read()

# Find and replace the detect_theme_command method
new_method = '''    async def detect_theme_command(self, message: str) -> Optional[tuple]:
        """Enhanced theme detection with categories and moods"""
        msg_lower = message.lower()
        
        # Check for "show" or "list" commands
        if any(word in msg_lower for word in ['show', 'list', 'what', 'available']):
            # Check for category requests
            for category in self.THEME_CATEGORIES.keys():
                if category in msg_lower:
                    themes = self.THEME_CATEGORIES[category]
                    theme_list = [{'theme': t, 'description': self.THEME_DESCRIPTIONS.get(t, '')} for t in themes]
                    return ('show_category', theme_list, category)
            
            # Show all themes if asking about themes in general
            if 'theme' in msg_lower:
                all_themes = list(self.AVAILABLE_THEMES)
                return ('show_all', all_themes, 'all available themes')
        
        # Check for current theme query
        if any(phrase in msg_lower for phrase in ['current theme', 'what theme', 'which theme', 'my theme']):
            return ('query_theme', None, 'User asked about current theme')
        
        # Check for theme suggestions
        if any(phrase in msg_lower for phrase in ['suggest theme', 'recommend theme', 'best theme', 'good theme']):
            return ('suggest_theme', None, 'User wants theme suggestions')
        
        # Direct theme switching
        for theme in self.AVAILABLE_THEMES:
            if theme.lower() in msg_lower:
                return ('switch_theme', theme, f'User requested {theme}')
        
        # Check aliases
        for alias, theme_name in self.THEME_ALIASES.items():
            if alias in msg_lower:
                return ('switch_theme', theme_name, f'User requested {theme_name} via alias')
        
        return None'''

# Find the old method and replace
import re
pattern = r'async def detect_theme_command\(self.*?\n(?:.*?\n)*?        return None'
content = re.sub(pattern, new_method, content, count=1)

with open('/root/openhermes_backend/theme_service_enhanced.py', 'w') as f:
    f.write(content)

print("Fixed detect_theme_command method")
