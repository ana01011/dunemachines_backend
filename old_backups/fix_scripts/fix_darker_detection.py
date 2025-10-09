# Read theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()

# Find the detect_theme_command method and replace it with improved version
import re

new_detect_method = '''    def detect_theme_command(self, message: str) -> Optional[tuple]:
        """Detect theme-related commands in user message"""
        msg_lower = message.lower()
        
        # Simple detection for dark/light requests
        if 'dark' in msg_lower:
            # Check for dark theme requests in various forms
            if any(word in msg_lower for word in ['background', 'theme', 'mode', 'color', 'switch', 'change', 'make', 'want', 'use', 'set']):
                return ('switch_theme', 'Simple Dark', 'User requested dark theme')
        
        if 'light' in msg_lower or 'bright' in msg_lower:
            # Check for light theme requests
            if any(word in msg_lower for word in ['background', 'theme', 'mode', 'color', 'switch', 'change', 'make', 'want', 'use', 'set']):
                return ('switch_theme', 'Pure Light', 'User requested light theme')
        
        # Direct theme name detection
        for theme in self.AVAILABLE_THEMES:
            theme_lower = theme.lower()
            if (theme_lower in msg_lower or 
                f"{theme_lower} theme" in msg_lower or
                f"{theme_lower} mode" in msg_lower):
                return ('switch_theme', theme, f'User requested {theme}')

        # Check for partial matches (e.g., "Pure Light" when user says "pure light")
        words = msg_lower.split()
        for theme in self.AVAILABLE_THEMES:
            theme_words = theme.lower().split()
            # Check if all words of theme name are in message
            if all(word in words for word in theme_words):
                return ('switch_theme', theme, f'User requested {theme}')

        # Check for theme query
        if any(phrase in msg_lower for phrase in ['what theme', 'which theme', 'current theme', 'my theme', 'theme am i']):
            return ('query_theme', None, 'User asked about current theme')

        # Check for theme suggestions
        if any(phrase in msg_lower for phrase in ['suggest theme', 'recommend theme', 'best theme', 'good theme']):
            return ('suggest_theme', None, 'User wants theme suggestions')

        # Check aliases
        for alias, theme_name in self.THEME_ALIASES.items():
            if alias in msg_lower:
                return ('switch_theme', theme_name, f'User requested {theme_name} via alias')

        return None'''

# Replace the old method
pattern = r'def detect_theme_command\(self.*?\n(?:.*?\n)*?        return None'
content = re.sub(pattern, new_detect_method, content, flags=re.DOTALL)

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed theme detection for 'darker background' and similar phrases")
