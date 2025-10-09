# Fix the theme detection patterns in theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()

# Find and replace the detect_theme_command method with a better version
new_detect_method = '''
    async def detect_theme_command(self, message: str) -> Optional[Tuple[str, str, str]]:
        """Detect theme commands in message - IMPROVED"""
        msg_lower = message.lower()
        
        # First, check for explicit theme names
        for theme in self.AVAILABLE_THEMES:
            theme_lower = theme.lower()
            # Check various patterns with the theme name
            if (f"to {theme_lower}" in msg_lower or 
                f"use {theme_lower}" in msg_lower or
                f"switch {theme_lower}" in msg_lower or
                f"change {theme_lower}" in msg_lower or
                f"set {theme_lower}" in msg_lower or
                f"activate {theme_lower}" in msg_lower or
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
        
        # Check for dark/light mode requests
        if any(phrase in msg_lower for phrase in ['dark mode', 'dark theme', 'want dark', 'make it dark']):
            return ('switch_theme', 'Simple Dark', 'User requested dark mode')
        if any(phrase in msg_lower for phrase in ['light mode', 'light theme', 'want light', 'make it light', 'bright mode']):
            return ('switch_theme', 'Pure Light', 'User requested light mode')
        
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
        
        return None
'''

# Replace the old method
import re

# Find the detect_theme_command method
pattern = r'async def detect_theme_command\(self.*?\n(?:.*?\n)*?        return None'
match = re.search(pattern, content, re.DOTALL)

if match:
    content = content[:match.start()] + new_detect_method.strip() + content[match.end():]
    print("✅ Replaced detect_theme_command with improved version")
else:
    print("❌ Could not find detect_theme_command method")
    print("Adding it at the end of the class...")
    # Find the last method of the class and add after it
    lines = content.split('\n')
    for i in range(len(lines)-1, -1, -1):
        if 'return suggestions' in lines[i]:
            lines.insert(i+2, new_detect_method)
            content = '\n'.join(lines)
            break

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.write(content)

print("✅ Theme detection patterns improved!")
