# Add smart suggestion methods to theme_service.py
with open('/root/openhermes_backend/app/services/theme_service.py', 'r') as f:
    content = f.read()

# Add new methods before the singleton creation
smart_methods = '''
    async def get_themes_by_category(self, category: str) -> List[Dict[str, str]]:
        """Get themes in a specific category with descriptions"""
        themes = self.THEME_CATEGORIES.get(category, [])
        return [
            {
                'name': theme,
                'description': self.THEME_DESCRIPTIONS.get(theme, '')
            }
            for theme in themes
        ]
    
    async def suggest_themes_for_mood(self, mood: str) -> List[str]:
        """Suggest themes based on user's mood or activity"""
        mood_lower = mood.lower()
        
        # Check direct mood mappings
        if mood_lower in self.SMART_ALIASES:
            return self.SMART_ALIASES[mood_lower][:3]
        
        # Check categories
        if mood_lower in self.THEME_CATEGORIES:
            return self.THEME_CATEGORIES[mood_lower][:3]
        
        # Fuzzy matching for moods
        suggestions = []
        if 'dark' in mood_lower or 'night' in mood_lower:
            suggestions = self.THEME_CATEGORIES['dark'][:3]
        elif 'light' in mood_lower or 'bright' in mood_lower:
            suggestions = self.THEME_CATEGORIES['light'][:3]
        elif 'color' in mood_lower or 'fun' in mood_lower:
            suggestions = self.THEME_CATEGORIES['colorful'][:3]
        elif 'work' in mood_lower or 'professional' in mood_lower:
            suggestions = self.THEME_CATEGORIES['professional'][:3]
        elif 'code' in mood_lower or 'coding' in mood_lower:
            suggestions = self.THEME_CATEGORIES['coding'][:3]
        else:
            # Default suggestions
            suggestions = ['Cyber Dark', 'Pure Light', 'Neon Nights']
        
        return suggestions
    
    async def detect_theme_intent(self, message: str) -> Optional[Tuple[str, Any, str]]:
        """Enhanced theme detection with categories and moods"""
        msg_lower = message.lower()
        
        # Check for category requests
        if 'show' in msg_lower or 'list' in msg_lower or 'what themes' in msg_lower:
            for category in self.THEME_CATEGORIES.keys():
                if category in msg_lower:
                    themes = await self.get_themes_by_category(category)
                    return ('show_category', themes, f'Showing {category} themes')
            
            # Show all themes if no specific category
            if 'all themes' in msg_lower or 'available themes' in msg_lower:
                return ('show_all', self.AVAILABLE_THEMES, 'Showing all themes')
        
        # Check for mood-based requests
        if any(word in msg_lower for word in ['something', 'theme for', 'good for', 'want something']):
            # Extract the mood/activity
            for mood in ['professional', 'fun', 'coding', 'reading', 'working', 'night', 'day']:
                if mood in msg_lower:
                    suggestions = await self.suggest_themes_for_mood(mood)
                    return ('mood_suggestion', suggestions, f'Themes for {mood}')
        
        # Original detection continues...
        return await self.detect_theme_command(message)
'''

# Insert before "# Create singleton instance"
lines = content.split('\n')
for i, line in enumerate(lines):
    if '# Create singleton instance' in line or 'theme_service = ThemeService()' in line:
        lines.insert(i-1, smart_methods)
        break

# Write back
with open('/root/openhermes_backend/app/services/theme_service.py', 'w') as f:
    f.write('\n'.join(lines))

print("✅ Added smart theme suggestion methods")
