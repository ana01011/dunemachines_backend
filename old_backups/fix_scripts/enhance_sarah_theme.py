# Add theme-specific personality traits to Sarah
with open('/root/openhermes_backend/app/agents/personalities/sarah.py', 'r') as f:
    content = f.read()

# Find where to add theme personality
if 'theme personality' not in content.lower():
    # Add theme personality traits after the system prompt
    theme_personality = '''
        # Theme preferences
        if user_context and 'theme_action' in user_context:
            base_prompt += f"\\n\\nThe user just {user_context['theme_action']}. Acknowledge this naturally."
        
        if user_context and 'current_theme' in user_context:
            theme = user_context.get('current_theme', 'Cyber Dark')
            if theme == 'Neon Nights':
                base_prompt += "\\nYou love the Neon Nights theme - it's vibrant and energetic!"
            elif theme == 'Cyber Dark':
                base_prompt += "\\nThe Cyber Dark theme is sleek and modern - perfect for coding!"
            elif theme == 'Pure Light':
                base_prompt += "\\nThe Pure Light theme is clean and minimalist - easy on the eyes."
'''
    
    # Insert before the return statement in get_system_prompt
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'return base_prompt' in line and 'get_system_prompt' in ''.join(lines[i-20:i]):
            lines.insert(i, theme_personality)
            break
    
    content = '\n'.join(lines)
    
    with open('/root/openhermes_backend/app/agents/personalities/sarah.py', 'w') as f:
        f.write(content)
    
    print("✅ Enhanced Sarah's theme personality")
else:
    print("Theme personality already added")
