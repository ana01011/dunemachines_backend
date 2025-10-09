# Read the chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Find and remove the incorrectly placed theme detection code (around line 330-350)
new_lines = []
skip_until = -1

for i, line in enumerate(lines):
    if skip_until > i:
        continue
    
    # Skip the incorrectly placed theme detection block
    if '# Check for theme commands in the message' in line and 'user_context[' in ''.join(lines[i:i+20]):
        # Find the end of this block
        for j in range(i, min(i+30, len(lines))):
            if 'user_context[\'theme_suggestions\']' in lines[j]:
                skip_until = j + 2  # Skip past this block
                break
        continue
    
    new_lines.append(line)

# Now add the theme detection in the RIGHT place - AFTER user_context is created
final_lines = []
for i, line in enumerate(new_lines):
    final_lines.append(line)
    
    # Add theme detection AFTER user_context is created
    if 'user_context = await get_user_context' in line:
        theme_code = '''
        # Check for theme commands and update context
        theme_action = await theme_service.detect_theme_command(chat_message.message)
        if theme_action:
            action_type, theme_name, reason = theme_action
            
            if action_type == 'switch_theme':
                # Switch the theme
                success = await theme_service.switch_theme(
                    user_id=current_user.id,
                    theme_name=theme_name,
                    trigger='chat_command'
                )
                if success:
                    user_context['theme_action'] = f"Switched to {theme_name}"
            
            elif action_type == 'query_theme':
                # Get current theme
                current_theme = await theme_service.get_current_theme(current_user.id)
                user_context['theme_query'] = f"Current theme: {current_theme}"
            
            elif action_type == 'suggest_theme':
                # Get theme suggestions
                suggestions = await theme_service.get_theme_suggestions(
                    user_id=current_user.id,
                    context={'time': datetime.now().hour}
                )
                user_context['theme_suggestions'] = suggestions
'''
        final_lines.append(theme_code)

# Write the fixed file
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(final_lines)

print("✅ Fixed theme detection placement in chat_router.py")
print("Theme detection is now AFTER user_context is created")
