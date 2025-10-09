# Update the theme detection in chat_router.py to handle smart suggestions
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Find and update the theme detection section
for i, line in enumerate(lines):
    if 'theme_action = await theme_service.detect_theme_command' in line:
        # Replace with enhanced detection
        enhanced_code = '''        # Check for theme commands and smart suggestions
        theme_action = await theme_service.detect_theme_intent(chat_message.message)
        if not theme_action:
            theme_action = await theme_service.detect_theme_command(chat_message.message)
        
        if theme_action:
            action_type, data, reason = theme_action
            
            if action_type == 'switch_theme':
                # Switch the theme
                success = await theme_service.switch_theme(
                    user_id=current_user.id,
                    theme_name=data,
                    trigger='chat_command'
                )
                if success:
                    user_context['theme_action'] = f"Switched to {data}"
            
            elif action_type == 'show_category':
                # Show themes in category
                user_context['theme_category'] = data
                user_context['theme_message'] = f"Here are the {reason} themes"
            
            elif action_type == 'show_all':
                # Show all available themes
                user_context['all_themes'] = data
            
            elif action_type == 'mood_suggestion':
                # Suggest themes based on mood
                user_context['theme_suggestions'] = data
                user_context['suggestion_reason'] = reason
            
            elif action_type == 'query_theme':
                # Get current theme with description
                current_theme = await theme_service.get_current_theme(current_user.id)
                description = theme_service.THEME_DESCRIPTIONS.get(current_theme, '')
                user_context['theme_query'] = f"{current_theme}: {description}"
            
            elif action_type == 'suggest_theme':
                # Get smart suggestions
                suggestions = await theme_service.get_theme_suggestions(
                    user_id=current_user.id,
                    context={'time': datetime.now().hour}
                )
                user_context['theme_suggestions'] = suggestions
'''
        # Replace the entire theme detection block
        # Find the end of the current block
        end_idx = i
        for j in range(i, min(i+30, len(lines))):
            if 'user_context[\'theme_suggestions\']' in lines[j]:
                end_idx = j + 1
                break
        
        # Replace the block
        lines[i:end_idx] = [enhanced_code + '\n']
        break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Updated chat router with smart theme handling")
