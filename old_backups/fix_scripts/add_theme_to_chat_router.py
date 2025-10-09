# Read chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Add import at the top (after line 17)
import_added = False
for i, line in enumerate(lines):
    if 'from datetime import datetime, timedelta' in line:
        lines.insert(i + 1, 'from app.services.theme_service import theme_service\n')
        import_added = True
        print("✅ Added theme_service import")
        break

# Find where to add theme detection (after extract_user_info, around line 329)
for i, line in enumerate(lines):
    if 'await extract_user_info(current_user.id, chat_message.message)' in line:
        # Add theme detection code after this line
        theme_code = '''
        # Check for theme commands in the message
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
        # Insert after the extract_user_info line
        lines.insert(i + 1, theme_code)
        print(f"✅ Added theme detection at line {i+2}")
        break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(lines)

print("✅ Theme detection added to chat router")
