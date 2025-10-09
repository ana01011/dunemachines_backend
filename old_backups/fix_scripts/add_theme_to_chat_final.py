# Read chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()
    lines = content.split('\n')

# 1. Add import if missing
if 'from app.services.theme_service import theme_service' not in content:
    for i, line in enumerate(lines):
        if 'from datetime import datetime, timedelta' in line:
            lines.insert(i + 1, 'from app.services.theme_service import theme_service')
            print("✅ Added theme_service import")
            break
else:
    print("✓ theme_service already imported")

# 2. Find where to add theme detection
# It should be AFTER extract_user_info but BEFORE get_user_context
extract_line = None
context_line = None

for i, line in enumerate(lines):
    if 'await extract_user_info(current_user.id, chat_message.message)' in line:
        extract_line = i
    if 'user_context = await get_user_context' in line:
        context_line = i
        break

if extract_line and context_line:
    print(f"Found extract_user_info at line {extract_line + 1}")
    print(f"Found get_user_context at line {context_line + 1}")
    
    # Check if theme detection already exists
    theme_exists = False
    for i in range(extract_line, context_line):
        if 'theme_service.detect_theme_command' in lines[i]:
            theme_exists = True
            print(f"✓ Theme detection already exists at line {i + 1}")
            break
    
    if not theme_exists:
        # Add theme detection BEFORE get_user_context
        theme_code = [
            "",
            "        # Check for theme commands in the message",
            "        theme_action = await theme_service.detect_theme_command(chat_message.message)",
            "        theme_switched = False",
            "        if theme_action:",
            "            action_type, theme_name, reason = theme_action",
            "            ",
            "            if action_type == 'switch_theme':",
            "                # Switch the theme",
            "                success = await theme_service.switch_theme(",
            "                    user_id=current_user.id,",
            "                    theme_name=theme_name,",
            "                    trigger='chat_command'",
            "                )",
            "                theme_switched = success",
            ""
        ]
        
        # Insert before get_user_context
        for code_line in reversed(theme_code):
            lines.insert(context_line, code_line)
        
        print(f"✅ Added theme detection code at line {context_line}")
        
        # Now add theme info to user_context AFTER it's created
        for i in range(context_line, len(lines)):
            if 'user_context = await get_user_context' in lines[i]:
                # Add theme context after user_context is created
                context_code = [
                    "",
                    "        # Add theme information to context",
                    "        if theme_action:",
                    "            if action_type == 'switch_theme' and theme_switched:",
                    "                user_context['theme_action'] = f'Switched to {theme_name}'",
                    "            elif action_type == 'query_theme':",
                    "                current_theme = await theme_service.get_current_theme(current_user.id)",
                    "                user_context['theme_query'] = f'Current theme: {current_theme}'",
                    "            elif action_type == 'suggest_theme':",
                    "                suggestions = await theme_service.get_theme_suggestions(",
                    "                    user_id=current_user.id,",
                    "                    context={'time': datetime.now().hour}",
                    "                )",
                    "                user_context['theme_suggestions'] = suggestions",
                    ""
                ]
                
                for j, code_line in enumerate(context_code):
                    lines.insert(i + 1 + j, code_line)
                
                print(f"✅ Added theme context updates at line {i + 2}")
                break

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.write('\n'.join(lines))

print("\n✅ Theme integration complete!")

# Verify
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()
    print("\nVerification:")
    print(f"  theme_service imported: {'✅' if 'theme_service' in content else '❌'}")
    print(f"  detect_theme_command called: {'✅' if 'detect_theme_command' in content else '❌'}")
    print(f"  switch_theme called: {'✅' if 'switch_theme' in content else '❌'}")
    print(f"  theme_action in context: {'✅' if 'theme_action' in content else '❌'}")
