# Read sarah.py
with open('/root/openhermes_backend/app/agents/personalities/sarah.py', 'r') as f:
    lines = f.readlines()

# Find the build_prompt method (around line 74)
for i, line in enumerate(lines):
    if 'def build_prompt(self, message: str' in line:
        # Find the return statement
        for j in range(i, len(lines)):
            if 'return f"{system_prompt}' in lines[j]:
                # Add theme context before the return
                theme_context = '''        
        # Add theme context if available
        if user_context:
            if 'theme_action' in user_context:
                system_prompt += f"\\n[The user just switched themes: {user_context['theme_action']}. Acknowledge this naturally.]"
            if 'theme_query' in user_context:
                system_prompt += f"\\n[{user_context['theme_query']}. Tell them about their current theme.]"
            if 'theme_suggestions' in user_context:
                system_prompt += f"\\n[You can suggest these themes if appropriate: {user_context['theme_suggestions']}]"
        
'''
                lines.insert(j, theme_context)
                print(f"✅ Added theme context at line {j}")
                break
        break

# Write back
with open('/root/openhermes_backend/app/agents/personalities/sarah.py', 'w') as f:
    f.writelines(lines)

print("✅ Sarah personality updated with theme awareness")
