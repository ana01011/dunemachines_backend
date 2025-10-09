# Read chat_router.py
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    lines = f.readlines()

# Find the function definition for send_message
for i, line in enumerate(lines):
    if 'async def send_message(' in line:
        # Find where to initialize theme variables (after try:)
        j = i + 1
        while j < len(lines):
            if 'try:' in lines[j]:
                # Add theme variables initialization right after try:
                lines.insert(j + 1, '        # Initialize theme tracking variables\n')
                lines.insert(j + 2, '        theme_switched = False\n')
                lines.insert(j + 3, '        theme_name = None\n')
                lines.insert(j + 4, '\n')
                print(f"Added theme variables initialization at line {j+2}")
                break
            j += 1
        break

# Now remove the duplicate initialization that's in the wrong place
new_lines = []
skip_next = 0
for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    # Skip the wrongly placed initialization
    if 'theme_switched = False' in line and 'if not theme_action:' in lines[i-1] if i > 0 else False:
        skip_next = 1  # Skip this and next line (theme_name = None)
        continue
    
    new_lines.append(line)

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Fixed theme variable scope")
