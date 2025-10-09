# Read the models file
with open('/root/openhermes_backend/app/models/chat.py', 'r') as f:
    lines = f.readlines()

# Check if theme_changed exists
has_theme_changed = any('theme_changed' in line for line in lines)

if not has_theme_changed:
    # Find ChatResponse class
    for i, line in enumerate(lines):
        if 'class ChatResponse(BaseModel):' in line:
            # Find a good place to add the field (after response field)
            j = i + 1
            while j < len(lines):
                if 'response:' in lines[j]:
                    # Add theme_changed after response
                    lines.insert(j + 1, '    theme_changed: Optional[str] = None\n')
                    print(f"Added theme_changed field at line {j+2}")
                    break
                j += 1
            break
    
    # Ensure Optional is imported
    for i, line in enumerate(lines):
        if 'from typing import' in line:
            if 'Optional' not in line:
                lines[i] = line.rstrip().rstrip(',') + ', Optional\n'
            break
    
    # Write back
    with open('/root/openhermes_backend/app/models/chat.py', 'w') as f:
        f.writelines(lines)
    
    print("✅ Added theme_changed to ChatResponse model")
else:
    print("ℹ️ theme_changed already in model")
