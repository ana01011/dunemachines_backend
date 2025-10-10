with open('app/agents/omnius_collaborative.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for i, line in enumerate(lines):
    # Skip the problematic lines
    if 'original_message = message' in line:
        continue  # Skip this line
    elif 'message = enhanced_message' in line:
        continue  # Skip this line
    elif 'neuro_state = get_user_neurochemical_state(user_id)' in line:
        # Check if this is a duplicate (appears twice)
        if i > 0 and 'self.current_neurochemical_vector = neurochemical_vector' in lines[i-1]:
            continue  # Skip duplicate
        else:
            new_lines.append(line)
    elif "print(f\"\\n🧬 Neurochemical State:" in line:
        # Check if this is a duplicate print
        if i > 0 and "print(f\"\\n🧬 Neurochemical State:" in lines[i-2]:
            continue  # Skip duplicate
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('app/agents/omnius_collaborative.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed: Removed message replacement and duplicates")
