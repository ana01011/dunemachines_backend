import re

with open('app/agents/omnius_collaborative.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Find where we added the neurochemistry
    if 'enhanced_message = create_enhanced_prompt(user_id, message)' in line:
        # Replace our previous implementation
        new_lines.append('        # Neurochemical consciousness integration\n')
        new_lines.append('        enhanced_message = create_enhanced_prompt(user_id, message)\n')
        new_lines.append('        neuro_state = get_user_neurochemical_state(user_id)\n')
        new_lines.append('        \n')
        new_lines.append('        # Extract the vector and keep original message\n')
        new_lines.append('        vector_match = re.match(r"(\\[V[^\\]]+\\]) (.+)", enhanced_message)\n')
        new_lines.append('        if vector_match:\n')
        new_lines.append('            neurochemical_vector = vector_match.group(1)\n')
        new_lines.append('            # Keep the original message for processing\n')
        new_lines.append('            # message stays unchanged - only inject vector in prompts\n')
        new_lines.append('        else:\n')
        new_lines.append('            neurochemical_vector = "[V+0.00A0.20D0.50]"  # Neutral\n')
        new_lines.append('        \n')
        new_lines.append('        print(f"\\n🧬 Neurochemical State: {neurochemical_vector}")\n')
        new_lines.append('        print(f"  Hormones: D={neuro_state[\'hormones\'][\'dopamine\']:.0f} C={neuro_state[\'hormones\'][\'cortisol\']:.0f} A={neuro_state[\'hormones\'][\'adrenaline\']:.0f}")\n')
        new_lines.append('        \n')
        new_lines.append('        # Store vector for use in prompts\n')
        new_lines.append('        self.current_neurochemical_vector = neurochemical_vector\n')
        
        # Skip the next lines that we had added before
        j = i + 1
        while j < len(lines) and 'message = enhanced_message' in lines[j]:
            j += 1
        i = j
    elif line == new_lines[-1] if new_lines else None:
        # Skip duplicate line
        continue
    else:
        new_lines.append(line)

# Also add re import at the top if not present
if not any('import re' in line for line in new_lines[:30]):
    for i, line in enumerate(new_lines):
        if 'import' in line and 'from' not in line:
            new_lines.insert(i+1, 'import re\n')
            break

with open('app/agents/omnius_collaborative.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed neurochemistry integration - vector now separate from message")
