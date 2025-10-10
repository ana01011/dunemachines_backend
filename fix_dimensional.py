# Fix the dimensional_emergence.py file
with open('app/neurochemistry/core/dimensional_emergence.py', 'r') as f:
    lines = f.readlines()

# Find and replace the create_prompt_injection method
new_lines = []
in_create_prompt = False
skip_lines = 0

for i, line in enumerate(lines):
    if skip_lines > 0:
        skip_lines -= 1
        continue
        
    if 'def create_prompt_injection' in line:
        # Replace the entire method
        new_lines.append('    @staticmethod\n')
        new_lines.append('    def create_prompt_injection(pos: DimensionalPosition, state=None) -> str:\n')
        new_lines.append('        """Create the prompt injection string - ONLY the vector"""\n')
        new_lines.append('        return f"[{pos.to_vector()}]"\n')
        
        # Skip the old method lines
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith('def ') and not lines[j].strip().startswith('class '):
            j += 1
            if lines[j-1].strip() and not lines[j-1].strip().startswith('#'):
                if 'return' in lines[j-1]:
                    break
        skip_lines = j - i - 1
    else:
        new_lines.append(line)

with open('app/neurochemistry/core/dimensional_emergence.py', 'w') as f:
    f.writelines(new_lines)
    
print("Fixed create_prompt_injection method")
