import re

with open('app/agents/omnius_collaborative.py', 'r') as f:
    lines = f.readlines()

# First, add the import if not already there
import_found = False
for line in lines:
    if 'from app.neurochemistry.integrated_system import' in line:
        import_found = True
        break

if not import_found:
    # Add import after the last import
    for i, line in enumerate(lines):
        if line.startswith('from app.') and 'import' in line:
            last_app_import = i
    lines.insert(last_app_import + 1, 'from app.neurochemistry.integrated_system import create_enhanced_prompt, get_user_neurochemical_state\n')
    print("Added neurochemistry import")

# Now find think_stream and add neurochemistry
new_lines = []
in_think_stream = False
added_neuro = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Look for the sandbox line in think_stream
    if 'sandbox = self._get_user_sandbox(user_id)' in line and not added_neuro:
        # Add neurochemistry right after sandbox creation
        new_lines.append('\n')
        new_lines.append('        # Neurochemical consciousness integration\n')
        new_lines.append('        enhanced_message = create_enhanced_prompt(user_id, message)\n')
        new_lines.append('        neuro_state = get_user_neurochemical_state(user_id)\n')
        new_lines.append('        print(f"\\n🧬 Neurochemical State: {neuro_state[\'position\']}")\n')
        new_lines.append('        print(f"  Hormones: D={neuro_state[\'hormones\'][\'dopamine\']:.0f} C={neuro_state[\'hormones\'][\'cortisol\']:.0f} A={neuro_state[\'hormones\'][\'adrenaline\']:.0f}")\n')
        new_lines.append('\n')
        new_lines.append('        # Use enhanced message for all processing\n')
        new_lines.append('        original_message = message\n')
        new_lines.append('        message = enhanced_message\n')
        new_lines.append('\n')
        added_neuro = True

with open('app/agents/omnius_collaborative.py', 'w') as f:
    f.writelines(new_lines)

if added_neuro:
    print("✅ Successfully integrated neurochemistry into think_stream")
else:
    print("❌ Could not find the right place to add neurochemistry")
