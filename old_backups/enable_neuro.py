# Read the file
with open('app/agents/omnius_neurochemical.py', 'r') as f:
    lines = f.readlines()

# Find and replace the think method section
new_lines = []
for i, line in enumerate(lines):
    if "# For now, just use basic mode" in line:
        new_lines.append("        # NEUROCHEMISTRY IS NOW FULLY ENABLED!\n")
    elif "return await self._think_basic(message, context)" in line and i > 70 and i < 85:
        new_lines.append("        if token_status['has_tokens'] and self.is_initialized:\n")
        new_lines.append("            try:\n")
        new_lines.append("                return await self._think_neurochemical(message, context, token_status)\n")
        new_lines.append("            except Exception as e:\n")
        new_lines.append("                logger.error(f'Neurochemistry error: {e}')\n")
        new_lines.append("                return await self._think_basic(message, context)\n")
        new_lines.append("        else:\n")
        new_lines.append("            return await self._think_basic(message, context)\n")
    else:
        new_lines.append(line)

# Write back
with open('app/agents/omnius_neurochemical.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Neurochemistry ENABLED!")
