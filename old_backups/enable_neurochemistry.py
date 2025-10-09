# Find line in app/agents/omnius_neurochemical.py around line 75-80
# Change from:
#     # For now, just use basic mode until neurochemistry is fully debugged
#     return await self._think_basic(message, context)
# 
# To:
#     if token_status['has_tokens'] and self.is_initialized:
#         return await self._think_neurochemical(message, context, token_status)
#     else:
#         return await self._think_basic(message, context)

import fileinput
import sys

for line in fileinput.input('app/agents/omnius_neurochemical.py', inplace=True):
    if "# For now, just use basic mode" in line:
        print("        # Neurochemistry is now ENABLED!")
    elif "return await self._think_basic(message, context)" in line and "For now" in prev_line:
        print("        if token_status['has_tokens'] and self.is_initialized:")
        print("            try:")
        print("                return await self._think_neurochemical(message, context, token_status)")
        print("            except Exception as e:")
        print("                logger.error(f'Neurochemistry error: {e}')")
        print("                return await self._think_basic(message, context)")
        print("        else:")
        print("            return await self._think_basic(message, context)")
    else:
        print(line, end='')
    prev_line = line
