import sys
sys.path.insert(0, '/root/openhermes_backend')

import asyncio
from app.services.theme_service import theme_service

async def test():
    messages = [
        "Switch to Pure Light theme",
        "Change to Simple Dark", 
        "I want dark mode",
        "Use Tech Blue theme",
        "switch theme to Cyber Dark",
        "set theme Neon Nights"
    ]
    
    print("Testing Theme Detection:")
    print("="*60)
    
    for msg in messages:
        result = await theme_service.detect_theme_command(msg)
        if result:
            print(f"✅ '{msg}' -> {result}")
        else:
            print(f"❌ '{msg}' -> Not detected")

asyncio.run(test())
