import sys
sys.path.insert(0, '/root/openhermes_backend')

import asyncio
from app.services.theme_service import theme_service

async def test():
    test_messages = [
        "Switch to Pure Light theme",
        "Change to Simple Dark",
        "I want dark mode",
        "Use Tech Blue theme",
        "switch theme to Cyber Dark",
        "set theme Neon Nights",
        "change theme to Deep Ocean",
        "activate Neon Nights theme",
        "I want Simple Dark theme",
        "Switch to Pure Light",
        "Change to Simple Dark theme",
        "dark mode",
        "light mode"
    ]
    
    print("Testing Theme Detection Patterns:")
    print("="*60)
    
    for msg in test_messages:
        result = await theme_service.detect_theme_command(msg)
        if result:
            action, theme, reason = result
            print(f"✅ '{msg}'")
            print(f"   → {action}: {theme}")
        else:
            print(f"❌ '{msg}' - NOT DETECTED")
    
    print("\n" + "="*60)
    print("Available themes:", theme_service.AVAILABLE_THEMES)

asyncio.run(test())
