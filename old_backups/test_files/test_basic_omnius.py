"""
Test basic Omnius without neurochemistry issues
"""
import asyncio
from app.agents.omnius_neurochemical import omnius_neurochemical
from app.core.database import db

async def test():
    await db.connect()
    
    # Test without neurochemistry (basic mode)
    omnius_neurochemical.is_initialized = False  # Force basic mode
    
    context = {
        "user_id": "test-user",
        "username": "test",
        "email": "test@example.com"
    }
    
    response, metadata = await omnius_neurochemical.think(
        "Write a simple hello world function",
        context
    )
    
    print("Response:", response[:200])
    print("Metadata:", metadata)
    
    await db.disconnect()

asyncio.run(test())
