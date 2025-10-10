"""Test Omnius with neurochemistry integration"""
import asyncio
import sys
sys.path.append('/root/openhermes_backend')

from app.agents.omnius_collaborative import omnius_neurochemical

async def test_omnius_with_neuro():
    user_id = "test_user_456"
    
    # Test with emotional message
    message = "I'm really frustrated! This error keeps happening!"
    
    print("=" * 60)
    print("TESTING OMNIUS WITH NEUROCHEMISTRY")
    print("=" * 60)
    print(f"\nMessage: {message}")
    print("\nGenerating response with neurochemical state...\n")
    
    # Collect the stream
    async for chunk in omnius_neurochemical.think_stream(
        message=message,
        user_id=user_id,
        temperature=0.7,
        max_tokens=500
    ):
        if chunk.get("type") == "content":
            print(chunk["content"], end="", flush=True)
        elif chunk.get("type") == "status":
            print(f"\n📊 {chunk['message']}")
    
    print("\n\nDone!")

if __name__ == "__main__":
    asyncio.run(test_omnius_with_neuro())
