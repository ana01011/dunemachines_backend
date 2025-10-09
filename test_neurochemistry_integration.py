"""Test the integrated neurochemistry system"""
import asyncio
import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.integrated_system import orchestrator, create_enhanced_prompt

async def test_neurochemistry():
    user_id = "test_user_123"
    
    # Test different types of messages
    test_messages = [
        "Hello, how are you?",
        "URGENT! The server is crashing!",
        "I'm really frustrated with this error",
        "This is amazing! Great work!",
        "Can you explain recursion?",
        "I hate this stupid bug",
        "Everything is working perfectly now"
    ]
    
    print("=" * 60)
    print("NEUROCHEMISTRY INTEGRATION TEST")
    print("=" * 60)
    
    for message in test_messages:
        enhanced = create_enhanced_prompt(user_id, message)
        state = orchestrator.get_user_state(user_id)
        
        print(f"\nMessage: {message}")
        print(f"3D Position: {state['position']}")
        print(f"Enhanced prompt: {enhanced[:100]}...")
        print(f"Hormones: D={state['hormones']['dopamine']:.1f} "
              f"C={state['hormones']['cortisol']:.1f} "
              f"A={state['hormones']['adrenaline']:.1f}")
        
        await asyncio.sleep(0.5)
    
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(test_neurochemistry())
