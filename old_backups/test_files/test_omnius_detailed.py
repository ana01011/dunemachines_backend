"""
Detailed test script to verify Omnius is working correctly
"""
import asyncio
import json
from app.agents.omnius_neurochemical import omnius_neurochemical
from app.core.database import db

async def test_omnius():
    print("=" * 50)
    print("TESTING OMNIUS NEUROCHEMICAL SYSTEM")
    print("=" * 50)
    
    # Connect to database
    await db.connect()
    print("✅ Database connected")
    
    # Initialize Omnius
    await omnius_neurochemical.initialize(db.pool)
    print("✅ Omnius initialized")
    
    # Check initialization status
    print(f"\n📊 Initialization Status:")
    print(f"  - Is initialized: {omnius_neurochemical.is_initialized}")
    print(f"  - State repository: {omnius_neurochemical.state_repository is not None}")
    print(f"  - Persistence manager: {omnius_neurochemical.persistence_manager is not None}")
    
    # Get status
    status = omnius_neurochemical.get_status()
    print(f"\n🧬 Omnius Status:")
    print(json.dumps(status, indent=2))
    
    # Test token checking
    test_user_id = "9772bf5d-8bd9-4b8c-9fa3-8301ad221cc7"
    tokens = await omnius_neurochemical.check_tokens(test_user_id)
    print(f"\n💰 Token Status for user:")
    print(f"  - Has tokens: {tokens['has_tokens']}")
    print(f"  - Remaining: {tokens['tokens_remaining']}")
    
    # Test thinking without code
    print("\n🧠 Testing general thinking...")
    context = {
        "user_id": test_user_id,
        "username": "test_user",
        "email": "test@example.com",
        "conversation_id": "test-conv-1"
    }
    
    response1, metadata1 = await omnius_neurochemical.think(
        "What is artificial intelligence?", 
        context
    )
    print(f"  - Response length: {len(response1)}")
    print(f"  - Regions used: {metadata1.get('consciousness_used')}")
    print(f"  - Neurochemistry active: {metadata1.get('neurochemistry_active')}")
    
    # Test thinking with code
    print("\n💻 Testing code generation...")
    response2, metadata2 = await omnius_neurochemical.think(
        "Write a Python function to calculate factorial",
        context
    )
    print(f"  - Response length: {len(response2)}")
    print(f"  - Regions used: {metadata2.get('consciousness_used')}")
    print(f"  - Has code blocks: {'```' in response2}")
    
    # Check if DeepSeek is actually being used
    print("\n🔍 Verifying Code Cortex usage:")
    print(f"  - Code keywords detected: {omnius_neurochemical._needs_code('Write a function')}")
    print(f"  - Code Cortex model loaded: {hasattr(omnius_neurochemical.models['code_cortex'], 'model')}")
    
    # Check neurochemical tables
    print("\n📊 Checking neurochemical database tables:")
    tables = await db.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE '%neurochemical%' OR table_name = 'user_tokens'
    """)
    for table in tables:
        count = await db.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
        print(f"  - {table['table_name']}: {count} records")
    
    await db.disconnect()
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_omnius())
