import asyncio
from app.core.database import db

async def check_schema():
    await db.connect()
    
    # List all tables
    tables = await db.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    print("=" * 60)
    print("YOUR EXISTING TABLES:")
    print("=" * 60)
    for table in tables:
        print(f"  • {table['table_name']}")
    
    # Check conversations table structure
    print("\n" + "=" * 60)
    print("CONVERSATIONS TABLE STRUCTURE:")
    print("=" * 60)
    columns = await db.fetch("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'conversations'
        ORDER BY ordinal_position
    """)
    
    for col in columns:
        print(f"  {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
    
    # Check messages table structure
    print("\n" + "=" * 60)
    print("MESSAGES TABLE STRUCTURE:")
    print("=" * 60)
    columns = await db.fetch("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'messages'
        ORDER BY ordinal_position
    """)
    
    for col in columns:
        print(f"  {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
    
    # Check if personality column exists
    personality_in_conversations = await db.fetchval("""
        SELECT COUNT(*) 
        FROM information_schema.columns 
        WHERE table_name = 'conversations' AND column_name = 'personality'
    """)
    
    personality_in_messages = await db.fetchval("""
        SELECT COUNT(*) 
        FROM information_schema.columns 
        WHERE table_name = 'messages' AND column_name = 'personality'
    """)
    
    print("\n" + "=" * 60)
    print("COLUMN CHECK:")
    print("=" * 60)
    print(f"  conversations.personality exists: {'YES' if personality_in_conversations else 'NO'}")
    print(f"  messages.personality exists: {'YES' if personality_in_messages else 'NO'}")
    
    await db.disconnect()

asyncio.run(check_schema())
