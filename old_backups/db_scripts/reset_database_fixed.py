#!/usr/bin/env python3
import asyncio
import asyncpg
from datetime import datetime

async def reset_database():
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='sarah_user',
        password='sarah_secure_2024',
        database='sarah_ai_fresh'
    )
    
    # Get current counts
    user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
    msg_count = await conn.fetchval("SELECT COUNT(*) FROM messages")
    conv_count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
    
    print("\n" + "="*60)
    print("DATABASE RESET WARNING")
    print("="*60)
    print(f"This will delete:")
    print(f"  • {user_count} users")
    print(f"  • {msg_count} messages")
    print(f"  • {conv_count} conversations")
    print(f"  • All relationships, profiles, and user data")
    print("="*60)
    
    confirm = input("\nType 'DELETE ALL' to confirm: ")
    
    if confirm != "DELETE ALL":
        print("❌ Cancelled - no data was deleted")
        await conn.close()
        return
    
    print("\n🔄 Deleting all data...")
    
    try:
        # Start transaction
        async with conn.transaction():
            # Delete in correct order
            await conn.execute("DELETE FROM messages")
            print("  ✓ Messages deleted")
            
            await conn.execute("DELETE FROM conversations")
            print("  ✓ Conversations deleted")
            
            await conn.execute("DELETE FROM relationship_events")
            print("  ✓ Relationship events deleted")
            
            await conn.execute("DELETE FROM relationships")
            print("  ✓ Relationships deleted")
            
            await conn.execute("DELETE FROM user_facts")
            print("  ✓ User facts deleted")
            
            await conn.execute("DELETE FROM user_profiles")
            print("  ✓ User profiles deleted")
            
            await conn.execute("DELETE FROM sessions")
            print("  ✓ Sessions deleted")
            
            # No otp_codes table - OTP data is stored in users table
            
            await conn.execute("DELETE FROM users")
            print("  ✓ Users deleted")
        
        print("\n✅ Database reset complete!")
        print("All users and data have been deleted.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Transaction rolled back - no data was deleted")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_database())
