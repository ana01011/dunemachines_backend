#!/usr/bin/env python3
import asyncio
import asyncpg

async def safe_delete_unverified():
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='sarah_user',
        password='sarah_secure_2024',
        database='sarah_ai_fresh'
    )
    
    # Get all unverified users
    unverified = await conn.fetch("""
        SELECT id, email, username 
        FROM users 
        WHERE is_verified = false
    """)
    
    print(f"\nFound {len(unverified)} unverified users to delete:")
    for user in unverified:
        print(f"  - {user['email']} ({user['username']})")
    
    if not unverified:
        print("No unverified users to delete")
        await conn.close()
        return
    
    confirm = input("\nDelete all these users and their data? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled")
        await conn.close()
        return
    
    # Delete in correct order to avoid foreign key constraints
    for user in unverified:
        user_id = user['id']
        
        # 1. Delete messages first
        await conn.execute("""
            DELETE FROM messages 
            WHERE user_id = $1 OR conversation_id IN (
                SELECT id FROM conversations WHERE user_id = $1
            )
        """, user_id)
        
        # 2. Delete conversations
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", user_id)
        
        # 3. Delete relationships
        await conn.execute("DELETE FROM relationships WHERE user_id = $1", user_id)
        
        # 4. Delete relationship events
        await conn.execute("DELETE FROM relationship_events WHERE user_id = $1", user_id)
        
        # 5. Delete user facts
        await conn.execute("DELETE FROM user_facts WHERE user_id = $1", user_id)
        
        # 6. Delete user profile
        await conn.execute("DELETE FROM user_profiles WHERE user_id = $1", user_id)
        
        # 7. Delete sessions
        await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)
        
        # 8. Finally delete the user
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
        
        print(f"✓ Deleted: {user['email']}")
    
    print(f"\n✅ Successfully deleted {len(unverified)} unverified users")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(safe_delete_unverified())
