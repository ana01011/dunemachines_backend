import asyncio
from app.core.database import db

async def clean_gud_references():
    await db.connect()
    
    user = await db.fetchrow("""
        SELECT id FROM users 
        WHERE email = 'uzma.maryam102021@gmail.com'
    """)
    
    if user:
        # Clean up all wrong name references
        await db.execute("""
            DELETE FROM user_facts 
            WHERE user_id = $1 
            AND fact_type = 'name' 
            AND fact_value IN ('Gud', 'but', 'User')
        """, user['id'])
        
        # Ensure correct name is set
        await db.execute("""
            UPDATE user_profiles 
            SET name = 'Uzma'
            WHERE user_id = $1
        """, user['id'])
        
        # Insert correct fact if not exists
        await db.execute("""
            INSERT INTO user_facts (user_id, fact_type, fact_value, source)
            VALUES ($1, 'name', 'Uzma', 'profile')
            ON CONFLICT (user_id, fact_type, fact_value) DO NOTHING
        """, user['id'])
        
        print("✅ Cleaned up all 'Gud' references")
        
        # Verify
        profile = await db.fetchrow("""
            SELECT name FROM user_profiles WHERE user_id = $1
        """, user['id'])
        
        facts = await db.fetch("""
            SELECT fact_value FROM user_facts 
            WHERE user_id = $1 AND fact_type = 'name'
        """, user['id'])
        
        print(f"Profile name: {profile['name']}")
        print(f"Name facts: {[f['fact_value'] for f in facts]}")
    
    await db.disconnect()

asyncio.run(clean_gud_references())
