# Fix the conversation creation in omnius_router.py
import re

# Read current file
with open('app/api/v1/routers/omnius_router.py', 'r') as f:
    content = f.read()

# Find the section after context creation and before "Omnius processes"
# We need to insert conversation creation logic

# Look for where we store user message
if "# Store user message in database" in content:
    # Replace the section with proper conversation handling
    
    # Split at the comment
    parts = content.split("# Store user message in database")
    
    # Find the end of the await db.execute block
    second_part = parts[1]
    lines = second_part.split('\n')
    
    # Find where the next comment or code block starts
    for i, line in enumerate(lines):
        if "# Omnius processes" in line:
            break
    
    # Reconstruct with new code
    new_code = '''# Create or get conversation
        if request.conversation_id:
            # Check if conversation exists
            existing_conv = await db.fetchrow(
                "SELECT id FROM conversations WHERE id = $1 AND user_id = $2",
                conversation_id, current_user.id
            )
            if not existing_conv:
                # Create conversation with provided ID
                await db.execute("""
                    INSERT INTO conversations (id, user_id, title, personality, started_at, message_count)
                    VALUES ($1, $2, $3, 'omnius', CURRENT_TIMESTAMP, 0)
                """, conversation_id, current_user.id, request.message[:100])
        else:
            # Create new conversation
            await db.execute("""
                INSERT INTO conversations (id, user_id, title, personality, started_at, message_count)
                VALUES ($1, $2, $3, 'omnius', CURRENT_TIMESTAMP, 0)
            """, conversation_id, current_user.id, request.message[:100])
        
        # Store user message in database
        await db.execute("""
            INSERT INTO messages (id, conversation_id, role, content, personality)
            VALUES ($1, $2, 'user', $3, 'omnius')
        """, message_id, conversation_id, request.message)
        
        '''
    
    # Rebuild the content
    content = parts[0] + new_code + "# Omnius processes" + content.split("# Omnius processes")[1]

# Write back
with open('app/api/v1/routers/omnius_router.py', 'w') as f:
    f.write(content)

print("✅ Fixed conversation creation in omnius_router.py")
