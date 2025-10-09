# Read the chat_router.py file
with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()

# Find where to insert the new endpoint (before the get conversations endpoint)
import_section = "from uuid import uuid4"
if "from datetime import datetime, timedelta" in content and "from uuid import uuid4" not in content:
    # Add uuid4 import if not present
    content = content.replace(
        "from datetime import datetime, timedelta",
        "from datetime import datetime, timedelta\nfrom uuid import uuid4"
    )

# Add the new endpoint before @router.get("/conversations")
new_endpoint = '''
@router.post("/conversations/new")
async def create_new_conversation(
    title: Optional[str] = None,
    current_user: User = Depends(get_current_user_dependency)
):
    """Create a new empty conversation"""
    conversation_id = uuid4()
    
    # Use "New Chat" as default title
    conversation_title = title or "New Chat"
    
    await db.execute("""
        INSERT INTO conversations (id, user_id, title, started_at, message_count)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP, 0)
    """, conversation_id, current_user.id, conversation_title)
    
    return {
        "id": str(conversation_id),
        "title": conversation_title,
        "message": "New conversation created",
        "success": True
    }

'''

# Find the line with @router.get("/conversations"
lines = content.split('\n')
insert_index = None
for i, line in enumerate(lines):
    if '@router.get("/conversations"' in line:
        insert_index = i
        break

if insert_index:
    # Insert the new endpoint before the get conversations endpoint
    lines.insert(insert_index, new_endpoint)
    content = '\n'.join(lines)
    
    # Write back
    with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
        f.write(content)
    
    print("✅ Added create conversation endpoint")
else:
    print("❌ Could not find where to insert endpoint")
    print("\nAdd this endpoint manually to chat_router.py:")
    print(new_endpoint)
