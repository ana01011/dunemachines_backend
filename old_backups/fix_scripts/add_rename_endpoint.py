import re

with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()

# Check if rename endpoint exists
if '/conversations/{conversation_id}/rename' not in content and 'rename' not in content.lower():
    # Add rename endpoint before the last route or at the end
    rename_endpoint = '''
@router.patch("/conversations/{conversation_id}/rename")
async def rename_conversation(
    conversation_id: str,
    title: dict,
    current_user: User = Depends(get_current_user_dependency)
):
    """Rename a conversation"""
    new_title = title.get('title', '')[:100]  # Limit title length
    
    result = await db.execute("""
        UPDATE conversations 
        SET title = $3
        WHERE id = $1 AND user_id = $2
    """, conversation_id, current_user.id, new_title)
    
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation renamed", "title": new_title}
'''
    
    # Find a good place to insert (before the last function)
    last_router_pos = content.rfind('@router.')
    if last_router_pos > 0:
        # Find the end of the previous function
        prev_function_end = content.rfind('\n\n', 0, last_router_pos)
        if prev_function_end > 0:
            content = content[:prev_function_end] + '\n' + rename_endpoint + content[prev_function_end:]
            
            with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
                f.write(content)
            print("✅ Added rename endpoint to backend")
    else:
        print("⚠️ Could not find insertion point")
else:
    print("✅ Rename functionality already exists")
