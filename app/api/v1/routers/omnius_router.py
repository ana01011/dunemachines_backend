"""
Omnius Router - Working with YOUR exact database schema
"""
from fastapi import APIRouter, HTTPException, Depends
from app.agents.omnius import omnius
from app.models.omnius import OmniusRequest, OmniusResponse, ConsciousnessStatus, ConsciousnessRegion
from app.models.auth import User
from app.api.v1.routers.auth_router import get_current_user_dependency
from app.core.database import db
import time
from uuid import uuid4
from datetime import datetime

router = APIRouter()

@router.post("/chat", response_model=OmniusResponse)
async def chat_with_omnius(
    request: OmniusRequest,
    current_user: User = Depends(get_current_user_dependency)
):
    """Chat with Omnius - The Evermind"""

    try:
        start_time = time.time()
        conversation_id = request.conversation_id or uuid4()
        message_id = uuid4()

        # Build context
        context = {
            "user_id": str(current_user.id),
            "username": current_user.username,
            "email": current_user.email,
            "conversation_id": str(conversation_id),
            "thinking_mode": request.thinking_mode
        }

        # Check if conversation exists
        existing = await db.fetchrow(
            "SELECT id FROM conversations WHERE id = $1 AND user_id = $2",
            conversation_id, current_user.id
        )

        if not existing:
            # Create new conversation (using YOUR schema exactly)
            await db.execute("""
                INSERT INTO conversations (id, user_id, title, started_at, message_count, last_message_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, conversation_id, current_user.id, f"Omnius: {request.message[:50]}",
                datetime.now(), 0, datetime.now())

        # Store user message (using YOUR messages schema)
        await db.execute("""
            INSERT INTO messages (id, conversation_id, user_id, role, content, personality, created_at, tokens, response_time)
            VALUES ($1, $2, $3, 'user', $4, 'omnius', $5, $6, $7)
        """, message_id, conversation_id, current_user.id, request.message,
            datetime.now(), len(request.message.split()), 0.0)

        # Omnius processes the thought - NOW RETURNS BOTH RESPONSE AND REGIONS USED
        response_text, consciousness_used = await omnius.think(request.message, context)

        # Calculate processing time
        processing_time = time.time() - start_time
        tokens_used = len(response_text.split())

        # Store Omnius response
        response_id = uuid4()
        await db.execute("""
            INSERT INTO messages (id, conversation_id, user_id, role, content, personality, created_at, tokens, response_time)
            VALUES ($1, $2, $3, 'assistant', $4, 'omnius', $5, $6, $7)
        """, response_id, conversation_id, current_user.id, response_text,
            datetime.now(), tokens_used, processing_time)

        # Update conversation
        await db.execute("""
            UPDATE conversations
            SET last_message_at = $2,
                message_count = message_count + 2
            WHERE id = $1
        """, conversation_id, datetime.now())

        # Get consciousness status
        status = omnius.get_status()

        # Build the enhanced response
        return OmniusResponse(
            response=response_text,
            conversation_id=conversation_id,
            message_id=response_id,
            consciousness_used=consciousness_used,  # NOW USING ACTUAL REGIONS USED
            consciousness_status=ConsciousnessStatus(
                prefrontal_cortex=status['consciousness_regions']['prefrontal_cortex'],
                code_cortex=status['consciousness_regions']['code_cortex'],
                math_region=status['consciousness_regions']['math_region'],
                creative_center=status['consciousness_regions']['creative_center'],
                total_parameters=status['total_parameters'],
                active_regions=len([r for r in status['consciousness_regions'].values() if r == 'active']),
                processing_power=0.7
            ),
            thinking_process=f"Processed through {len(consciousness_used)} consciousness regions using distributed thinking",
            tokens_used=tokens_used,
            processing_time=processing_time,
            parallel_thoughts=len(consciousness_used),
            confidence_score=0.95,
            memory_accessed=True,
            context_window_used=min(tokens_used * 4, 2048),
            user_context=context
        )

    except Exception as e:
        import traceback
        print(f"Omnius error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/status")
async def get_omnius_status():
    """Get Omnius consciousness status"""
    status = omnius.get_status()
    return ConsciousnessStatus(
        prefrontal_cortex=status['consciousness_regions']['prefrontal_cortex'],
        code_cortex=status['consciousness_regions']['code_cortex'],
        math_region=status['consciousness_regions']['math_region'],
        creative_center=status['consciousness_regions']['creative_center'],
        total_parameters=status['total_parameters'],
        active_regions=len([r for r in status['consciousness_regions'].values() if r == 'active']),
        processing_power=0.7
    )

@router.get("/conversations")
async def get_omnius_conversations(
    current_user: User = Depends(get_current_user_dependency)
):
    """Get all Omnius conversations for the user"""

    # Get conversations that have Omnius messages
    conversations = await db.fetch("""
        SELECT DISTINCT c.*
        FROM conversations c
        INNER JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = $1
        AND m.personality = 'omnius'
        ORDER BY c.last_message_at DESC
        LIMIT 20
    """, current_user.id)

    return [dict(c) for c in conversations]

@router.get("/conversation/{conversation_id}/messages")
async def get_omnius_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user_dependency)
):
    """Get all messages in an Omnius conversation"""

    # Verify user owns the conversation
    conv = await db.fetchrow(
        "SELECT id FROM conversations WHERE id = $1 AND user_id = $2",
        conversation_id, current_user.id
    )

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    messages = await db.fetch("""
        SELECT * FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at ASC
    """, conversation_id)

    return [dict(m) for m in messages]