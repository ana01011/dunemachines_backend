# Fix the imports in omnius_router.py
import sys

content = '''"""
Omnius API Router - Standalone Distributed Consciousness System
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.agents.omnius import omnius
from app.models.omnius import (
    OmniusRequest, 
    OmniusResponse, 
    ConsciousnessStatus,
    ConsciousnessRegion  # Add this import!
)
from app.models.auth import User
from app.api.v1.routers.auth_router import get_current_user_dependency
from app.core.database import db
from typing import Optional, List
import time
from uuid import uuid4
import asyncio

router = APIRouter()

@router.post("/chat", response_model=OmniusResponse)
async def chat_with_omnius(
    request: OmniusRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_dependency)
):
    """
    Direct interface to Omnius - The Evermind
    This is the most powerful AI endpoint with distributed consciousness
    """
    
    try:
        start_time = time.time()
        conversation_id = request.conversation_id or uuid4()
        message_id = uuid4()
        
        # Build enhanced context for Omnius
        context = {
            "user_id": str(current_user.id),
            "username": current_user.username,
            "email": current_user.email,
            "conversation_id": str(conversation_id),
            "thinking_mode": request.thinking_mode,
            "timestamp": str(time.time())
        }
        
        # Activate specific regions if requested
        if request.activate_regions:
            context["activate_regions"] = [r.value for r in request.activate_regions]
        
        # Store user message in database
        await db.execute("""
            INSERT INTO messages (id, conversation_id, role, content, personality)
            VALUES ($1, $2, 'user', $3, 'omnius')
        """, message_id, conversation_id, request.message)
        
        # Omnius processes with distributed consciousness
        response_text = await omnius.think(request.message, context)
        
        # Get consciousness status
        consciousness_status = omnius.get_status()
        
        # Determine which regions were actually used
        consciousness_used = []
        if consciousness_status['consciousness_regions']['prefrontal_cortex'] == 'active':
            consciousness_used.append('prefrontal_cortex')
        if consciousness_status['consciousness_regions']['code_cortex'] == 'active':
            consciousness_used.append('code_cortex')
        if 'code' in request.message.lower() or 'program' in request.message.lower():
            consciousness_used.append('code_cortex')
        
        # Calculate metrics
        processing_time = time.time() - start_time
        tokens = len(response_text.split())
        
        # Store Omnius response
        response_id = uuid4()
        background_tasks.add_task(
            store_omnius_response,
            response_id, conversation_id, response_text, tokens, processing_time
        )
        
        # Build enhanced response
        return OmniusResponse(
            response=response_text,
            conversation_id=conversation_id,
            message_id=response_id,
            consciousness_used=consciousness_used,
            consciousness_status=ConsciousnessStatus(
                prefrontal_cortex=consciousness_status['consciousness_regions']['prefrontal_cortex'],
                code_cortex=consciousness_status['consciousness_regions']['code_cortex'],
                math_region=consciousness_status['consciousness_regions']['math_region'],
                creative_center=consciousness_status['consciousness_regions']['creative_center'],
                total_parameters=consciousness_status['total_parameters'],
                active_regions=len([r for r in consciousness_status['consciousness_regions'].values() if r == 'active']),
                processing_power=0.7  # Calculate based on actual usage
            ),
            thinking_process=f"Processed through {len(consciousness_used)} consciousness regions",
            tokens_used=tokens,
            processing_time=processing_time,
            parallel_thoughts=len(consciousness_used),
            confidence_score=0.95,  # Omnius is highly confident
            memory_accessed=len(context) > 5,
            context_window_used=min(tokens * 4, 2048),
            user_context=context
        )
    
    except Exception as e:
        print(f"Omnius error: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness error: {str(e)}")

@router.get("/status", response_model=ConsciousnessStatus)
async def get_omnius_consciousness_status():
    """Get detailed Omnius consciousness status"""
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

@router.post("/activate-region")
async def activate_consciousness_region(
    region: ConsciousnessRegion,
    current_user: User = Depends(get_current_user_dependency)
):
    """Activate a specific consciousness region (admin only)"""
    # TODO: Add admin check
    
    if region == ConsciousnessRegion.CODE:
        from app.services.deepseek_coder_service import deepseek_coder
        deepseek_coder.load_model()
        return {"status": f"{region.value} activated"}
    elif region == ConsciousnessRegion.MATH:
        return {"status": f"{region.value} not yet installed"}
    else:
        return {"status": f"{region.value} already active"}

@router.get("/conversations")
async def get_omnius_conversations(
    current_user: User = Depends(get_current_user_dependency)
):
    """Get all Omnius conversations for the user"""
    conversations = await db.fetch("""
        SELECT DISTINCT c.* 
        FROM conversations c
        JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = $1 AND m.personality = 'omnius'
        ORDER BY c.last_message_at DESC
        LIMIT 20
    """, current_user.id)
    
    return [dict(c) for c in conversations]

@router.post("/think")
async def omnius_deep_think(
    query: str,
    duration: int = 5,
    current_user: User = Depends(get_current_user_dependency)
):
    """
    Let Omnius think deeply about something for a specified duration
    Uses all available consciousness regions
    """
    # TODO: Implement deep thinking with multiple iterations
    return {
        "status": "thinking",
        "duration": duration,
        "query": query,
        "message": "Omnius is contemplating deeply..."
    }

async def store_omnius_response(response_id, conversation_id, response_text, tokens, processing_time):
    """Background task to store Omnius response"""
    await db.execute("""
        INSERT INTO messages (id, conversation_id, role, content, personality, tokens_used)
        VALUES ($1, $2, 'assistant', $3, 'omnius', $4)
    """, response_id, conversation_id, response_text, tokens)
    
    # Update conversation
    await db.execute("""
        UPDATE conversations
        SET last_message_at = CURRENT_TIMESTAMP,
            message_count = message_count + 1
        WHERE id = $1
    """, conversation_id)
'''

# Write the fixed file
with open('app/api/v1/routers/omnius_router.py', 'w') as f:
    f.write(content)

print("✅ Fixed omnius_router.py imports")
