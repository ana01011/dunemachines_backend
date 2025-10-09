"""
Omnius Router with Neurochemistry Integration - Fixed for existing schema
"""
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from app.agents.omnius_neurochemical import omnius_neurochemical
from app.models.omnius import OmniusRequest, OmniusResponse, ConsciousnessStatus
from app.models.auth import User
from app.api.v1.routers.auth_router import get_current_user_dependency
from app.core.database import db
from app.websocket.chat_websocket import manager, get_current_user_from_websocket
import time
from uuid import uuid4
from datetime import datetime
import asyncio
import json

router = APIRouter()

@router.post("/chat", response_model=OmniusResponse)
async def chat_with_omnius(
    request: OmniusRequest,
    current_user: User = Depends(get_current_user_dependency)
):
    """Chat with Omnius - Now with Neurochemistry"""
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
        
        # Ensure conversation exists
        existing = await db.fetchrow(
            "SELECT id FROM conversations WHERE id = $1 AND user_id = $2",
            conversation_id, current_user.id
        )
        
        if not existing:
            await db.execute("""
                INSERT INTO conversations (id, user_id, title, started_at, message_count, last_message_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, conversation_id, current_user.id, f"Omnius: {request.message[:50]}",
                datetime.now(), 0, datetime.now())
        
        # Store user message (using only existing columns)
        await db.execute("""
            INSERT INTO messages (id, conversation_id, user_id, role, content, created_at)
            VALUES ($1, $2, $3, 'user', $4, $5)
        """, uuid4(), conversation_id, current_user.id, request.message, datetime.now())
        
        # Get response with neurochemistry
        response_text, metadata = await omnius_neurochemical.think(request.message, context)
        
        processing_time = time.time() - start_time
        
        # Store AI response (using only existing columns)
        # Note: We'll store neurochemical data in a separate table or in the neurochemical_states table
        await db.execute("""
            INSERT INTO messages (
                id, conversation_id, user_id, role, content, 
                created_at, tokens, response_time
            )
            VALUES ($1, $2, $3, 'assistant', $4, $5, $6, $7)
        """, message_id, conversation_id, current_user.id, response_text,
            datetime.now(), 
            len(response_text) // 4,  # Approximate tokens
            processing_time)
        
        # If neurochemistry is active, save the mood state
        if metadata.get('neurochemistry_active') and metadata.get('mood'):
            # Save to neurochemical_states table instead
            try:
                await db.execute("""
                    INSERT INTO neurochemical_states (
                        user_id, timestamp, 
                        valence, arousal, dominance,
                        event_context
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, current_user.id, datetime.now(),
                    metadata['mood'].get('valence'),
                    metadata['mood'].get('arousal'),
                    metadata['mood'].get('dominance'),
                    json.dumps({'message_id': str(message_id), 'conversation_id': str(conversation_id)})
                )
            except:
                pass  # Ignore if neurochemical tables don't exist yet
        
        # Build response
        response = OmniusResponse(
            response=response_text,
            conversation_id=conversation_id,
            message_id=message_id,
            consciousness_used=metadata.get('consciousness_used', ['prefrontal_cortex']),
            consciousness_status=omnius_neurochemical.get_status(),
            thinking_process=f"{'Neurochemical' if metadata.get('neurochemistry_active') else 'Standard'} processing",
            tokens_used=metadata.get('tokens_used', 0),
            processing_time=processing_time,
            parallel_thoughts=len(metadata.get('consciousness_used', [])),
            confidence_score=metadata.get('behavior', {}).get('confidence', 0.8) if metadata.get('behavior') else 0.8,
            memory_accessed=True,
            context_window_used=len(request.message),
            user_context=context
        )
        
        # Add neurochemical data if active
        if metadata.get('neurochemistry_active'):
            response.user_context['mood'] = metadata.get('mood')
            response.user_context['tokens_remaining'] = metadata.get('tokens_remaining')
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consciousness error: {str(e)}")

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with neurochemistry"""
    user = None
    connection_id = None
    
    try:
        # Authenticate user
        user = await get_current_user_from_websocket(websocket)
        user_id = str(user['id'])
        
        # Connect
        connection_id = await manager.connect(websocket, user_id)
        
        # Send initial status
        token_status = await omnius_neurochemical.check_tokens(user_id)
        await manager.send_token_update(user_id, token_status['tokens_remaining'])
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get('type') == 'message':
                message = data.get('content', '')
                
                # Send thinking indicator
                await manager.send_thinking_indicator(user_id, "Processing your request...")
                
                # Build context
                context = {
                    'user_id': user_id,
                    'username': user['username'],
                    'email': user['email'],
                    'conversation_id': data.get('conversation_id', str(uuid4()))
                }
                
                # Get response with neurochemistry
                response_text, metadata = await omnius_neurochemical.think(message, context)
                
                # Send mood update if neurochemistry is active
                if metadata.get('neurochemistry_active') and metadata.get('mood'):
                    await manager.send_mood_update(user_id, metadata['mood'])
                
                # Stream response in chunks
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    await manager.send_response_chunk(user_id, chunk)
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Send completion signal
                await websocket.send_json({
                    'type': 'complete',
                    'metadata': {
                        'tokens_used': metadata.get('tokens_used', 0),
                        'tokens_remaining': metadata.get('tokens_remaining'),
                        'neurochemistry_active': metadata.get('neurochemistry_active', False)
                    }
                })
                
                # Update token display
                if metadata.get('tokens_remaining') is not None:
                    await manager.send_token_update(user_id, metadata['tokens_remaining'])
                
    except WebSocketDisconnect:
        if connection_id and user:
            manager.disconnect(connection_id, str(user['id']))
    except Exception as e:
        await websocket.send_json({'type': 'error', 'message': str(e)})
        await websocket.close()

@router.get("/tokens/status")
async def get_token_status(current_user: User = Depends(get_current_user_dependency)):
    """Get user's neurochemistry token status"""
    token_status = await omnius_neurochemical.check_tokens(str(current_user.id))
    return token_status

@router.post("/tokens/purchase")
async def purchase_tokens(
    amount: int,
    current_user: User = Depends(get_current_user_dependency)
):
    """Purchase additional neurochemistry tokens"""
    # This is a placeholder - integrate with payment system
    await db.execute("""
        UPDATE user_tokens 
        SET daily_tokens = daily_tokens + $2,
            total_tokens_purchased = total_tokens_purchased + $2
        WHERE user_id = $1
    """, current_user.id, amount)
    
    return {"message": f"Purchased {amount} tokens", "success": True}
