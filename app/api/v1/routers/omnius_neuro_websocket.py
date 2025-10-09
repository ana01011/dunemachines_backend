"""
WebSocket endpoint for Omnius with neurochemical streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
import json
import asyncio
from typing import Optional

try:
    import jwt
except ImportError:
    from jose import jwt

from app.core.config import settings
from app.neurochemistry.streaming.wave_streamer import NeurochemicalWaveStreamer
from app.agents.omnius_collaborative import omnius_neurochemical

router = APIRouter()

# Store active streamers per user
active_streamers: dict[str, NeurochemicalWaveStreamer] = {}

@router.websocket("/ws/omnius")
async def omnius_websocket(websocket: WebSocket):
    """
    WebSocket endpoint with continuous neurochemical streaming
    """
    await websocket.accept()
    
    streamer: Optional[NeurochemicalWaveStreamer] = None
    user_id: Optional[str] = None
    
    try:
        # Authenticate
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)
        
        if "token" not in auth_data:
            await websocket.send_json({"error": "No token provided"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Verify JWT
        try:
            payload = jwt.decode(
                auth_data["token"],
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            user_id = payload.get("sub")
            
            if not user_id:
                raise ValueError("Invalid token")
                
        except Exception as e:
            await websocket.send_json({"error": f"Auth failed: {str(e)}"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Initialize neurochemical streamer
        streamer = NeurochemicalWaveStreamer(user_id)
        active_streamers[user_id] = streamer
        
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "initial_state": streamer.get_current_state_summary()
        })
        
        # Start neurochemical streaming in background
        stream_task = asyncio.create_task(
            streamer.start_streaming(websocket)
        )
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                # Process user message
                user_message = message_data.get("content", "")
                
                # Message affects neurochemistry
                await streamer.process_user_message(user_message)
                
                # Start generation with expected difficulty
                complexity = streamer._estimate_complexity(user_message)
                generation_id = f"gen_{user_id}_{int(time.time())}"
                streamer.start_generation(generation_id, complexity)
                
                # Get neurochemical-modulated prompt
                neuro_prompt = streamer.get_current_prompt_injection()
                
                # Stream response with neurochemical modulation
                quality_score = 0
                token_count = 0
                
                async for chunk in omnius_neurochemical.think_stream(
                    message=user_message,
                    user_id=user_id,
                    temperature=0.7,
                    max_tokens=2000,
                    neurochemical_prompt=neuro_prompt  # Inject mood
                ):
                    # Send content chunk
                    await websocket.send_json(chunk)
                    
                    # Update generation progress
                    if chunk.get("type") == "content":
                        token_count += 1
                        progress = min(1.0, token_count / 100)
                        streamer.update_generation_progress(progress)
                    
                    # Track quality
                    if chunk.get("quality_score"):
                        quality_score = chunk["quality_score"]
                
                # Complete generation - trigger dopamine response
                streamer.complete_generation(quality_score or 0.7)
                
            elif message_data.get("type") == "get_state":
                # Send current state
                await websocket.send_json({
                    "type": "state_update",
                    "state": streamer.get_current_state_summary()
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if streamer:
            streamer.stop_streaming()
        if user_id and user_id in active_streamers:
            del active_streamers[user_id]
        try:
            await websocket.close()
        except:
            pass

import time  # Add at top of file
