"""
WebSocket handler for real-time chat with neurochemistry
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, Optional
import json
import asyncio
from datetime import datetime
from jose import jwt, JWTError
from app.core.config import settings
from app.core.database import db
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> connection_id
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        connection_id = f"{user_id}_{datetime.now().timestamp()}"
        self.active_connections[connection_id] = websocket
        self.user_connections[user_id] = connection_id
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str, user_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user"""
        connection_id = self.user_connections.get(user_id)
        if connection_id and connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_json(message)
    
    async def send_mood_update(self, user_id: str, mood: dict):
        """Send mood update to user"""
        await self.send_personal_message({
            "type": "mood",
            "timestamp": datetime.now().isoformat(),
            "payload": mood
        }, user_id)
    
    async def send_response_chunk(self, user_id: str, chunk: str):
        """Send response chunk for streaming"""
        await self.send_personal_message({
            "type": "chunk",
            "timestamp": datetime.now().isoformat(),
            "payload": chunk
        }, user_id)
    
    async def send_token_update(self, user_id: str, tokens_remaining: int):
        """Send token usage update"""
        await self.send_personal_message({
            "type": "tokens",
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "remaining": tokens_remaining,
                "percentage": (tokens_remaining / 1000) * 100
            }
        }, user_id)
    
    async def send_thinking_indicator(self, user_id: str, message: str):
        """Send thinking/processing indicator"""
        await self.send_personal_message({
            "type": "thinking",
            "timestamp": datetime.now().isoformat(),
            "payload": message
        }, user_id)


manager = ConnectionManager()


async def get_current_user_from_websocket(websocket: WebSocket, token: Optional[str] = None):
    """Extract user from WebSocket connection"""
    if not token:
        # Try to get from query params
        token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=1008)
        raise HTTPException(status_code=401, detail="No token provided")
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        user_id = payload.get("sub")
        email = payload.get("email")
        
        if not user_id:
            await websocket.close(code=1008)
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        user = await db.fetchrow("""
            SELECT id, email, username FROM users WHERE id = $1::uuid
        """, user_id)
        
        if not user:
            await websocket.close(code=1008)
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    
    except JWTError:
        await websocket.close(code=1008)
        raise HTTPException(status_code=401, detail="Invalid token")
