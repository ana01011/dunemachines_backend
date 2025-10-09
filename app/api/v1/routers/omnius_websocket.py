"""WebSocket router for Omnius streaming"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from app.agents.omnius_collaborative import omnius_neurochemical
import json
import asyncio

try:
    import jwt
except ImportError:
    from jose import jwt  # Use jose if PyJWT is not available

from app.core.config import settings

router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        # Get JWT token from first message
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)

        if "token" not in auth_data:
            await websocket.send_json({"error": "No token provided"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Verify JWT token - use lowercase attribute names
        try:
            payload = jwt.decode(
                auth_data["token"],
                settings.jwt_secret,  # Changed from JWT_SECRET to jwt_secret
                algorithms=[settings.jwt_algorithm]  # Changed from JWT_ALGORITHM to jwt_algorithm
            )
            user_id = payload.get("sub")

            if not user_id:
                await websocket.send_json({"error": "Invalid token"})
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

        except jwt.ExpiredSignatureError:
            await websocket.send_json({"error": "Token expired"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except jwt.InvalidTokenError as e:
            await websocket.send_json({"error": f"Invalid token: {str(e)}"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except Exception as e:
            await websocket.send_json({"error": f"Authentication failed: {str(e)}"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.send_json({"status": "authenticated", "user_id": user_id})
        print(f"✅ WebSocket authenticated for user: {user_id}")

        # Now handle chat messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            print(f"📥 Received message: {message_data.get('message', '')[:50]}...")

            # Stream response
            try:
                async for chunk in omnius_neurochemical.think_stream(
                    message=message_data.get("message", ""),
                    user_id=user_id,
                    temperature=message_data.get("temperature", 0.7),
                    max_tokens=message_data.get("max_tokens", 2000)
                ):
                    await websocket.send_json(chunk)
            except Exception as e:
                print(f"❌ Error during streaming: {e}")
                await websocket.send_json({"error": f"Streaming error: {str(e)}"})
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            pass
