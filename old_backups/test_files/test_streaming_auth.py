import asyncio
import websockets
import json
import requests
from datetime import datetime

async def test_stream():
    # First, get a fresh token by logging in
    login_url = "http://localhost:8000/api/v1/auth/login"
    login_data = {
        "email": "anaa.ahmad01@gmail.com",
        "password": "Xhash@1234"
    }
    
    print("🔐 Getting fresh token...")
    response = requests.post(login_url, json=login_data)
    
    if response.status_code != 200:
        print(f"❌ Failed to login: {response.text}")
        return
        
    token = response.json().get("access_token")
    print(f"✅ Got token: {token[:20]}...")
    
    # Try the correct WebSocket path with the prefix
    uri = "ws://localhost:8000/api/v1/ws/stream"
    
    print(f"🔌 Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Send authentication
            await websocket.send(json.dumps({"token": token}))
            
            # Get auth response
            auth_response = await websocket.recv()
            auth_data = json.loads(auth_response)
            print(f"🔑 Auth: {auth_data}")
            
            if auth_data.get("error"):
                print(f"❌ Authentication failed: {auth_data['error']}")
                return
                
            # Send a message
            message = {
                "message": "Write a Python function to calculate fibonacci numbers with optimization",
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            print(f"📤 Sending: {message['message']}")
            await websocket.send(json.dumps(message))
            
            # Receive streaming response
            print("\n📥 Response:\n" + "="*50)
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data.get("type") == "status":
                        print(f"📊 Status: {data['message']}")
                    elif data.get("type") == "content":
                        print(data['content'], end='', flush=True)
                    elif data.get("error"):
                        print(f"\n❌ Error: {data['error']}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\n" + "="*50)
                    print("✅ Stream complete")
                    break
                    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nTrying alternative paths...")
        
        # Try alternative paths
        alternative_paths = [
            "ws://localhost:8000/ws/stream",
            "ws://localhost:8000/api/v1/omnius/ws/stream"
        ]
        
        for alt_uri in alternative_paths:
            print(f"  Trying: {alt_uri}")
            try:
                async with websockets.connect(alt_uri, timeout=2) as ws:
                    print(f"  ✅ Connected to {alt_uri}")
                    return
            except:
                print(f"  ❌ Failed")
                continue

if __name__ == "__main__":
    print("🚀 Testing WebSocket Streaming with Fresh Authentication")
    print("="*50)
    asyncio.run(test_stream())
