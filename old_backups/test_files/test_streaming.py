import asyncio
import websockets
import json

async def test_stream():
    uri = "ws://localhost:8000/api/v1/omnius/ws/stream"
    
    async with websockets.connect(uri) as websocket:
        # Send a message
        await websocket.send(json.dumps({
            "message": "Create a function to calculate factorial with optimization",
            "user_id": "test-user"
        }))
        
        # Receive streaming response
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'status':
                print(f"[STATUS] {data['content']}")
            elif data['type'] == 'content':
                print(data['chunk'], end='', flush=True)
            elif data['type'] == 'complete':
                print(f"\n[COMPLETE] {data['content']}")
                break

asyncio.run(test_stream())
