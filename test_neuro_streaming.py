#!/usr/bin/env python3
"""
Test neurochemical streaming system
"""

import asyncio
import websockets
import json
import time

async def test_streaming():
    """Test the neurochemical WebSocket streaming"""
    
    # Get auth token (you'll need to replace with actual)
    token = "YOUR_JWT_TOKEN"
    
    uri = "ws://localhost:8000/api/v1/ws/omnius"
    
    async with websockets.connect(uri) as websocket:
        # Authenticate
        await websocket.send(json.dumps({"token": token}))
        
        # Get initial response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Connected: {data.get('type')}")
        print(f"Initial mood: {data.get('initial_state', {}).get('mood')}")
        
        # Start receiving waves in background
        async def receive_waves():
            while True:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    if data.get("type") == "neuro_wave":
                        wave = data["data"]
                        print(f"\n🌊 Wave: {wave['mood']} | "
                              f"D:{wave['levels']['dopamine']:.0f} "
                              f"C:{wave['levels']['cortisol']:.0f} "
                              f"A:{wave['levels']['adrenaline']:.0f}")
                        
                        if wave['triggers']:
                            print(f"   Triggers: {wave['triggers']}")
                            
                except Exception as e:
                    break
        
        # Start wave receiver
        wave_task = asyncio.create_task(receive_waves())
        
        # Send test messages
        test_messages = [
            "Hello, how are you?",
            "URGENT! I need help with a critical bug!",
            "Can you write a complex sorting algorithm?",
            "I'm feeling really frustrated with this code",
            "Thank you so much! That worked perfectly!"
        ]
        
        for msg in test_messages:
            print(f"\n📤 Sending: {msg}")
            await websocket.send(json.dumps({
                "type": "message",
                "content": msg
            }))
            
            # Wait to see neurochemical response
            await asyncio.sleep(3)
        
        # Cancel wave receiver
        wave_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_streaming())
