#!/usr/bin/env python3
"""
Test neurochemistry with your existing WebSocket endpoint
"""

import asyncio
import aiohttp
import websockets
import json
import sys
sys.path.append('/workspace')

from app.neurochemistry.core.state_v2 import NeurochemicalState
from app.neurochemistry.core.mood_emergence_v2 import MoodEmergence

EMAIL = "anaa.ahmad01@gmail.com"
PASSWORD = "Xhash@1234"

async def test_existing_websocket():
    """Test with existing /ws/stream endpoint"""
    print("=" * 80)
    print("🌊 TESTING WITH EXISTING WEBSOCKET")
    print("=" * 80)
    
    # Get token
    print("\n🔐 Getting token...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"email": EMAIL, "password": PASSWORD}
        ) as response:
            if response.status == 200:
                data = await response.json()
                token = data.get("access_token")
                print("✅ Got token")
            else:
                print("❌ Login failed")
                return
    
    # Connect to existing WebSocket
    uri = "ws://localhost:8000/api/v1/ws/stream"  # Your existing endpoint
    
    # Create a local neurochemical state to simulate
    state = NeurochemicalState()
    
    async with websockets.connect(uri) as websocket:
        # Authenticate
        await websocket.send(json.dumps({"token": token}))
        
        # Get auth response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✅ Connected: {data}")
        
        # Test messages with neurochemical simulation
        test_cases = [
            ("Simple", "Hello, how are you?", 
             {"complexity": 0.2, "urgency": 0.1}),
            
            ("Urgent", "URGENT! Critical bug in production!", 
             {"complexity": 0.5, "urgency": 0.9}),
            
            ("Complex", "Implement a distributed consensus algorithm",
             {"complexity": 0.9, "urgency": 0.3}),
            
            ("Emotional", "I'm so frustrated with this code!",
             {"complexity": 0.3, "urgency": 0.4, "emotional_content": 0.8})
        ]
        
        for name, message, params in test_cases:
            print(f"\n📤 {name}: {message}")
            
            # Simulate neurochemical response locally
            event = Event(
                type=name.lower(),
                complexity=params.get("complexity", 0.5),
                urgency=params.get("urgency", 0.5),
                emotional_content=params.get("emotional_content", 0.5)
            )
            state.apply_dynamics(0.1, event)
            
            # Get mood
            mood = MoodEmergence.describe_emergent_state(state)
            prompt_injection = MoodEmergence.create_natural_prompt(state)
            
            print(f"   Neurochemical State: {prompt_injection}")
            print(f"   Mood: {mood}")
            print(f"   D={state.dopamine:.0f} C={state.cortisol:.0f} A={state.adrenaline:.0f}")
            
            # Send message with neurochemical context
            enhanced_message = f"{prompt_injection} {message}"
            
            await websocket.send(json.dumps({
                "message": enhanced_message,
                "temperature": 0.7,
                "max_tokens": 200
            }))
            
            # Get response
            response_text = ""
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "content":
                        response_text += data.get("content", "")
                    elif data.get("type") == "status":
                        pass  # Status update
                    else:
                        break
                        
                except asyncio.TimeoutError:
                    break
            
            if response_text:
                print(f"   AI Response: {response_text[:100]}...")
            
            # Simulate task completion and dopamine response
            quality = 0.7 if "error" not in response_text.lower() else 0.3
            state.complete_task(quality)
            print(f"   Post-response Dopamine: {state.dopamine:.0f}")

# Add Event import
from app.neurochemistry.core.state_v2 import Event

if __name__ == "__main__":
    asyncio.run(test_existing_websocket())
