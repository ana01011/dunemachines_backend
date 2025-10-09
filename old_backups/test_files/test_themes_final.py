import requests
import json

BASE_URL = "http://localhost:8000"

# Login
print("1. Testing login...")
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   ✅ Login successful")
    
    # Test theme endpoints
    print("\n2. Testing theme endpoints:")
    
    # Get current theme
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    print(f"   Current theme: {response.status_code}")
    
    # Get available themes
    response = requests.get(f"{BASE_URL}/api/v1/themes/available", headers=headers)
    print(f"   Available themes: {response.status_code}")
    
    # Switch theme
    response = requests.post(
        f"{BASE_URL}/api/v1/themes/switch",
        json={"theme": "Neon Nights"},
        headers=headers
    )
    print(f"   Switch theme: {response.status_code}")
    
    # Test theme command in chat
    print("\n3. Testing theme commands in chat:")
    
    test_messages = [
        "What theme am I using?",
        "Switch to cyber dark theme",
        "Show me available themes"
    ]
    
    for msg in test_messages:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": msg, "personality": "sarah"},
            headers=headers
        )
        if response.status_code == 200:
            resp = response.json()
            print(f"   '{msg}' -> Sarah: {resp['response'][:80]}...")
            if 'user_context' in resp:
                ctx = resp['user_context']
                if 'theme_action' in ctx:
                    print(f"      Theme action: {ctx['theme_action']}")
                if 'theme_query' in ctx:
                    print(f"      Theme query: {ctx['theme_query']}")
else:
    print(f"   ❌ Login failed: {response.status_code}")
