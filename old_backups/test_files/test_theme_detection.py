import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Testing Theme Detection")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    test_messages = [
        "darker background",
        "dark mode",
        "Switch to Simple Dark",
        "Change to dark theme",
        "I want dark mode",
        "Use Simple Dark theme"
    ]
    
    for msg in test_messages:
        print(f"\n📝 Testing: '{msg}'")
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
            headers=headers,
            json={"message": msg, "personality": "sarah"}
        )
        
        if response.status_code == 200:
            data = response.json()
            theme = data.get('theme_changed')
            if theme:
                print(f"   ✅ Theme changed to: {theme}")
            else:
                print(f"   ❌ No theme change detected")
                print(f"   Sarah says: {data['response'][:100]}...")

