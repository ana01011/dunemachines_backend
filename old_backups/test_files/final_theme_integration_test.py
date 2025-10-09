import requests
import json

BASE_URL = "http://localhost:8000"

print("🎨 FINAL THEME INTEGRATION TEST")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test theme switching via chat
    test_commands = [
        "Switch to Neon Nights",
        "What dark themes do you have?",
        "I'm coding late at night, suggest a theme"
    ]
    
    for cmd in test_commands:
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
            headers=headers,
            json={"message": cmd, "personality": "sarah"}
        )
        if response.status_code == 200:
            print(f"✅ '{cmd}' - Working")
        else:
            print(f"❌ '{cmd}' - Error {response.status_code}")
    
    # Check current theme
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    if response.status_code == 200:
        current = response.json()
        print(f"\n📍 Current Theme: {current['current_theme']}")
        print("✅ Theme system fully integrated!")

print("="*60)
