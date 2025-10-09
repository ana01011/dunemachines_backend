import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Debugging Theme Switch")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    test_commands = [
        "switch to darker",
        "darker theme",
        "dark mode",
        "Simple Dark",
        "switch to Simple Dark"
    ]
    
    for cmd in test_commands:
        print(f"\n📝 Testing: '{cmd}'")
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
            headers=headers,
            json={"message": cmd, "personality": "sarah"}
        )
        
        if response.status_code == 200:
            data = response.json()
            theme = data.get('theme_changed')
            context = data.get('user_context', {})
            
            print(f"   theme_changed: {theme}")
            if 'theme_action' in context:
                print(f"   theme_action in context: {context['theme_action']}")
            
            # Check if theme was detected
            if not theme:
                print(f"   ❌ Theme NOT changed")
                print(f"   Sarah says: {data['response'][:80]}...")

