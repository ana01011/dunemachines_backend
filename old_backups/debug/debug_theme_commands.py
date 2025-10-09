import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Debugging Theme Command Processing")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test each command and see full response
    test_commands = [
        "Switch to Pure Light theme",
        "Change to Simple Dark",
        "I want dark mode",
        "Use Tech Blue theme"
    ]
    
    for command in test_commands:
        print(f"\n📝 Command: '{command}'")
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": command, "personality": "sarah"},
            headers=headers
        )
        
        if response.status_code == 200:
            resp = response.json()
            
            # Check user_context for theme detection
            if 'user_context' in resp:
                ctx = resp['user_context']
                print(f"   Context keys: {list(ctx.keys())}")
                
                if 'theme_action' in ctx:
                    print(f"   ✅ Theme action: {ctx['theme_action']}")
                else:
                    print(f"   ❌ No theme_action detected")
                
                if 'theme_query' in ctx:
                    print(f"   Theme query: {ctx['theme_query']}")
                
                if 'theme_suggestions' in ctx:
                    print(f"   Suggestions: {ctx['theme_suggestions']}")
            else:
                print(f"   ❌ No user_context in response")
            
            # Check actual theme
            response2 = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
            if response2.status_code == 200:
                current = response2.json()['current_theme']
                print(f"   Current theme: {current}")
        else:
            print(f"   Error: {response.status_code}")

print("\n" + "="*60)
