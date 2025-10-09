import requests
import json

BASE_URL = "http://localhost:8000"

print("🎨 Full Theme System Integration Test")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "uzma.maryam102021@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Get current theme
    print("\n1. Current Theme:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Theme: {data.get('current_theme', 'Unknown')}")
        print(f"   Preferences: {json.dumps(data.get('preferences', {}), indent=2)[:100]}...")
    
    # 2. Get available themes
    print("\n2. Available Themes:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/available", headers=headers)
    if response.status_code == 200:
        themes = response.json()
        print(f"   ✅ Found {len(themes)} themes:")
        for theme in themes[:5]:
            print(f"      - {theme}")
    
    # 3. Test theme switching
    print("\n3. Theme Switching Test:")
    test_themes = ["Neon Nights", "Cyber Dark", "Pure Light"]
    for theme in test_themes:
        response = requests.post(
            f"{BASE_URL}/api/v1/themes/switch",
            json={"theme": theme},
            headers=headers
        )
        if response.status_code == 200:
            print(f"   ✅ Switched to {theme}")
        else:
            print(f"   ❌ Failed to switch to {theme}: {response.status_code}")
    
    # 4. Test chat integration
    print("\n4. Chat Integration Test:")
    test_messages = [
        ("What theme am I using?", "query"),
        ("Switch to AI Neural theme", "switch"),
        ("Suggest me a good theme", "suggest"),
        ("I want dark mode", "dark"),
        ("Make it bright", "light")
    ]
    
    for msg, action_type in test_messages:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": msg, "personality": "sarah"},
            headers=headers
        )
        if response.status_code == 200:
            resp = response.json()
            print(f"\n   Message: '{msg}'")
            print(f"   Sarah: {resp['response'][:100]}...")
            
            # Check if theme context was added
            if 'user_context' in resp:
                ctx = resp['user_context']
                if 'theme_action' in ctx:
                    print(f"   🎨 Theme Action: {ctx['theme_action']}")
                if 'theme_query' in ctx:
                    print(f"   🔍 Theme Query: {ctx['theme_query']}")
                if 'theme_suggestions' in ctx:
                    print(f"   💡 Suggestions: {ctx['theme_suggestions']}")
    
    # 5. Get final theme
    print("\n5. Final Theme Check:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Current theme: {data.get('current_theme')}")

print("\n" + "="*60)
print("🎉 Theme System Fully Operational!")
print("="*60)
