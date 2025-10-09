import requests

BASE_URL = "http://localhost:8000"

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    print("🎨 Testing Smart Theme System")
    print("="*60)
    
    test_commands = [
        "Show me dark themes",
        "What themes are good for coding?",
        "I want something professional",
        "Show me colorful themes",
        "I need a theme for night time",
        "Something fun and playful",
        "What's available in light themes?",
        "I want to work, suggest a theme",
        "Show all available themes"
    ]
    
    for cmd in test_commands:
        print(f"\n📝 You: {cmd}")
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": cmd, "personality": "sarah"},
            headers=headers
        )
        
        if response.status_code == 200:
            resp = response.json()
            print(f"🤖 Sarah: {resp['response'][:150]}...")
            
            ctx = resp.get('user_context', {})
            if 'theme_category' in ctx:
                print(f"   📂 Category: {ctx['theme_category'][:2]}")
            if 'theme_suggestions' in ctx:
                print(f"   💡 Suggestions: {ctx['theme_suggestions']}")
            if 'all_themes' in ctx:
                print(f"   📋 Total themes: {len(ctx['all_themes'])}")

print("\n" + "="*60)
print("Smart theme system is ready!")
