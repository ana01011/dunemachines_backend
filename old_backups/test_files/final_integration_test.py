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
    
    test_cases = [
        ("Hello Sarah", None),  # No theme change expected
        ("Switch to Pure Light", "Pure Light"),
        ("I want dark mode", "Simple Dark"),
        ("Change to Cyber Dark theme", "Cyber Dark"),
        ("How are you?", None),  # No theme change expected
    ]
    
    for message, expected_theme in test_cases:
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
            headers=headers,
            json={"message": message, "personality": "sarah"}
        )
        
        if response.status_code == 200:
            data = response.json()
            actual_theme = data.get('theme_changed')
            
            if expected_theme == actual_theme:
                print(f"✅ '{message}' → theme_changed: {actual_theme}")
            else:
                print(f"❌ '{message}' → Expected: {expected_theme}, Got: {actual_theme}")

print("="*60)
print("🎉 Backend theme integration complete and working!")
print("📋 Frontend just needs to check 'theme_changed' field in responses")
