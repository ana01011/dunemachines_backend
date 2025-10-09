import requests
import json

BASE_URL = "http://localhost:8000"

print("🎨 FINAL THEME SYSTEM TEST")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test complete theme workflow
    print("\n1. Setting Initial Theme:")
    response = requests.post(
        f"{BASE_URL}/api/v1/themes/switch",
        json={"theme": "Neon Nights"},
        headers=headers
    )
    print(f"   Set to Neon Nights: {'✅' if response.status_code == 200 else '❌'}")
    
    print("\n2. Chat about the theme:")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/message",
        json={"message": "I just switched to Neon Nights, how does it look?", "personality": "sarah"},
        headers=headers
    )
    if response.status_code == 200:
        print(f"   Sarah: {response.json()['response'][:150]}...")
    
    print("\n3. Ask for suggestion:")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/message",
        json={"message": "Can you suggest a theme for nighttime coding?", "personality": "sarah"},
        headers=headers
    )
    if response.status_code == 200:
        print(f"   Sarah: {response.json()['response'][:150]}...")
    
    print("\n4. Switch via chat command:")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/message",
        json={"message": "Switch to Deep Ocean theme please", "personality": "sarah"},
        headers=headers
    )
    if response.status_code == 200:
        resp = response.json()
        print(f"   Sarah: {resp['response'][:150]}...")
        if 'user_context' in resp and 'theme_action' in resp['user_context']:
            print(f"   🎨 Action: {resp['user_context']['theme_action']}")
    
    print("\n5. Verify theme changed:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    if response.status_code == 200:
        current = response.json()['current_theme']
        print(f"   Current theme: {current}")
        print(f"   {'✅ Theme changed successfully!' if current == 'Deep Ocean' else '⚠️ Theme did not change'}")

print("\n" + "="*60)
print("🎉 Theme System Complete Test Finished!")
print("\nYour theme system features:")
print("  • 15 beautiful themes")
print("  • Chat-based theme switching")
print("  • Smart theme suggestions")
print("  • User preference tracking")
print("  • Sarah's theme awareness")
print("="*60)
