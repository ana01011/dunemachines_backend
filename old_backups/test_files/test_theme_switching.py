import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("🔍 Testing if Sarah ACTUALLY switches themes")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Check initial theme
    print("\n1. Initial Theme:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    initial_theme = response.json()['current_theme'] if response.status_code == 200 else "Unknown"
    print(f"   Current: {initial_theme}")
    
    # 2. Send chat command to switch theme
    print("\n2. Sending chat command to switch to 'Neon Nights':")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/message",
        json={"message": "Switch to Neon Nights theme", "personality": "sarah"},
        headers=headers
    )
    
    if response.status_code == 200:
        resp = response.json()
        print(f"   Sarah's response: {resp['response'][:100]}...")
        
        # Check if theme action was detected
        if 'user_context' in resp:
            ctx = resp['user_context']
            if 'theme_action' in ctx:
                print(f"   ✅ Theme action detected: {ctx['theme_action']}")
            else:
                print(f"   ❌ No theme_action in context")
            if 'theme_query' in ctx:
                print(f"   Theme query: {ctx['theme_query']}")
        else:
            print(f"   ❌ No user_context in response")
    
    # 3. Wait a moment then check if theme actually changed
    time.sleep(1)
    print("\n3. Checking if theme actually changed:")
    response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
    new_theme = response.json()['current_theme'] if response.status_code == 200 else "Unknown"
    print(f"   Current theme now: {new_theme}")
    
    if new_theme == "Neon Nights" and initial_theme != "Neon Nights":
        print(f"   ✅ SUCCESS! Theme changed from '{initial_theme}' to '{new_theme}'")
    elif new_theme == "Neon Nights" and initial_theme == "Neon Nights":
        print(f"   ⚠️  Theme was already Neon Nights, trying different theme...")
        
        # Try switching to Cyber Dark
        print("\n4. Trying to switch to 'Cyber Dark':")
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": "Change theme to Cyber Dark", "personality": "sarah"},
            headers=headers
        )
        
        time.sleep(1)
        response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
        final_theme = response.json()['current_theme'] if response.status_code == 200 else "Unknown"
        print(f"   Theme is now: {final_theme}")
        
        if final_theme == "Cyber Dark":
            print(f"   ✅ SUCCESS! Theme changed to Cyber Dark")
        else:
            print(f"   ❌ FAILED! Theme did not change")
    else:
        print(f"   ❌ FAILED! Theme did not change from '{initial_theme}'")
    
    # 4. Test multiple theme commands
    print("\n5. Testing various theme commands:")
    test_commands = [
        ("Switch to Pure Light theme", "Pure Light"),
        ("I want dark mode", "Simple Dark"),
        ("Set theme to AI Neural", "AI Neural")
    ]
    
    for command, expected_theme in test_commands:
        print(f"\n   Command: '{command}'")
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": command, "personality": "sarah"},
            headers=headers
        )
        
        time.sleep(0.5)
        response = requests.get(f"{BASE_URL}/api/v1/themes/current", headers=headers)
        current = response.json()['current_theme'] if response.status_code == 200 else "Unknown"
        
        if current == expected_theme:
            print(f"   ✅ Changed to {current}")
        else:
            print(f"   ❌ Expected {expected_theme}, but theme is {current}")

print("\n" + "="*60)
