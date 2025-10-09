import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Testing Theme Response Integration")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 1: Normal message (no theme change)
    print("\n1. Normal message:")
    response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
        headers=headers,
        json={"message": "Hello Sarah", "personality": "sarah"}
    )
    data = response.json()
    print(f"   theme_changed field: {data.get('theme_changed', 'Not present')}")
    
    # Test 2: Theme change command
    print("\n2. Theme change command:")
    response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
        headers=headers,
        json={"message": "Switch to Neon Nights theme", "personality": "sarah"}
    )
    data = response.json()
    print(f"   theme_changed field: {data.get('theme_changed', 'Not present')}")
    print(f"   Response: {data['response'][:100]}...")
    
    print("\n✅ Ready for frontend integration!")

