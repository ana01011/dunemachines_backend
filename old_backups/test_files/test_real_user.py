import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Testing with your user account")
print("="*60)

# Login with your account
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com", 
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test theme change
    print("\nSending: 'switch to darker'")
    response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
        headers=headers,
        json={
            "message": "switch to darker",
            "personality": "sarah",
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: 200 OK")
        print(f"✅ theme_changed: {data.get('theme_changed')}")
        print(f"✅ Response: {data['response'][:100]}...")
        print("\nFull response structure:")
        print(json.dumps({k: v for k, v in data.items() if k != 'user_context'}, indent=2))

