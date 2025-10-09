import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Testing Backend Error")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "uzma.maryam102021@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test simple message
    print("\nTesting simple message...")
    response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
        headers=headers,
        json={
            "message": "hi",
            "personality": "sarah",
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
    else:
        data = response.json()
        print(f"Success: {data['response'][:100]}...")

