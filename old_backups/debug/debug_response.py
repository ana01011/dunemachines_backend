import requests
import json

BASE_URL = "http://localhost:8000"

print("🔍 Debugging API Response")
print("="*60)

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test theme change command
    print("\nSending theme change command...")
    response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
        headers=headers,
        json={"message": "Switch to Neon Nights theme", "personality": "sarah"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers.get('content-type')}")
    
    try:
        data = response.json()
        print(f"Response Keys: {list(data.keys())}")
        print(f"Full Response: {json.dumps(data, indent=2)[:500]}")
    except:
        print(f"Raw Response: {response.text[:500]}")

