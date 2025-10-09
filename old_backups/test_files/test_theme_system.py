import requests
import json

BASE_URL = "http://localhost:8000"

# First login
login_data = {
    "email": "anaa.ahmad01@gmail.com",
    "password": "Xhash@1234"
}

print("Testing Theme System...")
print("="*50)

response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test theme endpoints
    endpoints = [
        ("GET", "/api/v1/theme/current"),
        ("GET", "/api/v1/theme/available"),
        ("GET", "/api/v1/theme/suggestions"),
        ("GET", "/api/v1/theme/history")
    ]
    
    for method, endpoint in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", headers=headers)
            print(f"\n{method} {endpoint}:")
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {json.dumps(response.json(), indent=2)[:200]}...")
            else:
                print(f"  Error: {response.text[:100]}")
        except Exception as e:
            print(f"  Failed: {e}")
else:
    print(f"Login failed: {response.status_code}")
