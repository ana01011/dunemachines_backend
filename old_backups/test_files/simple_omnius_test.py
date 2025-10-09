import requests
import json

# Login
login_response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={"email": "anaa.ahmad01@gmail.com", "password": "Xhash@1234"}
)
token = login_response.json()["access_token"]

# Headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test messages
messages = [
    "Hello Omnius, what are you?",
    "Explain your distributed consciousness",
    "Write a Python function to sort a list"
]

for msg in messages:
    print(f"\n{'='*50}")
    print(f"User: {msg}")
    print('='*50)
    
    response = requests.post(
        "http://localhost:8000/api/v1/omnius/chat",
        headers=headers,
        json={
            "message": msg,
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Omnius: {data['response']}")
        print(f"\nConsciousness used: {data.get('consciousness_used', [])}")
        print(f"Processing time: {data.get('processing_time', 0):.2f}s")
    else:
        print(f"Error: {response.json()}")
