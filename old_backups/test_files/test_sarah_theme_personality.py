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
    
    print("Testing Sarah's Theme Personality")
    print("="*60)
    
    # Test Sarah's responses to theme-related messages
    test_cases = [
        "I love the Neon Nights theme!",
        "This theme is boring",
        "What do you think about the Cyber Dark theme?",
        "Should I use a dark or light theme?",
        "The colors are hurting my eyes"
    ]
    
    for msg in test_cases:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"message": msg, "personality": "sarah"},
            headers=headers
        )
        if response.status_code == 200:
            resp = response.json()
            print(f"\nYou: {msg}")
            print(f"Sarah: {resp['response']}")
            print("-"*40)
