import requests
import json

BASE_URL = "http://localhost:8000"

# Login
response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
    "email": "uzma.maryam102021@gmail.com",
    "password": "Xhash@1234"
})

if response.status_code == 200:
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    test_phrases = [
        "darker background",
        "i say darker background",
        "make it darker",
        "dark background please",
        "change to darker background"
    ]
    
    print("🧪 Testing 'darker' variations:")
    print("="*60)
    
    for phrase in test_phrases:
        response = requests.post(f"{BASE_URL}/api/v1/chat/message", 
            headers=headers,
            json={"message": phrase, "personality": "sarah"}
        )
        
        if response.status_code == 200:
            data = response.json()
            theme = data.get('theme_changed')
            if theme:
                print(f"✅ '{phrase}' → {theme}")
            else:
                print(f"❌ '{phrase}' → No change")

