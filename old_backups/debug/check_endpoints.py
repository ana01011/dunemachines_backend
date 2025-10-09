import requests

response = requests.get("http://localhost:8000/openapi.json")
if response.status_code == 200:
    api_spec = response.json()
    paths = api_spec.get("paths", {})
    
    print("All registered endpoints:")
    for path in paths:
        methods = list(paths[path].keys())
        print(f"  {path}: {methods}")
        
    # Check for WebSocket endpoints
    ws_paths = [p for p in paths if 'ws' in p.lower() or 'websocket' in p.lower()]
    if ws_paths:
        print("\nWebSocket endpoints found:")
        for path in ws_paths:
            print(f"  {path}")
else:
    print(f"Failed to get API spec: {response.status_code}")
