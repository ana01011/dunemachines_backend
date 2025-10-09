"""
Script to integrate neurochemistry into your existing server
"""

print("""
To integrate the neurochemistry system into your server, add these lines:

1. In main_neurochemical_fixed.py, add the import:
""")

print('''
# Add after other imports:
from app.api.v1.routers import omnius_neuro_websocket
''')

print("""
2. In main_neurochemical_fixed.py, add the router:
""")

print('''
# Add after other routers:
app.include_router(
    omnius_neuro_websocket.router, 
    prefix="/api/v1",
    tags=["Neurochemical WebSocket"]
)
print("✅ Neurochemical WebSocket router registered")
''')

print("""
3. Create the directory for streaming if it doesn't exist:
""")

print('''
mkdir -p app/neurochemistry/streaming
''')

print("""
4. Update your omnius_collaborative.py to accept neurochemical prompts:
""")

print('''
# In the think_stream method, add:
async def think_stream(self, message: str, user_id: str, 
                       temperature: float = 0.7, max_tokens: int = 2000,
                       neurochemical_prompt: str = None):
    
    # Inject neurochemical state into prompt if provided
    if neurochemical_prompt:
        message = f"{neurochemical_prompt} {message}"
''')
