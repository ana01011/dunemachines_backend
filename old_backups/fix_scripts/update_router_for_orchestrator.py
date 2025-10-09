# Update omnius_router.py to use the new orchestrator

print("""
In app/api/v1/routers/omnius_router.py, update the chat endpoint:

1. Change:
    response_text = await omnius.think(request.message, context)
    
2. To:
    response_text = await omnius.think(request.message, context)
    
The new orchestrator returns just the text, but logs everything internally.
""")
