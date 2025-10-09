print("""
To integrate Sarah's memory into the chat router, add this to 
/root/openhermes_backend/app/api/v1/routers/chat_router.py:

1. After getting the agent, pass conversation history:

# Around line where agent generates response, change from:
response_text = await agent.generate_response(
    message=message.content,
    user_id=user_id,
    relationship_score=relationship_score
)

# To:
response_text = await agent.generate_response(
    message=message.content,
    user_id=user_id,
    conversation_history=context,  # Add this
    relationship_score=relationship_score,
    name=user_profile.get('name'),
    age=user_profile.get('age'),
    location=user_profile.get('location'),
    occupation=user_profile.get('occupation'),
    facts=user_facts
)
""")
