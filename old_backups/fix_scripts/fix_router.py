# Read the router file
with open('/root/openhermes_backend/app/api/v1/routers/auth_router.py', 'r') as f:
    content = f.read()

# Fix the message handling in register endpoint
old_pattern = '"message": result["message"],'
new_pattern = '"message": result.get("message", "Registration successful. Please check your email for verification code."),'

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("Fixed message handling in register endpoint")
else:
    print("Message handling already correct or not found")

# Write back
with open('/root/openhermes_backend/app/api/v1/routers/auth_router.py', 'w') as f:
    f.write(content)

print("Router check complete!")
