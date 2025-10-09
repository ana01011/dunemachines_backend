# Read the file
with open('/root/openhermes_backend/app/services/auth/auth_service.py', 'r') as f:
    lines = f.readlines()

# Remove the duplicate line at line 158 (index 157)
# This is the floating message line that shouldn't be there
if 'message": "Registration successful! Please check your email for verification code."' in lines[157]:
    del lines[157]
    print("Removed duplicate message line at line 158")

# Find and remove the first resend_otp method (around line 141-158)
# Keep only the second one (around line 224)
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'async def resend_otp(self, email: str)' in line:
        if start_idx is None:
            start_idx = i
            # Find the end of this method (next method or class end)
            for j in range(i+1, len(lines)):
                if 'async def ' in lines[j] or 'class ' in lines[j]:
                    end_idx = j
                    break
            if start_idx and end_idx:
                # Remove the first resend_otp method
                del lines[start_idx:end_idx]
                print(f"Removed duplicate resend_otp method from line {start_idx+1} to {end_idx+1}")
                break

# Write the fixed file
with open('/root/openhermes_backend/app/services/auth/auth_service.py', 'w') as f:
    f.writelines(lines)

print("File fixed successfully!")
