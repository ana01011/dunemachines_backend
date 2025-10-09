# Read main.py
with open('/root/openhermes_backend/main.py', 'r') as f:
    lines = f.readlines()

# Find where routers are imported (around line 49-51)
import_index = None
for i, line in enumerate(lines):
    if 'from app.api.v1.routers import user_router' in line:
        import_index = i
        break

if import_index:
    # Add theme router import after user_router
    lines.insert(import_index + 1, 'from app.api.v1.routers import theme_router\n')
    print("✅ Added theme_router import")

# Find where routers are included (around line 228-230)
include_index = None
for i, line in enumerate(lines):
    if 'app.include_router(user_router.router' in line:
        include_index = i
        break

if include_index:
    # Add theme router inclusion after user_router
    lines.insert(include_index + 1, 'app.include_router(theme_router.router, prefix="/api/v1", tags=["Themes"])\n')
    print("✅ Added theme router registration")

# Write back
with open('/root/openhermes_backend/main.py', 'w') as f:
    f.writelines(lines)

print("✅ Theme router integrated into main.py")
