#!/bin/bash

echo "🧹 OMNIUS System Cleanup Script"
echo "================================"

# Create backup directory for safety
mkdir -p old_backups
echo "📦 Created backup directory: old_backups/"

# 1. Move all test files to a temporary directory
echo "🗑️  Moving test files..."
mkdir -p old_backups/test_files
mv test_*.py old_backups/test_files/ 2>/dev/null
mv test_*.sh old_backups/test_files/ 2>/dev/null
mv simple_omnius_test.py old_backups/test_files/ 2>/dev/null
mv final_*test*.py old_backups/test_files/ 2>/dev/null
mv verify_*.py old_backups/test_files/ 2>/dev/null
echo "  ✓ Test files moved"

# 2. Move all fix scripts (they're no longer needed)
echo "🔧 Moving old fix scripts..."
mkdir -p old_backups/fix_scripts
mv fix_*.py old_backups/fix_scripts/ 2>/dev/null
mv add_*.py old_backups/fix_scripts/ 2>/dev/null
mv update_*.py old_backups/fix_scripts/ 2>/dev/null
mv enhance_*.py old_backups/fix_scripts/ 2>/dev/null
mv ensure_*.py old_backups/fix_scripts/ 2>/dev/null
echo "  ✓ Fix scripts moved"

# 3. Move old/backup files
echo "📋 Moving backup files..."
mkdir -p old_backups/backups
mv *_old.py old_backups/backups/ 2>/dev/null
mv *_backup*.py old_backups/backups/ 2>/dev/null
mv auth_service_fixed.py old_backups/backups/ 2>/dev/null
mv sarah_enhanced.py old_backups/backups/ 2>/dev/null
echo "  ✓ Backup files moved"

# 4. Move debug files
echo "🐛 Moving debug files..."
mkdir -p old_backups/debug
mv debug_*.py old_backups/debug/ 2>/dev/null
mv check_*.py old_backups/debug/ 2>/dev/null
echo "  ✓ Debug files moved"

# 5. Clean up random/temp files
echo "🗑️  Cleaning temporary files..."
rm -f EOF 2>/dev/null
rm -f -H 2>/dev/null
rm -f -d 2>/dev/null
rm -f email: 2>/dev/null
rm -f username: 2>/dev/null
rm -f password: 2>/dev/null
rm -f gender: 2>/dev/null
rm -f "name:" 2>/dev/null
rm -f if 2>/dev/null
rm -f import 2>/dev/null
rm -f response.json 2>/dev/null
rm -f response.txt 2>/dev/null
rm -f login.json 2>/dev/null
rm -f token.txt 2>/dev/null
rm -f nohup.out 2>/dev/null
rm -f server.log 2>/dev/null
rm -f backend.log 2>/dev/null
echo "  ✓ Temp files removed"

# 6. Move relationship system files (old versions)
echo "💔 Moving old relationship files..."
mkdir -p old_backups/relationship
mv relationship_system*.py old_backups/relationship/ 2>/dev/null
mv relationship_profiles.json old_backups/relationship/ 2>/dev/null
echo "  ✓ Old relationship files moved"

# 7. Move theme related files (old versions)
echo "🎨 Moving old theme files..."
mkdir -p old_backups/themes
mv theme_service_enhanced.py old_backups/themes/ 2>/dev/null
mv complete_theme_integration.py old_backups/themes/ 2>/dev/null
mv create_theme_tables.sql old_backups/themes/ 2>/dev/null
echo "  ✓ Old theme files moved"

# 8. Clean pycache
echo "🗑️  Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  ✓ Python cache cleaned"

# 9. Move database reset scripts
echo "🗄️  Moving database scripts..."
mkdir -p old_backups/db_scripts
mv reset_database*.py old_backups/db_scripts/ 2>/dev/null
mv safe_delete_users.py old_backups/db_scripts/ 2>/dev/null
echo "  ✓ Database scripts moved"

# 10. Move system test scripts
echo "🧪 Moving system test scripts..."
mv system_test.sh old_backups/test_files/ 2>/dev/null
mv setup_agents.sh old_backups/ 2>/dev/null
mv remove_gud_references.py old_backups/ 2>/dev/null
echo "  ✓ System test scripts moved"

# 11. Clean up neurochemistry enable scripts
echo "🧬 Moving neurochemistry scripts..."
mv enable_*.py old_backups/ 2>/dev/null
echo "  ✓ Neurochemistry scripts moved"

# Show what remains
echo ""
echo "📁 Essential files that remain:"
echo "================================"
ls -la | grep -E "^-" | grep -v ".log$" | awk '{print "  ✓", $9}'

echo ""
echo "📁 Essential directories:"
echo "========================"
ls -d */ | grep -v old_backups | awk '{print "  ✓", $1}'

echo ""
echo "💾 Space saved:"
du -sh old_backups/ 2>/dev/null

echo ""
echo "✅ Cleanup complete! All junk moved to old_backups/"
echo "   You can safely delete old_backups/ when ready with: rm -rf old_backups/"
