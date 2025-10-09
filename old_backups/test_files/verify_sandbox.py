"""
Quick script to verify sandbox execution is working
"""
import subprocess
import tempfile
import os

def test_sandbox():
    """Test if sandbox execution works"""
    
    test_code = """
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Test
print(f"fibonacci(10) = {fibonacci(10)}")
print(f"fibonacci(20) = {fibonacci(20)}")
# This should print 55 and 6765
"""
    
    print("Testing sandbox execution...")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Execute
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        os.unlink(temp_file)
        
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        print(f"Errors: {result.stderr}")
        
        if result.returncode == 0 and "55" in result.stdout and "6765" in result.stdout:
            print("✅ Sandbox execution VERIFIED - working correctly!")
            return True
        else:
            print("❌ Sandbox execution FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False

if __name__ == "__main__":
    test_sandbox()
