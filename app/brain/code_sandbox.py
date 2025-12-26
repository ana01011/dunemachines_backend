"""
Secure Code Sandbox - Safe execution environment for code testing
"""
import subprocess
import tempfile
import os
import sys
from typing import Tuple, Optional
from dataclasses import dataclass
 
 
@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str
    return_code: int
 
 
class CodeSandbox:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.allowed_imports = [
            'math', 'random', 'datetime', 'json', 're', 
            'collections', 'itertools', 'functools', 'string',
            'statistics', 'decimal', 'fractions'
        ]
    
    def extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks"""
        import re
        
        # Try to find ```python blocks
        matches = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try generic ``` blocks
        matches = re.findall(r'```\n?(.*?)```', text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Return as-is if no blocks
        return text.strip()
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Basic security validation"""
        dangerous = [
            'import os', 'import sys', 'import subprocess',
            '__import__', 'eval(', 'exec(', 'open(',
            'file(', 'input(', 'raw_input(',
            'os.system', 'subprocess', 'shutil',
            'import socket', 'import requests',
        ]
        
        for d in dangerous:
            if d in code:
                return False, f"Dangerous operation detected: {d}"
        
        return True, "OK"
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in isolated subprocess"""
        
        # Extract code from markdown if needed
        clean_code = self.extract_code(code)
        
        # Validate
        valid, msg = self.validate_code(clean_code)
        if not valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security validation failed: {msg}",
                return_code=-1
            )
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(clean_code)
            temp_path = f.name
        
        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout}s",
                return_code=-2
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-3
            )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_and_fix(self, code: str, max_attempts: int = 2) -> Tuple[str, ExecutionResult]:
        """Test code and return result"""
        result = self.execute(code)
        return code, result
 
 
code_sandbox = CodeSandbox()
