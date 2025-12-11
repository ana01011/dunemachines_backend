"""
Code Area - Handles programming and code-related tasks
"""
from typing import Dict, List, Any, Optional
from app.brain.brain_area import BrainAreaBase, Tool, AreaOutput


class CodeArea(BrainAreaBase):
    """
    Code brain area - specializes in programming tasks.
    Tools: Sandbox, Linter, Debugger, Plotter, Formatter, Git
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(
            name="code",
            input_size=128,
            hidden_size=256,
            learning_rate=learning_rate,
            tool_threshold=0.5
        )
        # Set up connections to related areas
        self.connect_to("math", strength=0.4)
        self.connect_to("memory", strength=0.3)
    
    def _init_tools(self):
        """Initialize code-related tools"""
        self.tools = [
            Tool(
                name="sandbox",
                description="Execute code in isolated environment",
                category="execution"
            ),
            Tool(
                name="linter",
                description="Check code syntax and style",
                category="validation"
            ),
            Tool(
                name="debugger",
                description="Debug and trace code execution",
                category="debugging"
            ),
            Tool(
                name="plotter",
                description="Generate visualizations and plots",
                category="visualization"
            ),
            Tool(
                name="formatter",
                description="Format and clean code",
                category="formatting"
            ),
            Tool(
                name="git",
                description="Version control operations",
                category="versioning"
            )
        ]
    
    def execute_tools(self, active_tools: List[str], code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute the selected tools on the code.
        This is a placeholder - actual tool execution would be implemented here.
        """
        results = {}
        
        for tool_name in active_tools:
            if tool_name == "sandbox":
                results["sandbox"] = {"status": "ready", "message": "Code sandbox available"}
            elif tool_name == "linter":
                results["linter"] = {"status": "ready", "message": "Linter ready"}
            elif tool_name == "debugger":
                results["debugger"] = {"status": "ready", "message": "Debugger ready"}
            elif tool_name == "plotter":
                results["plotter"] = {"status": "ready", "message": "Plotter ready"}
            elif tool_name == "formatter":
                results["formatter"] = {"status": "ready", "message": "Formatter ready"}
            elif tool_name == "git":
                results["git"] = {"status": "ready", "message": "Git ready"}
        
        return results
    
    def get_tool_prompt_hints(self, active_tools: List[str]) -> str:
        """Generate prompt hints based on active tools"""
        hints = []
        
        if "sandbox" in active_tools:
            hints.append("You can execute code - provide runnable code.")
        if "linter" in active_tools:
            hints.append("Code will be checked for errors - ensure correct syntax.")
        if "plotter" in active_tools:
            hints.append("Visualization available - include plotting code if helpful.")
        if "debugger" in active_tools:
            hints.append("Debugging available - can trace execution.")
        if "formatter" in active_tools:
            hints.append("Code will be formatted - focus on logic, not style.")
        if "git" in active_tools:
            hints.append("Version control available.")
        
        return " ".join(hints)


def create_code_area(learning_rate: float = 0.01) -> CodeArea:
    """Factory function"""
    return CodeArea(learning_rate=learning_rate)
