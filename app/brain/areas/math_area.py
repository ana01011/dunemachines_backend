"""
Math Area - Handles mathematical computations and reasoning
"""
from typing import Dict, List, Any, Optional
from app.brain.brain_area import BrainAreaBase, Tool, AreaOutput


class MathArea(BrainAreaBase):
    """
    Math brain area - specializes in mathematical tasks.
    Tools: Solver, Plotter, Formulas, Symbolic, Calculator
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(
            name="math",
            input_size=128,
            hidden_size=256,
            learning_rate=learning_rate,
            tool_threshold=0.5
        )
        # Set up connections to related areas
        self.connect_to("code", strength=0.4)
        self.connect_to("physics", strength=0.5)
        self.connect_to("memory", strength=0.2)
    
    def _init_tools(self):
        """Initialize math-related tools"""
        self.tools = [
            Tool(
                name="solver",
                description="Solve equations numerically",
                category="computation"
            ),
            Tool(
                name="symbolic",
                description="Symbolic math operations (algebra, calculus)",
                category="symbolic"
            ),
            Tool(
                name="plotter",
                description="Plot mathematical functions",
                category="visualization"
            ),
            Tool(
                name="formulas",
                description="Look up mathematical formulas",
                category="knowledge"
            ),
            Tool(
                name="calculator",
                description="Basic and scientific calculations",
                category="computation"
            )
        ]
    
    def execute_tools(self, active_tools: List[str], expression: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute selected math tools"""
        results = {}
        
        for tool_name in active_tools:
            if tool_name == "solver":
                results["solver"] = {"status": "ready", "message": "Numerical solver ready"}
            elif tool_name == "symbolic":
                results["symbolic"] = {"status": "ready", "message": "Symbolic engine ready"}
            elif tool_name == "plotter":
                results["plotter"] = {"status": "ready", "message": "Math plotter ready"}
            elif tool_name == "formulas":
                results["formulas"] = {"status": "ready", "message": "Formula database ready"}
            elif tool_name == "calculator":
                results["calculator"] = {"status": "ready", "message": "Calculator ready"}
        
        return results
    
    def get_tool_prompt_hints(self, active_tools: List[str]) -> str:
        """Generate prompt hints based on active tools"""
        hints = []
        
        if "solver" in active_tools:
            hints.append("Numerical solver available for equations.")
        if "symbolic" in active_tools:
            hints.append("Symbolic math available - can do algebra and calculus.")
        if "plotter" in active_tools:
            hints.append("Can plot functions and graphs.")
        if "formulas" in active_tools:
            hints.append("Mathematical formula database accessible.")
        if "calculator" in active_tools:
            hints.append("Calculator available for computations.")
        
        return " ".join(hints)


def create_math_area(learning_rate: float = 0.01) -> MathArea:
    """Factory function"""
    return MathArea(learning_rate=learning_rate)
