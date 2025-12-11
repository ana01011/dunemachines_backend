"""
Memory Area (Hippocampus) - Handles memory storage and retrieval
"""
from typing import Dict, List, Any, Optional
from app.brain.brain_area import BrainAreaBase, Tool, AreaOutput


class MemoryArea(BrainAreaBase):
    """
    Memory brain area (Hippocampus) - stores and retrieves information.
    Tools: VectorSearch, KeywordSearch, GraphQuery, Timeline
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(
            name="memory",
            input_size=128,
            hidden_size=256,
            learning_rate=learning_rate,
            tool_threshold=0.5
        )
        self.connect_to("code", strength=0.3)
        self.connect_to("math", strength=0.3)
        self.connect_to("physics", strength=0.3)
        self.connect_to("language", strength=0.3)
    
    def _init_tools(self):
        self.tools = [
            Tool(name="vector_search", description="Semantic similarity search", category="retrieval"),
            Tool(name="keyword_search", description="Exact keyword matching", category="retrieval"),
            Tool(name="graph_query", description="Query relationship graphs", category="retrieval"),
            Tool(name="timeline", description="Chronological memory access", category="retrieval")
        ]
    
    def execute_tools(self, active_tools: List[str], query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        results = {}
        for tool_name in active_tools:
            results[tool_name] = {"status": "ready", "message": f"{tool_name} ready"}
        return results
    
    def get_tool_prompt_hints(self, active_tools: List[str]) -> str:
        hints = []
        if "vector_search" in active_tools:
            hints.append("Can search for similar past information.")
        if "keyword_search" in active_tools:
            hints.append("Can search for exact terms.")
        if "graph_query" in active_tools:
            hints.append("Can query relationships.")
        if "timeline" in active_tools:
            hints.append("Can access history.")
        return " ".join(hints)


def create_memory_area(learning_rate: float = 0.01) -> MemoryArea:
    return MemoryArea(learning_rate=learning_rate)
