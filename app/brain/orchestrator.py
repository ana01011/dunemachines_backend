"""
Brain Orchestrator - Connects PFC, Thalamus, and all Brain Areas
The main entry point for the multi-brain system
"""
import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from uuid import uuid4

from app.brain.thalamus import Thalamus, BrainArea, ThalamusOutput, create_thalamus
from app.brain.brain_area import BrainAreaBase, AreaOutput
from app.brain.areas import CodeArea, MathArea, MemoryArea


@dataclass
class BrainInput:
    """Input to the brain system"""
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class BrainOutput:
    """Output from the brain system"""
    query: str
    decision_id: str
    
    # Routing info
    active_areas: List[str]
    area_activations: Dict[str, float]
    
    # Tool info
    tools_by_area: Dict[str, List[str]]
    all_active_tools: List[str]
    
    # Prompt hints for LLM
    prompt_hints: str
    
    # Timing
    total_time_ms: float
    routing_time_ms: float
    
    # For learning later
    thalamus_pattern: Optional[str] = None
    confidence: float = 0.5


class BrainOrchestrator:
    """
    Main orchestrator for the multi-brain system.
    
    Connects:
    - PFC (Mistral) - understanding and integration
    - Thalamus - routing between areas
    - Brain Areas - specialist processing with tools
    - Neurochemistry - modulates everything
    """
    
    def __init__(self):
        # Initialize Thalamus (central router)
        self.thalamus = create_thalamus(
            input_size=256,
            hidden_size=512,
            num_areas=5
        )
        
        # Initialize brain areas
        self.areas: Dict[str, BrainAreaBase] = {
            "code": CodeArea(),
            "math": MathArea(),
            "memory": MemoryArea()
        }
        
        # Wire up inter-area connections
        self._setup_inter_area_connections()
        
        # Neurochemistry state (will be updated from external system)
        self.neuro_state = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "norepinephrine": 0.5,
            "oxytocin": 0.5,
            "adrenaline": 0.3,
            "endorphins": 0.5
        }
        
        # Statistics
        self.total_queries = 0
        self.last_output: Optional[BrainOutput] = None
    
    def _setup_inter_area_connections(self):
        """Setup bidirectional connections between areas"""
        # Code <-> Math
        if "code" in self.areas and "math" in self.areas:
            self.areas["code"].connect_to("math", 0.4)
            self.areas["math"].connect_to("code", 0.4)
        
        # Memory connects to all
        if "memory" in self.areas:
            for name in self.areas:
                if name != "memory":
                    self.areas["memory"].connect_to(name, 0.3)
                    self.areas[name].connect_to("memory", 0.2)
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query into a vector.
        Simple hash-based encoding for now.
        In production, use actual embeddings from Mistral.
        """
        # Create a simple encoding based on query characteristics
        encoding = np.zeros(256)
        
        # Hash-based features
        query_hash = hashlib.sha256(query.encode()).digest()
        for i, byte in enumerate(query_hash):
            encoding[i] = (byte - 128) / 128.0
        
        # Length feature
        encoding[32] = min(len(query) / 500.0, 1.0)
        
        # Keyword-based features
        keywords = {
            "code": [33, 34], "python": [35, 36], "function": [37],
            "math": [40, 41], "calculate": [42], "equation": [43], "solve": [44],
            "remember": [50], "memory": [51], "previous": [52],
            "physics": [60], "force": [61], "energy": [62],
            "explain": [70], "what": [71], "how": [72], "why": [73]
        }
        
        query_lower = query.lower()
        for keyword, indices in keywords.items():
            if keyword in query_lower:
                for idx in indices:
                    encoding[idx] = 0.8
        
        # Normalize
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        
        return encoding
    
    def _get_pfc_hints(self, query: str) -> Dict[BrainArea, float]:
        """
        Get routing hints from PFC analysis.
        Simple keyword-based for now.
        In production, use Mistral to analyze.
        """
        hints = {}
        query_lower = query.lower()
        
        # Code indicators
        code_keywords = ["code", "program", "function", "python", "javascript", "debug", "error", "bug", "script"]
        code_score = sum(1 for k in code_keywords if k in query_lower) / len(code_keywords)
        hints[BrainArea.CODE] = min(code_score * 2, 1.0)
        
        # Math indicators
        math_keywords = ["math", "calculate", "equation", "solve", "integral", "derivative", "sum", "number", "formula"]
        math_score = sum(1 for k in math_keywords if k in query_lower) / len(math_keywords)
        hints[BrainArea.MATH] = min(math_score * 2, 1.0)
        
        # Memory indicators
        memory_keywords = ["remember", "previous", "last time", "history", "before", "recall", "forgot"]
        memory_score = sum(1 for k in memory_keywords if k in query_lower) / len(memory_keywords)
        hints[BrainArea.MEMORY] = min(memory_score * 2, 1.0)
        
        # Physics indicators
        physics_keywords = ["physics", "force", "energy", "motion", "gravity", "velocity", "acceleration"]
        physics_score = sum(1 for k in physics_keywords if k in query_lower) / len(physics_keywords)
        hints[BrainArea.PHYSICS] = min(physics_score * 2, 1.0)
        
        # Language (default for explanations)
        hints[BrainArea.LANGUAGE] = 0.3
        if any(k in query_lower for k in ["explain", "what is", "describe", "tell me"]):
            hints[BrainArea.LANGUAGE] = 0.7
        
        return hints
    
    def process(self, brain_input: BrainInput) -> BrainOutput:
        """
        Process a query through the brain system.
        
        1. Encode query
        2. Get PFC hints
        3. Route through Thalamus
        4. Activate brain areas
        5. Collect tool selections
        6. Generate output
        """
        start_time = time.time()
        
        # Update neurochemistry in all components
        self._propagate_neuro_state()
        
        # Encode query
        query_encoding = self._encode_query(brain_input.query)
        
        # Get PFC hints
        pfc_hints = self._get_pfc_hints(brain_input.query)
        
        # Route through Thalamus
        thalamus_output = self.thalamus.route(query_encoding, pfc_hints)
        
        # Process through active areas
        tools_by_area: Dict[str, List[str]] = {}
        all_active_tools: List[str] = []
        prompt_hints_parts: List[str] = []
        
        # Create area input (subset of query encoding)
        area_input = query_encoding[:128]
        
        for brain_area in thalamus_output.active_areas:
            area_name = brain_area.value
            
            if area_name in self.areas:
                area = self.areas[area_name]
                activation = thalamus_output.activations.get(brain_area, 0.5)
                
                # Process through area
                area_output = area.process(area_input, activation)
                
                # Collect tools
                tools_by_area[area_name] = area_output.active_tools
                all_active_tools.extend(area_output.active_tools)
                
                # Get prompt hints
                if area_output.active_tools:
                    hints = area.get_tool_prompt_hints(area_output.active_tools)
                    if hints:
                        prompt_hints_parts.append(f"[{area_name.upper()}] {hints}")
                
                # Send signals to connected areas
                for target_area, signal in (area_output.signals_sent or {}).items():
                    if target_area in self.areas:
                        self.areas[target_area].receive_signal(area_name, signal)
        
        # Build output
        total_time = (time.time() - start_time) * 1000
        
        output = BrainOutput(
            query=brain_input.query,
            decision_id=str(uuid4()),
            active_areas=[a.value for a in thalamus_output.active_areas],
            area_activations={a.value: v for a, v in thalamus_output.activations.items()},
            tools_by_area=tools_by_area,
            all_active_tools=list(set(all_active_tools)),
            prompt_hints=" | ".join(prompt_hints_parts) if prompt_hints_parts else "",
            total_time_ms=total_time,
            routing_time_ms=thalamus_output.routing_time_ms,
            thalamus_pattern=thalamus_output.pattern_used,
            confidence=thalamus_output.confidence
        )
        
        self.last_output = output
        self.total_queries += 1
        
        return output
    
    def learn(self, decision_id: str, reward: float, tools_actually_used: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Learn from outcome.
        
        Args:
            decision_id: ID of the decision to learn from
            reward: Reward signal (-1 to 1)
            tools_actually_used: Which tools were actually used (optional)
        """
        if self.last_output is None or self.last_output.decision_id != decision_id:
            return {"error": "Decision not found"}
        
        # Learn in Thalamus
        areas_used = [BrainArea(a) for a in self.last_output.active_areas]
        thalamus_result = self.thalamus.learn(reward, areas_used)
        
        # Learn in each active area
        area_results = {}
        for area_name in self.last_output.active_areas:
            if area_name in self.areas:
                tools = tools_actually_used or self.last_output.tools_by_area.get(area_name, [])
                result = self.areas[area_name].learn(reward, tools)
                area_results[area_name] = result
        
        return {
            "decision_id": decision_id,
            "reward": reward,
            "thalamus": thalamus_result,
            "areas": area_results
        }
    
    def set_neuro_state(self, neuro_state: Dict[str, float]):
        """Update neurochemistry state"""
        self.neuro_state.update(neuro_state)
        self._propagate_neuro_state()
    
    def _propagate_neuro_state(self):
        """Propagate neurochemistry to all components"""
        self.thalamus.set_neuro_state(self.neuro_state)
        for area in self.areas.values():
            area.set_neuro_state(self.neuro_state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_queries": self.total_queries,
            "thalamus": self.thalamus.get_stats(),
            "areas": {name: area.get_stats() for name, area in self.areas.items()},
            "neuro_state": self.neuro_state.copy()
        }
    
    def get_prompt_for_llm(self, brain_output: BrainOutput, query: str) -> str:
        """
        Generate a prompt section for the LLM based on brain output.
        This gets injected into the LLM prompt.
        """
        lines = [
            "BRAIN STATE:",
            f"Active regions: {', '.join(brain_output.active_areas)}",
            f"Confidence: {brain_output.confidence:.0%}",
            ""
        ]
        
        if brain_output.all_active_tools:
            lines.append(f"Available tools: {', '.join(brain_output.all_active_tools)}")
        
        if brain_output.prompt_hints:
            lines.append(f"Hints: {brain_output.prompt_hints}")
        
        return "\n".join(lines)


def create_brain_orchestrator() -> BrainOrchestrator:
    """Factory function"""
    return BrainOrchestrator()
