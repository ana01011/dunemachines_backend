"""
OMNIUS v3 - Hybrid Brain Pipeline
Flow: Query → Thalamus → Rank by Signals → Primary First → Parallel Secondary → Synthesize

This version uses the new BrainPipeline for intelligent multi-area orchestration
instead of binary USE_CODE/USE_MATH decisions.
"""
from typing import Dict, Any, Tuple, List
import time
import numpy as np

from app.services.llm_service import llm_service
from app.services.deepseek_coder_service import deepseek_coder
from app.brain.thalamus import create_thalamus, BrainArea
from app.brain.areas import CodeArea, MathArea, MemoryArea
from app.brain.pretrain import ensure_pretrained
from app.brain.pipeline import BrainPipeline, PipelineResult, AreaRole


class OmniusV3Orchestrator:
    """
    OMNIUS v3 - Hybrid Pipeline Brain
    
    Improvements over v2:
    - Uses activation percentages to determine processing order
    - Primary area processes first, output feeds secondary areas
    - Secondary areas run in parallel
    - PFC synthesizes all outputs into coherent response
    - Better handling of multi-domain queries (e.g., "explain gravity and write simulation code")
    """
    
    def __init__(self):
        self.name = "OMNIUS"
        self.version = "3.0"
        self._init_brain()
        self.total_thoughts = 0
        self._last_decision = None
        self._last_stats = {}
        self._last_pipeline_result = None

    def _init_brain(self):
        print("[OMNIUS v3] Initializing hybrid brain pipeline...")
        
        # Initialize Thalamus (neural router)
        self.thalamus = create_thalamus(input_size=256, hidden_size=512, num_areas=5)
        ensure_pretrained(self.thalamus)
        
        # Initialize brain areas (for learning/stats)
        self.brain_areas = {
            "code": CodeArea(), 
            "math": MathArea(), 
            "memory": MemoryArea()
        }
        
        # Initialize hybrid pipeline
        self.pipeline = BrainPipeline()
        
        print("[OMNIUS v3] Hybrid brain ready!")

    def _encode_query(self, message: str) -> np.ndarray:
        """Encode query into neural signal for thalamus"""
        encoding = np.zeros(256)
        for i, char in enumerate(message.encode()[:200]):
            encoding[i % 256] += (char - 128) / 128.0
        norm = np.linalg.norm(encoding)
        return encoding / norm if norm > 0 else encoding

    async def think(self, message: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Main thinking process using hybrid pipeline
        
        Args:
            message: User's message/query
            context: Additional context (user_id, conversation history, etc.)
        
        Returns:
            Tuple of (response_text, list_of_regions_used)
        """
        start_time = time.time()
        stats = {"timings": {}, "signals": {}, "pipeline": {}}

        print(f"\n{'='*70}")
        print(f"[OMNIUS v3] {message[:80]}...")
        print(f"{'='*70}")

        # STEP 1: Thalamus routes query → outputs signals
        t0 = time.time()
        query_signal = self._encode_query(message)
        thalamus_out = self.thalamus.route(query_signal)
        stats["timings"]["thalamus"] = time.time() - t0

        # Convert to simple dict
        area_signals = {a.value: v for a, v in thalamus_out.activations.items()}
        
        print(f"\n[1. Thalamus Signals]")
        for area, signal in sorted(area_signals.items(), key=lambda x: x[1], reverse=True):
            role = self._get_role_name(signal)
            print(f"    {area:10}: {int(signal*100):3}% → {role}")
        
        stats["signals"] = area_signals

        # STEP 2: Run hybrid pipeline
        t0 = time.time()
        pipeline_result = await self.pipeline.process(
            query=message,
            signals=area_signals,
            context=context
        )
        stats["timings"]["pipeline"] = time.time() - t0
        
        # Collect regions used
        regions_used = ['prefrontal_cortex']  # PFC always involved
        for result in pipeline_result.area_results:
            if result.success:
                regions_used.append(f"{result.area_name}_cortex")
        
        # Build pipeline stats
        stats["pipeline"] = {
            "order": pipeline_result.pipeline_order,
            "synthesis_method": pipeline_result.synthesis_method,
            "area_timings": {r.area_name: r.processing_time for r in pipeline_result.area_results},
            "area_roles": {r.area_name: r.role.value for r in pipeline_result.area_results}
        }
        
        stats["timings"]["total"] = time.time() - start_time

        # Print summary
        print(f"\n[Summary]")
        print(f"    Pipeline order: {' → '.join(pipeline_result.pipeline_order)}")
        print(f"    Synthesis: {pipeline_result.synthesis_method}")
        print(f"    Total time: {stats['timings']['total']:.1f}s")
        print(f"{'='*70}\n")

        # Store for learning
        self._last_decision = {
            "areas": thalamus_out.active_areas, 
            "signals": area_signals,
            "pipeline_order": pipeline_result.pipeline_order
        }
        self._last_stats = stats
        self._last_pipeline_result = pipeline_result
        self.total_thoughts += 1

        return pipeline_result.final_response, regions_used

    def _get_role_name(self, signal: float) -> str:
        """Get human-readable role name from signal strength"""
        if signal >= 0.60:
            return "PRIMARY"
        elif signal >= 0.45:
            return "SECONDARY"
        elif signal >= 0.35:
            return "SUPPORTING"
        else:
            return "skip"

    def learn(self, reward: float) -> Dict:
        """Learn from feedback"""
        if not self._last_decision:
            return {"error": "Nothing to learn"}
        
        # Learn in thalamus
        result = self.thalamus.learn(reward, self._last_decision["areas"])
        
        # Save weights
        from app.brain.pretrain import save_weights
        save_weights(self.thalamus)
        
        return {
            "reward": reward, 
            "saved": True,
            "areas_reinforced": [a.value for a in self._last_decision["areas"]]
        }

    def get_last_stats(self) -> Dict:
        """Get stats from last thinking process"""
        return self._last_stats

    def get_last_pipeline_result(self) -> PipelineResult:
        """Get full pipeline result from last thinking process"""
        return self._last_pipeline_result

    def get_status(self) -> Dict:
        """Get OMNIUS status"""
        return {
            "identity": "OMNIUS v3 - Hybrid Pipeline",
            "version": self.version,
            "thoughts": self.total_thoughts,
            "status": "operational",
            "pipeline_stats": self.pipeline.get_stats(),
            "thalamus_stats": self.thalamus.get_stats()
        }

    def get_consciousness_regions(self) -> Dict[str, str]:
        """Get status of consciousness regions (for API compatibility)"""
        return {
            "prefrontal_cortex": "active",
            "code_cortex": "active" if deepseek_coder.model else "standby",
            "math_region": "active",
            "creative_center": "active",
            "memory_cortex": "active"
        }


# Create global instance
omnius_v3 = OmniusV3Orchestrator()
