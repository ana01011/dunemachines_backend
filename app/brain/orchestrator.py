"""
Brain Orchestrator - Main controller for the Multi-Brain AI System
OMNIUS v2 - Coordinates neural routing, specialist models, and learning

This is the main entry point that:
1. Takes user queries
2. Uses NeuralRouter to decide which models to activate
3. Calls specialist models (Language, Math)
4. Manages memory via Hippocampus
5. Applies learning from feedback
6. Integrates with neurochemistry
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

from app.brain.neuron import LIFNeuron, NeuronType
from app.brain.neural_router import NeuralRouter, RouterOutput, BrainRegionType, create_neural_router
from app.brain.learning import HebbianLearner, RewardSignal, RewardType, create_learner
from app.brain.hippocampus import Hippocampus, create_hippocampus, MemoryRetrievalResult
from app.models.brain import (
    BrainRegion, ThinkingMode, RouterActivation,
    BrainRequest, BrainResponse, BrainStatus,
    RouterStatus, BrainRegionStatus, FeedbackRequest
)

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for the brain orchestrator"""
    use_memory: bool = True
    use_learning: bool = True
    use_neurochemistry: bool = True
    memory_influence_weight: float = 0.3
    default_thinking_mode: ThinkingMode = ThinkingMode.FAST
    activation_threshold: float = 0.5
    max_memory_results: int = 5
    embedding_size: int = 768


@dataclass 
class PendingDecision:
    """Tracks a decision awaiting feedback"""
    decision_id: str
    user_id: str
    query: str
    router_output: RouterOutput
    regions_used: List[BrainRegion]
    response: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None


class BrainOrchestrator:
    """
    Main orchestrator for the multi-brain AI system.
    
    Coordinates:
    - NeuralRouter: Decides which brain regions to activate
    - Specialist Models: Language (Mistral), Math (DeepSeek)
    - Hippocampus: Memory storage and retrieval
    - HebbianLearner: Learning from feedback
    - Neurochemistry: Modulates behavior
    """
    
    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        language_model: Any = None,
        math_model: Any = None,
        neurochemistry_system: Any = None
    ):
        """
        Initialize the brain orchestrator.
        
        Args:
            config: Brain configuration
            language_model: Language model service (Mistral)
            math_model: Math/code model service (DeepSeek)
            neurochemistry_system: Neurochemistry system for modulation
        """
        self.config = config or BrainConfig()
        
        # External model services (injected)
        self.language_model = language_model
        self.math_model = math_model
        self.neurochemistry_system = neurochemistry_system
        
        # Internal components
        self.router = create_neural_router(
            input_size=self.config.embedding_size,
            hidden_size=128,
            output_size=3
        )
        self.learner = create_learner()
        self.hippocampus = create_hippocampus()
        
        # State tracking
        self.pending_decisions: Dict[str, PendingDecision] = {}
        self.is_initialized = False
        self.start_time = datetime.now()
        
        # Statistics
        self.total_queries = 0
        self.total_feedbacks = 0
        self.region_usage = {
            BrainRegion.LANGUAGE: 0,
            BrainRegion.MATH: 0,
            BrainRegion.MEMORY: 0
        }
        
        logger.info("BrainOrchestrator initialized")
    
    def initialize(
        self,
        language_model: Any = None,
        math_model: Any = None,
        neurochemistry_system: Any = None
    ):
        """
        Initialize or update model connections.
        
        Args:
            language_model: Language model service
            math_model: Math model service
            neurochemistry_system: Neurochemistry system
        """
        if language_model is not None:
            self.language_model = language_model
        if math_model is not None:
            self.math_model = math_model
        if neurochemistry_system is not None:
            self.neurochemistry_system = neurochemistry_system
        
        self.is_initialized = (
            self.language_model is not None and 
            self.router.is_initialized
        )
        
        logger.info(f"BrainOrchestrator initialized: {self.is_initialized}")
        return self.is_initialized
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        For now, uses a simple hash-based embedding.
        In production, use the language model's encoder.
        """
        if self.language_model is not None and hasattr(self.language_model, 'encode'):
            try:
                return self.language_model.encode(text)
            except Exception as e:
                logger.warning(f"Failed to get model embedding: {e}")
        
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.config.embedding_size)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def _get_neurochemistry_state(self, user_id: str) -> Dict[str, float]:
        """Get current neurochemistry state for user"""
        if self.neurochemistry_system is not None:
            try:
                state = self.neurochemistry_system.get_state(user_id)
                if state:
                    return {
                        "dopamine": state.get("dopamine", 0.5),
                        "serotonin": state.get("serotonin", 0.5),
                        "cortisol": state.get("cortisol", 0.5),
                        "norepinephrine": state.get("norepinephrine", 0.5),
                        "adrenaline": state.get("adrenaline", 0.5),
                        "oxytocin": state.get("oxytocin", 0.5),
                        "endorphins": state.get("endorphins", 0.5)
                    }
            except Exception as e:
                logger.warning(f"Failed to get neurochemistry state: {e}")
        
        return {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "norepinephrine": 0.5,
            "adrenaline": 0.3,
            "oxytocin": 0.5,
            "endorphins": 0.5
        }
    
    def _apply_memory_bias(
        self,
        embedding: np.ndarray,
        user_id: str
    ) -> Tuple[np.ndarray, MemoryRetrievalResult]:
        """
        Apply memory bias to embedding based on past experiences.
        
        Returns modified embedding and memory retrieval result.
        """
        if not self.config.use_memory:
            return embedding, MemoryRetrievalResult(
                memories=[], similarity_scores=[],
                suggested_regions=[], confidence=0.0, retrieval_time_ms=0.0
            )
        
        memory_result = self.hippocampus.retrieve(
            user_id=user_id,
            query_embedding=embedding,
            n_results=self.config.max_memory_results
        )
        
        if memory_result.memories and memory_result.confidence > 0.3:
            bias_weight = self.config.memory_influence_weight * memory_result.confidence
            memory_embedding = np.mean([
                self.hippocampus.get_embedding(m.id)
                for m in memory_result.memories
                if self.hippocampus.get_embedding(m.id) is not None
            ] or [embedding], axis=0)
            
            biased_embedding = (1 - bias_weight) * embedding + bias_weight * memory_embedding
            biased_embedding = biased_embedding / (np.linalg.norm(biased_embedding) + 1e-8)
            return biased_embedding, memory_result
        
        return embedding, memory_result
    
    def _map_router_to_brain_regions(self, router_regions: List[BrainRegionType]) -> List[BrainRegion]:
        """Map router region types to brain region enums"""
        mapping = {
            BrainRegionType.LANGUAGE: BrainRegion.LANGUAGE,
            BrainRegionType.MATH: BrainRegion.MATH,
            BrainRegionType.MEMORY: BrainRegion.MEMORY
        }
        return [mapping[r] for r in router_regions if r in mapping]
    
    async def _generate_language_response(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using language model"""
        if self.language_model is None:
            return "[Language model not available]"
        
        try:
            prompt = query
            if context:
                prompt = f"{context}\n\nUser: {query}"
            
            if hasattr(self.language_model, 'generate_async'):
                response = await self.language_model.generate_async(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif hasattr(self.language_model, 'generate'):
                response = self.language_model.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = str(self.language_model(prompt))
            
            return response
        except Exception as e:
            logger.error(f"Language model error: {e}")
            return f"[Language model error: {str(e)}]"
    
    async def _generate_math_response(
        self,
        query: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using math/code model"""
        if self.math_model is None:
            return await self._generate_language_response(
                query, context="This is a math/code question. Please solve it step by step.",
                temperature=temperature, max_tokens=max_tokens
            )
        
        try:
            if hasattr(self.math_model, 'generate_async'):
                response = await self.math_model.generate_async(
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif hasattr(self.math_model, 'generate'):
                response = self.math_model.generate(
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = str(self.math_model(query))
            
            return response
        except Exception as e:
            logger.error(f"Math model error: {e}")
            return await self._generate_language_response(
                query, context="This is a math/code question.",
                temperature=temperature, max_tokens=max_tokens
            )
    
    async def process(
        self,
        request: BrainRequest,
        user_id: str,
        conversation_id: Optional[UUID] = None
    ) -> BrainResponse:
        """
        Process a request through the multi-brain system.
        
        Args:
            request: The brain request
            user_id: User identifier
            conversation_id: Optional conversation ID
        
        Returns:
            BrainResponse with the generated response and metadata
        """
        start_time = time.time()
        query = request.message
        
        self.total_queries += 1
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")
        
        neuro_state = self._get_neurochemistry_state(user_id)
        
        if self.config.use_neurochemistry:
            self.router.modulate_from_neurochemistry(neuro_state)
            self.learner.modulate_from_neurochemistry(neuro_state)
        
        embedding = self._get_embedding(query)
        
        biased_embedding, memory_result = self._apply_memory_bias(embedding, user_id)
        
        if request.force_regions:
            router_output = RouterOutput(
                activations={
                    "language": 1.0 if BrainRegion.LANGUAGE in request.force_regions else 0.0,
                    "math": 1.0 if BrainRegion.MATH in request.force_regions else 0.0,
                    "memory": 1.0 if BrainRegion.MEMORY in request.force_regions else 0.0
                },
                active_regions=[
                    BrainRegionType.LANGUAGE if BrainRegion.LANGUAGE in request.force_regions else None,
                    BrainRegionType.MATH if BrainRegion.MATH in request.force_regions else None,
                    BrainRegionType.MEMORY if BrainRegion.MEMORY in request.force_regions else None
                ],
                confidence=1.0,
                hidden_activations=np.zeros(128),
                output_raw=np.zeros(3),
                routing_time_ms=0.0,
                query_hash=""
            )
            router_output.active_regions = [r for r in router_output.active_regions if r is not None]
        else:
            router_output = self.router.forward(biased_embedding, query)
        
        if memory_result.suggested_regions and memory_result.confidence > 0.5:
            for suggested in memory_result.suggested_regions:
                if suggested == "math" and BrainRegionType.MATH not in router_output.active_regions:
                    if router_output.activations["math"] > 0.3:
                        router_output.active_regions.append(BrainRegionType.MATH)
                elif suggested == "language" and BrainRegionType.LANGUAGE not in router_output.active_regions:
                    router_output.active_regions.append(BrainRegionType.LANGUAGE)
        
        regions_used = self._map_router_to_brain_regions(router_output.active_regions)
        
        for region in regions_used:
            self.region_usage[region] = self.region_usage.get(region, 0) + 1
        
        thinking_mode = request.thinking_mode or self.config.default_thinking_mode
        
        response_text = ""
        
        if BrainRegion.MATH in regions_used and BrainRegion.LANGUAGE in regions_used:
            math_response = await self._generate_math_response(
                query, temperature=0.3, max_tokens=request.max_tokens
            )
            synthesis_prompt = f"""The math/code specialist provided this solution:

{math_response}

Please synthesize this into a clear, helpful response for the user's question: {query}"""
            
            response_text = await self._generate_language_response(
                synthesis_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        elif BrainRegion.MATH in regions_used:
            response_text = await self._generate_math_response(
                query,
                temperature=0.3,
                max_tokens=request.max_tokens
            )
        
        else:
            response_text = await self._generate_language_response(
                query,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        decision_id = router_output.decision_id
        message_id = uuid4()
        conv_id = conversation_id or uuid4()
        
        if request.store_in_memory and self.config.use_memory:
            self.hippocampus.store(
                user_id=user_id,
                query=query,
                embedding=embedding,
                regions_used=[r.value for r in regions_used],
                was_successful=True,
                reward=0.0
            )
        
        self.pending_decisions[decision_id] = PendingDecision(
            decision_id=decision_id,
            user_id=user_id,
            query=query,
            router_output=router_output,
            regions_used=regions_used,
            response=response_text,
            timestamp=datetime.now(),
            embedding=embedding
        )
        
        if len(self.pending_decisions) > 1000:
            oldest_keys = sorted(
                self.pending_decisions.keys(),
                key=lambda k: self.pending_decisions[k].timestamp
            )[:500]
            for key in oldest_keys:
                del self.pending_decisions[key]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BrainResponse(
            response=response_text,
            conversation_id=conv_id,
            message_id=message_id,
            regions_activated=regions_used,
            activations=RouterActivation(
                language=router_output.activations["language"],
                math=router_output.activations["math"],
                memory=router_output.activations["memory"]
            ),
            routing_confidence=router_output.confidence,
            thinking_mode=thinking_mode,
            tokens_used=len(response_text.split()),
            processing_time_ms=processing_time_ms,
            routing_time_ms=router_output.routing_time_ms,
            decision_id=UUID(decision_id),
            memories_retrieved=len(memory_result.memories),
            memory_influence
=memory_result.confidence * self.config.memory_influence_weight, neurochemistry=neuro_state if self.config.use_neurochemistry else None )

def process_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
    """
    Process user feedback and trigger learning.
    
    Args:
        feedback: User feedback on a response
    
    Returns:
        Dictionary with learning results
    """
    decision_id = str(feedback.decision_id)
    
    if decision_id not in self.pending_decisions:
        return {
            "success": False,
            "error": "Decision not found or expired"
        }
    
    decision = self.pending_decisions[decision_id]
    
    reward = (feedback.rating - 3) / 2.0
    
    reward_signal = RewardSignal(
        value=reward,
        reward_type=RewardType.EXPLICIT_FEEDBACK,
        source="user",
        decision_id=decision_id,
        timestamp=datetime.now()
    )
    
    if self.config.use_learning:
        neuro_state = self._get_neurochemistry_state(decision.user_id)
        self.learner.modulate_from_neurochemistry(neuro_state)
        
        if decision.router_output.hidden_activations is not None:
            learning_result = self.learner.apply_reward(
                reward_signal=reward_signal,
                pre_activation=decision.embedding,
                post_activation=decision.router_output.hidden_activations
            )
            
            self.router.learn(
                reward=reward,
                learning_rate_modifier=self.learner.get_learning_rate() / 0.01
            )
    
    if self.config.use_memory and decision.embedding is not None:
        self.hippocampus.update_memory_reward(
            embedding=decision.embedding,
            reward=reward,
            was_successful=(feedback.rating >= 3)
        )
    
    del self.pending_decisions[decision_id]
    self.total_feedbacks += 1
    
    return {
        "success": True,
        "decision_id": decision_id,
        "reward_applied": reward,
        "learning_triggered": self.config.use_learning,
        "router_decisions": self.router.total_decisions,
        "router_learning_events": self.router.total_learning_events
    }

def get_status(self) -> BrainStatus:
    """Get current brain system status"""
    uptime = (datetime.now() - self.start_time).total_seconds()
    
    router_status = RouterStatus(
        is_initialized=self.router.is_initialized,
        weights_version=1,
        total_neurons=len(self.router.hidden_neurons) + len(self.router.output_neurons),
        architecture=self.router.get_stats()["architecture"],
        total_decisions=self.router.total_decisions,
        accuracy_estimate=0.5,
        learning_enabled=self.config.use_learning
    )
    
    region_statuses = [
        BrainRegionStatus(
            region=BrainRegion.LANGUAGE,
            is_loaded=self.language_model is not None,
            is_healthy=self.language_model is not None,
            model_name="Mistral 7B" if self.language_model else None,
            parameters="7B" if self.language_model else None,
            total_calls=self.region_usage.get(BrainRegion.LANGUAGE, 0)
        ),
        BrainRegionStatus(
            region=BrainRegion.MATH,
            is_loaded=self.math_model is not None,
            is_healthy=self.math_model is not None,
            model_name="DeepSeek Coder" if self.math_model else None,
            parameters="6.7B" if self.math_model else None,
            total_calls=self.region_usage.get(BrainRegion.MATH, 0)
        ),
        BrainRegionStatus(
            region=BrainRegion.MEMORY,
            is_loaded=True,
            is_healthy=True,
            model_name="Hippocampus",
            total_calls=self.region_usage.get(BrainRegion.MEMORY, 0)
        )
    ]
    
    hippo_stats = self.hippocampus.get_stats()
    neuro_state = None
    if self.neurochemistry_system is not None:
        neuro_state = self._get_neurochemistry_state("system")
    
    return BrainStatus(
        router=router_status,
        regions=region_statuses,
        hippocampus_connected=True,
        hippocampus_memories=hippo_stats.get("total_memories", 0),
        neurochemistry_connected=self.neurochemistry_system is not None,
        neurochemistry_state=neuro_state,
        uptime_seconds=uptime
    )

def get_stats(self) -> Dict[str, Any]:
    """Get orchestrator statistics"""
    return {
        "total_queries": self.total_queries,
        "total_feedbacks": self.total_feedbacks,
        "pending_decisions": len(self.pending_decisions),
        "region_usage": {k.value: v for k, v in self.region_usage.items()},
        "router_stats": self.router.get_stats(),
        "learner_stats": {
            "base_learning_rate": self.learner.base_learning_rate,
            "current_learning_rate": self.learner.get_learning_rate()
        },
        "hippocampus_stats": self.hippocampus.get_stats(),
        "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
    }
_global_orchestrator: Optional[BrainOrchestrator] = None

def get_brain_orchestrator() -> BrainOrchestrator: """Get or create the global brain orchestrator""" global _global_orchestrator if _global_orchestrator is None: _global_orchestrator = BrainOrchestrator() return _global_orchestrator

def create_brain_orchestrator( language_model: Any = None, math_model: Any = None, neurochemistry_system: Any = None, config: Optional[BrainConfig] = None ) -> BrainOrchestrator: """ Create a new brain orchestrator.

Args:
    language_model: Language model service
    math_model: Math model service  
    neurochemistry_system: Neurochemistry system
    config: Brain configuration

Returns:
    Configured BrainOrchestrator instance
"""
global _global_orchestrator
_global_orchestrator = BrainOrchestrator(
    config=config,
    language_model=language_model,
    math_model=math_model,
    neurochemistry_system=neurochemistry_system
)
return _global_orchestrator
EOFcat > /root/openhermes_backend/app/brain/orchestrator.py << 'EOF'
"""
Brain Orchestrator - Main controller for the Multi-Brain AI System
OMNIUS v2 - Coordinates neural routing, specialist models, and learning

This is the main entry point that:
1. Takes user queries
2. Uses NeuralRouter to decide which models to activate
3. Calls specialist models (Language, Math)
4. Manages memory via Hippocampus
5. Applies learning from feedback
6. Integrates with neurochemistry
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

from app.brain.neuron import LIFNeuron, NeuronType
from app.brain.neural_router import NeuralRouter, RouterOutput, BrainRegionType, create_neural_router
from app.brain.learning import HebbianLearner, RewardSignal, RewardType, create_learner
from app.brain.hippocampus import Hippocampus, create_hippocampus, MemoryRetrievalResult
from app.models.brain import (
    BrainRegion, ThinkingMode, RouterActivation,
    BrainRequest, BrainResponse, BrainStatus,
    RouterStatus, BrainRegionStatus, FeedbackRequest
)

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for the brain orchestrator"""
    use_memory: bool = True
    use_learning: bool = True
    use_neurochemistry: bool = True
    memory_influence_weight: float = 0.3
    default_thinking_mode: ThinkingMode = ThinkingMode.FAST
    activation_threshold: float = 0.5
    max_memory_results: int = 5
    embedding_size: int = 768


@dataclass 
class PendingDecision:
    """Tracks a decision awaiting feedback"""
    decision_id: str
    user_id: str
    query: str
    router_output: RouterOutput
    regions_used: List[BrainRegion]
    response: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None


class BrainOrchestrator:
    """
    Main orchestrator for the multi-brain AI system.
    
    Coordinates:
    - NeuralRouter: Decides which brain regions to activate
    - Specialist Models: Language (Mistral), Math (DeepSeek)
    - Hippocampus: Memory storage and retrieval
    - HebbianLearner: Learning from feedback
    - Neurochemistry: Modulates behavior
    """
    
    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        language_model: Any = None,
        math_model: Any = None,
        neurochemistry_system: Any = None
    ):
        """
        Initialize the brain orchestrator.
        
        Args:
            config: Brain configuration
            language_model: Language model service (Mistral)
            math_model: Math/code model service (DeepSeek)
            neurochemistry_system: Neurochemistry system for modulation
        """
        self.config = config or BrainConfig()
        
        # External model services (injected)
        self.language_model = language_model
        self.math_model = math_model
        self.neurochemistry_system = neurochemistry_system
        
        # Internal components
        self.router = create_neural_router(
            input_size=self.config.embedding_size,
            hidden_size=128,
            output_size=3
        )
        self.learner = create_learner()
        self.hippocampus = create_hippocampus()
        
        # State tracking
        self.pending_decisions: Dict[str, PendingDecision] = {}
        self.is_initialized = False
        self.start_time = datetime.now()
        
        # Statistics
        self.total_queries = 0
        self.total_feedbacks = 0
        self.region_usage = {
            BrainRegion.LANGUAGE: 0,
            BrainRegion.MATH: 0,
            BrainRegion.MEMORY: 0
        }
        
        logger.info("BrainOrchestrator initialized")
    
    def initialize(
        self,
        language_model: Any = None,
        math_model: Any = None,
        neurochemistry_system: Any = None
    ):
        """
        Initialize or update model connections.
        
        Args:
            language_model: Language model service
            math_model: Math model service
            neurochemistry_system: Neurochemistry system
        """
        if language_model is not None:
            self.language_model = language_model
        if math_model is not None:
            self.math_model = math_model
        if neurochemistry_system is not None:
            self.neurochemistry_system = neurochemistry_system
        
        self.is_initialized = (
            self.language_model is not None and 
            self.router.is_initialized
        )
        
        logger.info(f"BrainOrchestrator initialized: {self.is_initialized}")
        return self.is_initialized
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        For now, uses a simple hash-based embedding.
        In production, use the language model's encoder.
        """
        if self.language_model is not None and hasattr(self.language_model, 'encode'):
            try:
                return self.language_model.encode(text)
            except Exception as e:
                logger.warning(f"Failed to get model embedding: {e}")
        
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.config.embedding_size)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def _get_neurochemistry_state(self, user_id: str) -> Dict[str, float]:
        """Get current neurochemistry state for user"""
        if self.neurochemistry_system is not None:
            try:
                state = self.neurochemistry_system.get_state(user_id)
                if state:
                    return {
                        "dopamine": state.get("dopamine", 0.5),
                        "serotonin": state.get("serotonin", 0.5),
                        "cortisol": state.get("cortisol", 0.5),
                        "norepinephrine": state.get("norepinephrine", 0.5),
                        "adrenaline": state.get("adrenaline", 0.5),
                        "oxytocin": state.get("oxytocin", 0.5),
                        "endorphins": state.get("endorphins", 0.5)
                    }
            except Exception as e:
                logger.warning(f"Failed to get neurochemistry state: {e}")
        
        return {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "norepinephrine": 0.5,
            "adrenaline": 0.3,
            "oxytocin": 0.5,
            "endorphins": 0.5
        }
    
    def _apply_memory_bias(
        self,
        embedding: np.ndarray,
        user_id: str
    ) -> Tuple[np.ndarray, MemoryRetrievalResult]:
        """
        Apply memory bias to embedding based on past experiences.
        
        Returns modified embedding and memory retrieval result.
        """
        if not self.config.use_memory:
            return embedding, MemoryRetrievalResult(
                memories=[], similarity_scores=[],
                suggested_regions=[], confidence=0.0, retrieval_time_ms=0.0
            )
        
        memory_result = self.hippocampus.retrieve(
            user_id=user_id,
            query_embedding=embedding,
            n_results=self.config.max_memory_results
        )
        
        if memory_result.memories and memory_result.confidence > 0.3:
            bias_weight = self.config.memory_influence_weight * memory_result.confidence
            memory_embedding = np.mean([
                self.hippocampus.get_embedding(m.id)
                for m in memory_result.memories
                if self.hippocampus.get_embedding(m.id) is not None
            ] or [embedding], axis=0)
            
            biased_embedding = (1 - bias_weight) * embedding + bias_weight * memory_embedding
            biased_embedding = biased_embedding / (np.linalg.norm(biased_embedding) + 1e-8)
            return biased_embedding, memory_result
        
        return embedding, memory_result
    
    def _map_router_to_brain_regions(self, router_regions: List[BrainRegionType]) -> List[BrainRegion]:
        """Map router region types to brain region enums"""
        mapping = {
            BrainRegionType.LANGUAGE: BrainRegion.LANGUAGE,
            BrainRegionType.MATH: BrainRegion.MATH,
            BrainRegionType.MEMORY: BrainRegion.MEMORY
        }
        return [mapping[r] for r in router_regions if r in mapping]
    
    async def _generate_language_response(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using language model"""
        if self.language_model is None:
            return "[Language model not available]"
        
        try:
            prompt = query
            if context:
                prompt = f"{context}\n\nUser: {query}"
            
            if hasattr(self.language_model, 'generate_async'):
                response = await self.language_model.generate_async(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif hasattr(self.language_model, 'generate'):
                response = self.language_model.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = str(self.language_model(prompt))
            
            return response
        except Exception as e:
            logger.error(f"Language model error: {e}")
            return f"[Language model error: {str(e)}]"
    
    async def _generate_math_response(
        self,
        query: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using math/code model"""
        if self.math_model is None:
            return await self._generate_language_response(
                query, context="This is a math/code question. Please solve it step by step.",
                temperature=temperature, max_tokens=max_tokens
            )
        
        try:
            if hasattr(self.math_model, 'generate_async'):
                response = await self.math_model.generate_async(
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif hasattr(self.math_model, 'generate'):
                response = self.math_model.generate(
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = str(self.math_model(query))
            
            return response
        except Exception as e:
            logger.error(f"Math model error: {e}")
            return await self._generate_language_response(
                query, context="This is a math/code question.",
                temperature=temperature, max_tokens=max_tokens
            )
    
    async def process(
        self,
        request: BrainRequest,
        user_id: str,
        conversation_id: Optional[UUID] = None
    ) -> BrainResponse:
        """
        Process a request through the multi-brain system.
        
        Args:
            request: The brain request
            user_id: User identifier
            conversation_id: Optional conversation ID
        
        Returns:
            BrainResponse with the generated response and metadata
        """
        start_time = time.time()
        query = request.message
        
        self.total_queries += 1
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")
        
        neuro_state = self._get_neurochemistry_state(user_id)
        
        if self.config.use_neurochemistry:
            self.router.modulate_from_neurochemistry(neuro_state)
            self.learner.modulate_from_neurochemistry(neuro_state)
        
        embedding = self._get_embedding(query)
        
        biased_embedding, memory_result = self._apply_memory_bias(embedding, user_id)
        
        if request.force_regions:
            router_output = RouterOutput(
                activations={
                    "language": 1.0 if BrainRegion.LANGUAGE in request.force_regions else 0.0,
                    "math": 1.0 if BrainRegion.MATH in request.force_regions else 0.0,
                    "memory": 1.0 if BrainRegion.MEMORY in request.force_regions else 0.0
                },
                active_regions=[
                    BrainRegionType.LANGUAGE if BrainRegion.LANGUAGE in request.force_regions else None,
                    BrainRegionType.MATH if BrainRegion.MATH in request.force_regions else None,
                    BrainRegionType.MEMORY if BrainRegion.MEMORY in request.force_regions else None
                ],
                confidence=1.0,
                hidden_activations=np.zeros(128),
                output_raw=np.zeros(3),
                routing_time_ms=0.0,
                query_hash=""
            )
            router_output.active_regions = [r for r in router_output.active_regions if r is not None]
        else:
            router_output = self.router.forward(biased_embedding, query)
        
        if memory_result.suggested_regions and memory_result.confidence > 0.5:
            for suggested in memory_result.suggested_regions:
                if suggested == "math" and BrainRegionType.MATH not in router_output.active_regions:
                    if router_output.activations["math"] > 0.3:
                        router_output.active_regions.append(BrainRegionType.MATH)
                elif suggested == "language" and BrainRegionType.LANGUAGE not in router_output.active_regions:
                    router_output.active_regions.append(BrainRegionType.LANGUAGE)
        
        regions_used = self._map_router_to_brain_regions(router_output.active_regions)
        
        for region in regions_used:
            self.region_usage[region] = self.region_usage.get(region, 0) + 1
        
        thinking_mode = request.thinking_mode or self.config.default_thinking_mode
        
        response_text = ""
        
        if BrainRegion.MATH in regions_used and BrainRegion.LANGUAGE in regions_used:
            math_response = await self._generate_math_response(
                query, temperature=0.3, max_tokens=request.max_tokens
            )
            synthesis_prompt = f"""The math/code specialist provided this solution:

{math_response}

Please synthesize this into a clear, helpful response for the user's question: {query}"""
            
            response_text = await self._generate_language_response(
                synthesis_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        elif BrainRegion.MATH in regions_used:
            response_text = await self._generate_math_response(
                query,
                temperature=0.3,
                max_tokens=request.max_tokens
            )
        
        else:
            response_text = await self._generate_language_response(
                query,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        decision_id = router_output.decision_id
        message_id = uuid4()
        conv_id = conversation_id or uuid4()
        
        if request.store_in_memory and self.config.use_memory:
            self.hippocampus.store(
                user_id=user_id,
                query=query,
                embedding=embedding,
                regions_used=[r.value for r in regions_used],
                was_successful=True,
                reward=0.0
            )
        
        self.pending_decisions[decision_id] = PendingDecision(
            decision_id=decision_id,
            user_id=user_id,
            query=query,
            router_output=router_output,
            regions_used=regions_used,
            response=response_text,
            timestamp=datetime.now(),
            embedding=embedding
        )
        
        if len(self.pending_decisions) > 1000:
            oldest_keys = sorted(
                self.pending_decisions.keys(),
                key=lambda k: self.pending_decisions[k].timestamp
            )[:500]
            for key in oldest_keys:
                del self.pending_decisions[key]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BrainResponse(
            response=response_text,
            conversation_id=conv_id,
            message_id=message_id,
            regions_activated=regions_used,
            activations=RouterActivation(
                language=router_output.activations["language"],
                math=router_output.activations["math"],
                memory=router_output.activations["memory"]
            ),
            routing_confidence=router_output.confidence,
            thinking_mode=thinking_mode,
            tokens_used=len(response_text.split()),
            processing_time_ms=processing_time_ms,
            routing_time_ms=router_output.routing_time_ms,
            decision_id=UUID(decision_id),
            memories_retrieved=len(memory_result.memories),
            memory_influence
=memory_result.confidence * self.config.memory_influence_weight, neurochemistry=neuro_state if self.config.use_neurochemistry else None )

def process_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
    """
    Process user feedback and trigger learning.
    
    Args:
        feedback: User feedback on a response
    
    Returns:
        Dictionary with learning results
    """
    decision_id = str(feedback.decision_id)
    
    if decision_id not in self.pending_decisions:
        return {
            "success": False,
            "error": "Decision not found or expired"
        }
    
    decision = self.pending_decisions[decision_id]
    
    reward = (feedback.rating - 3) / 2.0
    
    reward_signal = RewardSignal(
        value=reward,
        reward_type=RewardType.EXPLICIT_FEEDBACK,
        source="user",
        decision_id=decision_id,
        timestamp=datetime.now()
    )
    
    if self.config.use_learning:
        neuro_state = self._get_neurochemistry_state(decision.user_id)
        self.learner.modulate_from_neurochemistry(neuro_state)
        
        if decision.router_output.hidden_activations is not None:
            learning_result = self.learner.apply_reward(
                reward_signal=reward_signal,
                pre_activation=decision.embedding,
                post_activation=decision.router_output.hidden_activations
            )
            
            self.router.learn(
                reward=reward,
                learning_rate_modifier=self.learner.get_learning_rate() / 0.01
            )
    
    if self.config.use_memory and decision.embedding is not None:
        self.hippocampus.update_memory_reward(
            embedding=decision.embedding,
            reward=reward,
            was_successful=(feedback.rating >= 3)
        )
    
    del self.pending_decisions[decision_id]
    self.total_feedbacks += 1
    
    return {
        "success": True,
        "decision_id": decision_id,
        "reward_applied": reward,
        "learning_triggered": self.config.use_learning,
        "router_decisions": self.router.total_decisions,
        "router_learning_events": self.router.total_learning_events
    }

def get_status(self) -> BrainStatus:
    """Get current brain system status"""
    uptime = (datetime.now() - self.start_time).total_seconds()
    
    router_status = RouterStatus(
        is_initialized=self.router.is_initialized,
        weights_version=1,
        total_neurons=len(self.router.hidden_neurons) + len(self.router.output_neurons),
        architecture=self.router.get_stats()["architecture"],
        total_decisions=self.router.total_decisions,
        accuracy_estimate=0.5,
        learning_enabled=self.config.use_learning
    )
    
    region_statuses = [
        BrainRegionStatus(
            region=BrainRegion.LANGUAGE,
            is_loaded=self.language_model is not None,
            is_healthy=self.language_model is not None,
            model_name="Mistral 7B" if self.language_model else None,
            parameters="7B" if self.language_model else None,
            total_calls=self.region_usage.get(BrainRegion.LANGUAGE, 0)
        ),
        BrainRegionStatus(
            region=BrainRegion.MATH,
            is_loaded=self.math_model is not None,
            is_healthy=self.math_model is not None,
            model_name="DeepSeek Coder" if self.math_model else None,
            parameters="6.7B" if self.math_model else None,
            total_calls=self.region_usage.get(BrainRegion.MATH, 0)
        ),
        BrainRegionStatus(
            region=BrainRegion.MEMORY,
            is_loaded=True,
            is_healthy=True,
            model_name="Hippocampus",
            total_calls=self.region_usage.get(BrainRegion.MEMORY, 0)
        )
    ]
    
    hippo_stats = self.hippocampus.get_stats()
    neuro_state = None
    if self.neurochemistry_system is not None:
        neuro_state = self._get_neurochemistry_state("system")
    
    return BrainStatus(
        router=router_status,
        regions=region_statuses,
        hippocampus_connected=True,
        hippocampus_memories=hippo_stats.get("total_memories", 0),
        neurochemistry_connected=self.neurochemistry_system is not None,
        neurochemistry_state=neuro_state,
        uptime_seconds=uptime
    )

def get_stats(self) -> Dict[str, Any]:
    """Get orchestrator statistics"""
    return {
        "total_queries": self.total_queries,
        "total_feedbacks": self.total_feedbacks,
        "pending_decisions": len(self.pending_decisions),
        "region_usage": {k.value: v for k, v in self.region_usage.items()},
        "router_stats": self.router.get_stats(),
        "learner_stats": {
            "base_learning_rate": self.learner.base_learning_rate,
            "current_learning_rate": self.learner.get_learning_rate()
        },
        "hippocampus_stats": self.hippocampus.get_stats(),
        "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
    }
_global_orchestrator: Optional[BrainOrchestrator] = None

def get_brain_orchestrator() -> BrainOrchestrator: """Get or create the global brain orchestrator""" global _global_orchestrator if _global_orchestrator is None: _global_orchestrator = BrainOrchestrator() return _global_orchestrator

def create_brain_orchestrator( language_model: Any = None, math_model: Any = None, neurochemistry_system: Any = None, config: Optional[BrainConfig] = None ) -> BrainOrchestrator: """ Create a new brain orchestrator.

Args:
    language_model: Language model service
    math_model: Math model service  
    neurochemistry_system: Neurochemistry system
    config: Brain configuration

Returns:
    Configured BrainOrchestrator instance
"""
global _global_orchestrator
_global_orchestrator = BrainOrchestrator(
    config=config,
    language_model=language_model,
    math_model=math_model,
    neurochemistry_system=neurochemistry_system
)
return _global_orchestrator
