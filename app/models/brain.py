"""
Neural Router and Multi-Brain System models
OMNIUS v2 - Biological Neural Router
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ============================================
# BRAIN REGIONS
# ============================================

class BrainRegion(str, Enum):
    """Available brain regions (specialist models)"""
    LANGUAGE = "language"      # Mistral 7B - General language
    MATH = "math"              # DeepSeek - Math/Code
    MEMORY = "memory"          # Hippocampus - Retrieval
    # Future regions:
    # CREATIVE = "creative"
    # VISION = "vision"


class ThinkingMode(str, Enum):
    """How the brain processes"""
    FAST = "fast"              # Single region, quick response
    DELIBERATE = "deliberate"  # Multiple regions, thorough
    CREATIVE = "creative"      # Exploratory, divergent
    ANALYTICAL = "analytical"  # Focused, convergent


# ============================================
# NEURAL ROUTER
# ============================================

class RouterActivation(BaseModel):
    """Activation levels for each brain region"""
    language: float = Field(default=0.0, ge=0.0, le=1.0)
    math: float = Field(default=0.0, ge=0.0, le=1.0)
    memory: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def get_active_regions(self, threshold: float = 0.5) -> List[BrainRegion]:
        """Get regions above activation threshold"""
        active = []
        if self.language >= threshold:
            active.append(BrainRegion.LANGUAGE)
        if self.math >= threshold:
            active.append(BrainRegion.MATH)
        if self.memory >= threshold:
            active.append(BrainRegion.MEMORY)
        return active
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "language": self.language,
            "math": self.math,
            "memory": self.memory
        }


class RouterState(BaseModel):
    """Current state of the neural router"""
    id: Optional[UUID] = None
    user_id: Optional[UUID] = None  # None for global router
    weights_version: int = 1
    total_decisions: int = 0
    accuracy_estimate: float = 0.5
    last_updated: Optional[datetime] = None
    
    # Neurochemistry influence
    dopamine_level: float = Field(default=0.5, ge=0.0, le=1.0)
    learning_rate_modifier: float = Field(default=1.0, ge=0.1, le=2.0)


class RoutingDecision(BaseModel):
    """A single routing decision"""
    id: Optional[UUID] = None
    user_id: UUID
    conversation_id: Optional[UUID] = None
    
    # Query info
    query_hash: str  # SHA256 hash of the query
    query_length: int
    query_preview: Optional[str] = None  # First 200 chars
    
    # Decision
    activations: RouterActivation
    regions_used: List[BrainRegion]
    thinking_mode: ThinkingMode = ThinkingMode.FAST
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Timing
    routing_time_ms: float = 0.0
    created_at: Optional[datetime] = None
    
    # Outcome (filled later via feedback)
    reward: Optional[float] = None
    feedback_received: bool = False
    was_successful: Optional[bool] = None


# ============================================
# LEARNING
# ============================================

class RewardSignal(BaseModel):
    """Reward signal for learning"""
    decision_id: UUID
    reward_type: str  # 'explicit_feedback', 'implicit_success', 'completion', 'error'
    reward_value: float = Field(ge=-1.0, le=1.0)
    source: str  # 'user', 'system', 'task'
    timestamp: Optional[datetime] = None


class LearningEvent(BaseModel):
    """A learning update event"""
    id: Optional[UUID] = None
    user_id: Optional[UUID] = None  # None for global learning
    
    # Trigger
    decision_id: UUID
    reward_type: str
    reward_value: float
    
    # Update info
    learning_rate_used: float
    weight_change_magnitude: float
    
    # Neurochemistry state at time of learning
    dopamine_level: Optional[float] = None
    serotonin_level: Optional[float] = None
    cortisol_level: Optional[float] = None
    
    # Accuracy tracking
    pre_accuracy: Optional[float] = None
    post_accuracy: Optional[float] = None
    
    created_at: Optional[datetime] = None


# ============================================
# HIPPOCAMPUS (MEMORY)
# ============================================

class NeuralMemory(BaseModel):
    """Memory stored in hippocampus for routing assistance"""
    id: Optional[UUID] = None
    user_id: UUID
    
    # What was stored
    embedding_id: str  # Reference to vector DB (ChromaDB)
    query_summary: str  # Short text summary (max 500 chars)
    
    # Routing outcome that was stored
    regions_used: List[BrainRegion]
    was_successful: bool
    reward_received: Optional[float] = None
    
    # Retrieval tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    # Importance for memory consolidation
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryRetrievalResult(BaseModel):
    """Result from hippocampus query"""
    memories: List[NeuralMemory]
    similarity_scores: List[float]
    suggested_regions: List[BrainRegion]
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0


# ============================================
# API REQUEST/RESPONSE MODELS
# ============================================

class BrainRequest(BaseModel):
    """Request to the multi-brain system"""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[UUID] = None
    
    # Optional overrides
    force_regions: Optional[List[BrainRegion]] = None  # Force specific regions
    thinking_mode: Optional[ThinkingMode] = None
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    
    # Memory settings
    use_memory: bool = True
    store_in_memory: bool = True


class BrainResponse(BaseModel):
    """Response from the multi-brain system"""
    response: str
    conversation_id: UUID
    message_id: UUID
    
    # Routing info
    regions_activated: List[BrainRegion]
    activations: RouterActivation
    routing_confidence: float = Field(ge=0.0, le=1.0)
    thinking_mode: ThinkingMode
    
    # Performance metrics
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    routing_time_ms: float = 0.0
    
    # For providing feedback later
    decision_id: UUID
    
    # Memory info
    memories_retrieved: int = 0
    memory_influence: float = 0.0  # How much memory affected routing (0-1)
    
    # Neurochemistry state (optional)
    neurochemistry: Optional[Dict[str, float]] = None


class FeedbackRequest(BaseModel):
    """User feedback on a response for learning"""
    decision_id: UUID
    rating: int = Field(ge=1, le=5)  # 1-5 stars
    feedback_type: str = "rating"  # 'rating', 'correction', 'preference'
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response after processing feedback"""
    decision_id: UUID
    reward_applied: float
    learning_triggered: bool
    message: str = "Feedback received"


# ============================================
# BRAIN STATUS / DIAGNOSTICS
# ============================================

class BrainRegionStatus(BaseModel):
    """Status of a single brain region"""
    region: BrainRegion
    is_loaded: bool = False
    is_healthy: bool = False
    model_name: Optional[str] = None
    parameters: Optional[str] = None  # e.g., "7B"
    last_used: Optional[datetime] = None
    total_calls: int = 0
    avg_response_time_ms: float = 0.0


class RouterStatus(BaseModel):
    """Status of the neural router"""
    is_initialized: bool = False
    weights_version: int = 0
    total_neurons: int = 0
    architecture: str = ""  # e.g., "768->128->3"
    total_decisions: int = 0
    accuracy_estimate: float = 0.5
    learning_enabled: bool = True


class BrainStatus(BaseModel):
    """Overall brain system status"""
    router: RouterStatus
    regions: List[BrainRegionStatus]
    hippocampus_connected: bool = False
    hippocampus_memories: int = 0
    neurochemistry_connected: bool = False
    neurochemistry_state: Optional[Dict[str, float]] = None
    uptime_seconds: float = 0.0
