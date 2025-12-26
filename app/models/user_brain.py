"""
User Brain Profile Models - Database persistence for user-specific brain parameters
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class UserBiasCreate(BaseModel):
    """Create new user bias profile"""
    user_id: UUID
    code_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    math_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    physics_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    language_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    memory_bias: float = Field(default=0.0, ge=-1.0, le=1.0)


class UserBiasDB(BaseModel):
    """User bias profile from database"""
    user_id: UUID
    
    # Biases: -1.0 (avoid) to +1.0 (prefer)
    code_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    math_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    physics_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    language_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    memory_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    # Ranges: How much bias affects threshold (0.1 to 1.0)
    code_range: float = Field(default=0.3, ge=0.1, le=1.0)
    math_range: float = Field(default=0.3, ge=0.1, le=1.0)
    physics_range: float = Field(default=0.3, ge=0.1, le=1.0)
    language_range: float = Field(default=0.3, ge=0.1, le=1.0)
    memory_range: float = Field(default=0.3, ge=0.1, le=1.0)
    
    # Learning progress
    maturity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_sessions: int = 0
    total_queries: int = 0
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_session_at: Optional[datetime] = None
    
    def to_bias_dict(self) -> Dict[str, float]:
        return {
            "code": self.code_bias,
            "math": self.math_bias,
            "physics": self.physics_bias,
            "language": self.language_bias,
            "memory": self.memory_bias
        }
    
    def to_range_dict(self) -> Dict[str, float]:
        return {
            "code": self.code_range,
            "math": self.math_range,
            "physics": self.physics_range,
            "language": self.language_range,
            "memory": self.memory_range
        }


class UserThresholdDB(BaseModel):
    """User-specific area thresholds from database"""
    id: Optional[int] = None
    user_id: UUID
    area: str  # code, math, physics, language, memory
    
    # Current threshold (starts low at 0.1 for learning)
    threshold: float = Field(default=0.1, ge=0.05, le=0.95)
    
    # Stats for learning
    activations: int = 0
    useful_activations: int = 0
    skipped_count: int = 0  # Times PFC skipped this area's output
    
    # Time-based decay tracking
    last_activated: Optional[datetime] = None
    last_decay_applied: Optional[datetime] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def usefulness_ratio(self) -> float:
        if self.activations == 0:
            return 0.5  # Default
        return self.useful_activations / self.activations


class SessionSummary(BaseModel):
    """Summary of a user session for bias learning"""
    user_id: UUID
    session_id: UUID
    
    # Query counts by detected intent
    code_queries: int = 0
    math_queries: int = 0
    physics_queries: int = 0
    language_queries: int = 0
    emotional_queries: int = 0
    
    # What was useful
    code_useful: int = 0
    math_useful: int = 0
    physics_useful: int = 0
    language_useful: int = 0
    memory_useful: int = 0
    
    # What was skipped by PFC
    code_skipped: int = 0
    math_skipped: int = 0
    physics_skipped: int = 0
    memory_skipped: int = 0
    
    # Duration
    started_at: datetime
    ended_at: Optional[datetime] = None
    query_count: int = 0
    
    def calculate_bias_updates(self) -> Dict[str, float]:
        """Calculate bias adjustments based on session"""
        updates = {}
        
        # Code bias: positive if code was often useful, negative if often skipped
        if self.code_queries > 0:
            useful_ratio = self.code_useful / max(1, self.code_queries)
            skip_ratio = self.code_skipped / max(1, self.code_queries)
            updates["code"] = (useful_ratio - skip_ratio) * 0.1
        
        # Math bias
        if self.math_queries > 0:
            useful_ratio = self.math_useful / max(1, self.math_queries)
            skip_ratio = self.math_skipped / max(1, self.math_queries)
            updates["math"] = (useful_ratio - skip_ratio) * 0.1
        
        # Language is almost always useful, slight positive bias
        if self.language_queries > 0:
            useful_ratio = self.language_useful / max(1, self.language_queries)
            updates["language"] = useful_ratio * 0.05
        
        return updates


class GateEvaluationLog(BaseModel):
    """Log of a gate evaluation for debugging/analysis"""
    id: Optional[int] = None
    user_id: UUID
    query_hash: str
    
    # Input signals from thalamus
    raw_signals: Dict[str, float]
    
    # User modifiers applied
    bias_modifiers: Dict[str, float]
    neuro_modifiers: Dict[str, float]
    decay_modifiers: Dict[str, float]
    
    # Final thresholds used
    final_thresholds: Dict[str, float]
    
    # Result
    gates_opened: List[str]
    gates_closed: List[str]
    
    # Timing
    created_at: Optional[datetime] = None


class UserBrainProfile(BaseModel):
    """Complete user brain profile (aggregate view)"""
    user_id: UUID
    
    # Bias profile
    bias: UserBiasDB
    
    # Per-area thresholds
    thresholds: Dict[str, UserThresholdDB]
    
    # Current neurochemical state (from session)
    neuro_state: Optional[Dict[str, float]] = None
    
    # Stats
    total_queries: int = 0
    total_sessions: int = 0
    
    # Last activity
    last_active: Optional[datetime] = None
    
    def get_threshold(self, area: str) -> float:
        if area in self.thresholds:
            return self.thresholds[area].threshold
        return 0.1  # Default low threshold


class AreaOutputDB(BaseModel):
    """Stored area output for learning"""
    id: Optional[int] = None
    user_id: UUID
    query_hash: str
    area: str
    
    # Output
    output_text: str
    output_type: str  # "code", "latex", "text", "memory"
    
    # PFC evaluation
    relevance_score: float = Field(default=0.85, ge=0.0, le=1.0)
    was_included: bool = True
    was_skipped: bool = False
    
    # User feedback (if any)
    user_rating: Optional[int] = None  # 1-5
    
    # Timing
    execution_time_ms: float = 0.0
    created_at: Optional[datetime] = None
