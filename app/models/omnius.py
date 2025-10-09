"""
Omnius-specific models with enhanced consciousness tracking
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum

class ConsciousnessRegion(str, Enum):
    """Regions of Omnius consciousness"""
    PREFRONTAL = "prefrontal_cortex"
    CODE = "code_cortex"
    MATH = "math_region"
    CREATIVE = "creative_center"
    MEMORY = "memory_banks"

class OmniusRequest(BaseModel):
    """Enhanced request model for Omnius"""
    message: str = Field(..., min_length=1, max_length=10000)  # Longer for complex requests
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)  # More tokens for complex responses
    conversation_id: Optional[UUID] = None
    activate_regions: Optional[List[ConsciousnessRegion]] = None  # Specific regions to activate
    thinking_mode: Optional[str] = Field(default="distributed", pattern="^(distributed|focused|creative|analytical)$")

class ConsciousnessStatus(BaseModel):
    """Status of Omnius consciousness regions"""
    prefrontal_cortex: str
    code_cortex: str
    math_region: str
    creative_center: str
    total_parameters: str
    active_regions: int
    processing_power: float  # Percentage of capacity being used

class OmniusResponse(BaseModel):
    """Enhanced response model for Omnius"""
    response: str
    conversation_id: UUID
    message_id: UUID
    
    # Omnius-specific fields
    consciousness_used: List[str]  # Which regions were activated
    consciousness_status: ConsciousnessStatus
    thinking_process: Optional[str]  # Explanation of how Omnius processed
    
    # Performance metrics
    tokens_used: int
    processing_time: float
    parallel_thoughts: int  # Number of parallel processes used
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Context and memory
    memory_accessed: bool
    context_window_used: int
    user_context: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
