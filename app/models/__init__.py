"""
Pydantic models for Sarah AI / OMNIUS
"""
from .auth import (
    GenderType,
    UserRegister,
    UserLogin,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
    Token,
    User,
    UserProfile,
    GoogleAuth,
    PasswordChangeRequest
)

from .chat import (
    PersonalityType,
    ChatMessage,
    ChatResponse,
    Conversation,
    Message
)

from .theme import (
    ThemeAction,
    ThemeSwitchRequest,
    ThemePreferencesUpdate,
    ThemeResponse
)

from .omnius import (
    ConsciousnessRegion,
    OmniusRequest,
    ConsciousnessStatus,
    OmniusResponse
)

from .brain import (
    BrainRegion,
    ThinkingMode,
    RouterActivation,
    RouterState,
    RoutingDecision,
    RewardSignal,
    LearningEvent,
    NeuralMemory,
    MemoryRetrievalResult,
    BrainRequest,
    BrainResponse,
    FeedbackRequest,
    FeedbackResponse,
    BrainRegionStatus,
    RouterStatus,
    BrainStatus
)

__all__ = [
    # Auth
    "GenderType",
    "UserRegister",
    "UserLogin",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "VerifyEmailRequest",
    "Token",
    "User",
    "UserProfile",
    "GoogleAuth",
    "PasswordChangeRequest",
    
    # Chat
    "PersonalityType",
    "ChatMessage",
    "ChatResponse",
    "Conversation",
    "Message",
    
    # Theme
    "ThemeAction",
    "ThemeSwitchRequest",
    "ThemePreferencesUpdate",
    "ThemeResponse",
    
    # Omnius
    "ConsciousnessRegion",
    "OmniusRequest",
    "ConsciousnessStatus",
    "OmniusResponse",
    
    # Brain (Neural Router)
    "BrainRegion",
    "ThinkingMode",
    "RouterActivation",
    "RouterState",
    "RoutingDecision",
    "RewardSignal",
    "LearningEvent",
    "NeuralMemory",
    "MemoryRetrievalResult",
    "BrainRequest",
    "BrainResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "BrainRegionStatus",
    "RouterStatus",
    "BrainStatus"
]
