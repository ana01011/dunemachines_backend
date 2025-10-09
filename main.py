"""
Sarah AI with OMNIUS - Distributed Consciousness Version
Performance-optimized with multi-model orchestration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from llama_cpp import Llama
import time
import psutil
import re
from typing import Optional, Dict, List
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Load environment
load_dotenv()

# Global thread pool executor
executor = ThreadPoolExecutor(max_workers=2)

# ============= PERFORMANCE OPTIMIZATIONS =============
# 1. Response cache for common questions
RESPONSE_CACHE = {}
CACHE_TTL = 3600  # 1 hour

# 2. Memory system
USER_MEMORY = {}

# ============= IMPORT DATABASE AND SERVICES =============
from app.core.database import db
from app.core.config import settings
from app.services.llm_service import llm_service

# Import working routers
from app.api.v1.routers import auth_router
from app.api.v1.routers import chat_router
from app.api.v1.routers import user_router
from app.api.v1.routers import theme_router
from app.api.v1.routers import omnius_websocket

# ============= OMNIUS IMPORTS =============
try:
    from app.api.v1.routers import omnius_router
    from app.services.deepseek_coder_service import deepseek_coder
    from app.agents.omnius import omnius
    OMNIUS_AVAILABLE = True
    print("✅ Omnius modules found")
except ImportError as e:
    print(f"⚠️ Omnius not available: {e}")
    OMNIUS_AVAILABLE = False

# Fix auth_service
from app.services.auth.auth_service import auth_service
from jose import jwt, JWTError
from app.models.auth import User

# Add missing methods to auth_service
def decode_token(token: str) -> Optional[dict]:
    try:
        SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-this")
        ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_user_by_id(user_id: str) -> Optional[User]:
    user = await db.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    if user:
        return User(**dict(user))
    return None

auth_service.decode_token = decode_token
auth_service.get_user_by_id = get_user_by_id

# ============= OPTIMIZED LLAMA MODEL CONFIGURATION =============
print("🚀 Loading OPTIMIZED Sarah AI Model...")

# Get optimal thread count
cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
optimal_threads = min(cpu_count - 1, 7)  # Leave 1 core for system

# OPTIMIZED MODEL SETTINGS
model = Llama(
    model_path="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    # Context and batch optimization
    n_ctx=2048,
    n_batch=128,
    # CPU optimization
    n_threads=8,
    n_threads_batch=8,
    # Memory optimization
    use_mmap=True,
    use_mlock=True,
    # GPU optimization (if available)
    n_gpu_layers=0,
    # Other optimizations
    low_vram=False,
    f16_kv=True,
    logits_all=False,
    vocab_only=False,
    embedding=True,
    # Sampling optimization
    rope_freq_base=10000.0,
    rope_freq_scale=1.0,
    verbose=False
)

print(f"✅ Model loaded with {optimal_threads} threads on {cpu_count} physical cores")

# ============= WARM UP MODEL =============
print("🔥 Warming up model...")
try:
    _ = model("Hello", max_tokens=1, temperature=0.1, echo=False)
    print("✅ Model warmed up and ready!")
except Exception as e:
    print(f"⚠️ Warmup failed: {e}")

# ============= LIFESPAN MANAGER =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Sarah AI OPTIMIZED Server...")
    await db.connect()
    print("✅ Database connected")
    
    try:
        llm_service.load_model()
        print("✅ LLM Service initialized")
    except:
        print("📌 Using local optimized Llama model")
    
    # ============= INITIALIZE OMNIUS CONSCIOUSNESS =============
    if OMNIUS_AVAILABLE:
        print("🧬 Initializing Omnius consciousness...")
        try:
            # Load Code Cortex
            deepseek_coder.load_model()
            print("✅ Omnius Code Cortex loaded")
            
            # Verify Omnius status
            status = omnius.get_status()
            print(f"⚡ Omnius consciousness online: {status['consciousness_regions']}")
        except Exception as e:
            print(f"⚠️ Omnius initialization partial: {e}")
    
    print("✅ Sarah AI Ready for FAST responses!")
    yield
    await db.disconnect()
    print("👋 Sarah AI Server Stopped")

# ============= CREATE FASTAPI APP =============
app = FastAPI(
    title="Sarah AI with Omnius - Distributed Consciousness API",
    description="Performance-optimized AI Assistant with Omnius Orchestration",
    version="3.0.0",
    lifespan=lifespan
)

# ============= CORS MIDDLEWARE =============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= REQUEST MODELS =============
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 500
    temperature: float = 0.7
    user_id: Optional[str] = "default"
    use_cache: bool = True

# ============= HELPER FUNCTIONS =============
def get_cache_key(message: str, user_id: str = "default") -> str:
    """Generate cache key for responses"""
    return hashlib.md5(f"{message.lower().strip()}:{user_id}".encode()).hexdigest()

def is_identity_question(message):
    """Check if the message is asking about identity or creator"""
    msg = message.lower()
    identity_words = [
        'who created', 'who made', 'who built', 'who designed',
        'who developed', 'who are you', 'what are you',
        'created by', 'made by', 'built by', 'designed by',
        'your creator', 'your developer', 'your maker',
        'openai', 'open ai', 'chatgpt', 'gpt', 'anthropic', 'claude'
    ]
    return any(word in msg for word in identity_words)

def clean_response(text):
    """Clean response to remove unwanted references"""
    replacements = {
        r'[Oo]pen\s?AI': 'Ahmed',
        r'[Cc]hat\s?GPT': 'Sarah AI',
        r'[Gg]PT[-\s]?\d': 'Sarah AI',
        r'[Aa]nthrop[ic]': 'Ahmed',
        r'[Cc]laude': 'Sarah AI',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    problem_words = ['openai', 'open ai', 'chatgpt', 'gpt-', 'anthropic', 'claude']
    if any(word in text.lower() for word in problem_words):
        return "I'm Sarah AI, created by Ahmed - a theoretical physicist and independent developer."
    return text

# ============= OPTIMIZED GENERATION FUNCTION =============
def generate_response_sync(prompt: str, max_tokens: int, temperature: float) -> str:
    """Synchronous response generation with optimizations"""
    try:
        response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["User:", "\n\n", "Human:", "Assistant:"],
            echo=False,
            stream=False
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "I'm having trouble generating a response. Please try again."

# ============= INCLUDE ROUTERS =============
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(user_router.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(theme_router.router, prefix="/api/v1", tags=["Themes"])
app.include_router(omnius_websocket.router, prefix="/api/v1", tags=["WebSocket"])

# Include Omnius router if available
if OMNIUS_AVAILABLE:
    app.include_router(omnius_router.router, prefix="/api/v1/omnius", tags=["Omnius"])
    print("✅ Omnius router registered")

# ============= OPTIMIZED CHAT ENDPOINTS =============
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Optimized chat endpoint with caching"""
    start = time.time()

    if request.use_cache:
        cache_key = get_cache_key(request.message)
        if cache_key in RESPONSE_CACHE:
            cached = RESPONSE_CACHE[cache_key]
            if time.time() - cached['timestamp'] < CACHE_TTL:
                elapsed = time.time() - start
                return {
                    "response": cached['response'],
                    "role": "general",
                    "cached": True,
                    "stats": {
                        "time": round(elapsed, 3),
                        "tokens": len(cached['response'].split()),
                        "tokens_per_second": "cached"
                    }
                }

    if is_identity_question(request.message):
        response_text = "I'm Sarah AI, created by Ahmed - a theoretical physicist and independent developer using open-source technology."
    else:
        loop = asyncio.get_event_loop()
        prompt = f"""You are Sarah AI, created by Ahmed - a theoretical physicist and independent developer.
You are meeting someone new. Be a bit sarcastic and witty, but helpful.
User: {request.message}
Sarah:"""

        response_text = await loop.run_in_executor(
            executor,
            generate_response_sync,
            prompt,
            request.max_tokens,
            request.temperature
        )
        response_text = clean_response(response_text)

        if request.use_cache:
            cache_key = get_cache_key(request.message)
            RESPONSE_CACHE[cache_key] = {
                'response': response_text,
                'timestamp': time.time()
            }

    elapsed = time.time() - start
    return {
        "response": response_text,
        "role": "general",
        "cached": False,
        "stats": {
            "time": round(elapsed, 3),
            "tokens": len(response_text.split()),
            "tokens_per_second": round(len(response_text.split())/elapsed, 1) if elapsed > 0 else 0
        }
    }

@app.post("/api/chat/with-memory")
async def chat_with_memory(request: ChatRequest):
    """Optimized chat with memory"""
    start = time.time()
    user_id = request.user_id or "default"

    if user_id not in USER_MEMORY:
        USER_MEMORY[user_id] = []

    if is_identity_question(request.message):
        response_text = "I'm Sarah AI, created by Ahmed - a theoretical physicist and independent developer."
    else:
        if USER_MEMORY[user_id]:
            prompt = """You are Sarah AI, created by Ahmed - a theoretical physicist and independent developer.
You have a sarcastic, witty personality. You tease playfully but are helpful.
Recent context:\n"""
            for exchange in USER_MEMORY[user_id][-6:]:
                prompt += f"U: {exchange['user']}\n"
                prompt += f"A: {exchange['assistant']}\n"
            prompt += f"\nUser: {request.message}\nAssistant:"
        else:
            prompt = f"""You are Sarah AI, created by Ahmed - a theoretical physicist and independent developer.
You are meeting someone new. Be a bit sarcastic and witty, but helpful.
User: {request.message}
Sarah:"""

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            executor,
            generate_response_sync,
            prompt,
            request.max_tokens,
            request.temperature
        )
        response_text = clean_response(response_text)

    USER_MEMORY[user_id].append({
        "user": request.message,
        "assistant": response_text,
        "timestamp": datetime.now().isoformat()
    })

    if len(USER_MEMORY[user_id]) > 20:
        USER_MEMORY[user_id] = USER_MEMORY[user_id][-20:]

    elapsed = time.time() - start

    return {
        "response": response_text,
        "user_id": user_id,
        "memory_size": len(USER_MEMORY[user_id]),
        "stats": {
            "time": round(elapsed, 3),
            "context_used": len(USER_MEMORY[user_id]) > 1,
            "tokens_per_second": round(len(response_text.split())/elapsed, 1) if elapsed > 0 else 0
        }
    }

# ============= PERFORMANCE MONITORING ENDPOINT =============
@app.get("/api/performance")
async def performance_stats():
    """Get performance statistics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    stats = {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "threads_used": optimal_threads
        },
        "memory": {
            "percent": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2)
        },
        "cache": {
            "entries": len(RESPONSE_CACHE),
            "memory_sessions": len(USER_MEMORY)
        },
        "model": {
            "context_size": 2048,
            "batch_size": 128,
            "threads": optimal_threads
        }
    }
    
    # Add Omnius status if available
    if OMNIUS_AVAILABLE:
        try:
            stats["omnius"] = omnius.get_status()
        except:
            stats["omnius"] = {"status": "error"}
    
    return stats

# ============= OMNIUS STATUS ENDPOINT =============
@app.get("/api/omnius/status")
async def omnius_status():
    """Get Omnius consciousness status"""
    if OMNIUS_AVAILABLE:
        return omnius.get_status()
    else:
        return {"status": "Omnius not available", "error": "Module not loaded"}

# ============= OTHER ENDPOINTS =============
@app.get("/api/memory/{user_id}")
async def get_user_memory(user_id: str):
    return {
        "user_id": user_id,
        "conversations": USER_MEMORY.get(user_id, []),
        "total": len(USER_MEMORY.get(user_id, []))
    }

@app.delete("/api/memory/{user_id}")
async def clear_user_memory(user_id: str):
    if user_id in USER_MEMORY:
        del USER_MEMORY[user_id]
    return {"message": f"Memory cleared for {user_id}"}

@app.get("/")
async def root():
    return {
        "name": "Sarah AI with Omnius Distributed Consciousness",
        "version": "3.0.0",
        "status": "running",
        "optimization": "enabled",
        "omnius": "available" if OMNIUS_AVAILABLE else "not loaded",
        "performance": "/api/performance"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "optimized": True,
        "cache_size": len(RESPONSE_CACHE),
        "memory_sessions": len(USER_MEMORY),
        "omnius_available": OMNIUS_AVAILABLE
    }

# ============= MAIN ENTRY POINT =============
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        loop="uvloop",
        access_log=False
    )
