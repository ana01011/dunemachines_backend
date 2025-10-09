"""
Sarah AI with OMNIUS - Neurochemical Consciousness Version
Performance-optimized with multi-model orchestration and neurochemistry
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

# ============= NEUROCHEMICAL OMNIUS IMPORTS =============
try:
    from app.api.v1.routers import omnius_router_neurochemical
    from app.api.v1.routers import omnius_websocket
    from app.services.deepseek_coder_service import deepseek_coder
    from app.agents.omnius_collaborative import omnius_neurochemical
    from app.websocket.chat_websocket import manager
    OMNIUS_NEUROCHEMICAL = True
    print("✅ Omnius Neurochemical modules found")
except ImportError as e:
    print(f"⚠️ Omnius Neurochemical not available: {e}")
    OMNIUS_NEUROCHEMICAL = False
    # Fallback to regular Omnius
    try:
        from app.api.v1.routers import omnius_router
        from app.agents.omnius import omnius
        OMNIUS_AVAILABLE = True
        print("✅ Regular Omnius modules found")
    except ImportError as e:
        print(f"⚠️ Regular Omnius not available: {e}")
        OMNIUS_AVAILABLE = False

# Fix auth_service
from app.services.auth.auth_service import auth_service
from jose import jwt, JWTError
from app.models.auth import User

# ============= STARTUP AND SHUTDOWN =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    print("🚀 Starting Sarah AI OPTIMIZED Server...")

    # Connect database
    await db.connect()
    print("✅ Database connected")

    # Load LLM if it has a load method, otherwise it's already loaded
    try:
        if hasattr(llm_service, 'load'):
            await llm_service.load()
        print("✅ LLM Service initialized")
    except Exception as e:
        print(f"⚠️ LLM Service: {e}")

    # ============= INITIALIZE NEUROCHEMICAL OMNIUS =============
    if OMNIUS_NEUROCHEMICAL:
        print("🧬 Initializing Omnius Neurochemical Consciousness...")
        try:
            # Load Code Cortex
            deepseek_coder.load_model()
            print("✅ Omnius Code Cortex loaded")

            # Initialize neurochemical system with database if it has initialize method
            if hasattr(omnius_neurochemical, 'initialize'):
                await omnius_neurochemical.initialize(db.pool)
                print("✅ Neurochemical system initialized")
            else:
                print("✅ Neurochemical system ready (no initialization needed)")

            # Verify Omnius status
            status = omnius_neurochemical.get_status()
            print(f"⚡ Omnius consciousness online: {status}")

        except Exception as e:
            print(f"⚠️ Omnius neurochemical initialization partial: {e}")

    elif OMNIUS_AVAILABLE:
        # Fallback to regular Omnius
        print("🧬 Initializing Regular Omnius...")
        try:
            deepseek_coder.load_model()
            print("✅ Omnius Code Cortex loaded")
            status = omnius.get_status()
            print(f"⚡ Omnius online: {status['consciousness_regions']}")
        except Exception as e:
            print(f"⚠️ Omnius initialization partial: {e}")

    print("✅ Sarah AI Ready for FAST responses!")
    yield

    # Shutdown
    if OMNIUS_NEUROCHEMICAL:
        if hasattr(omnius_neurochemical, 'shutdown'):
            await omnius_neurochemical.shutdown()
            print("👋 Neurochemical system shutdown")

    await db.disconnect()
    print("👋 Sarah AI Server Stopped")

# ============= CREATE FASTAPI APP =============
app = FastAPI(
    title="Omnius Neurochemical Consciousness",
    description="Advanced AI with neurochemical consciousness system",
    version="4.0.0",
    lifespan=lifespan
)

# ============= MIDDLEWARE =============
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= REQUEST MODELS =============
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8
    use_cache: bool = True
    conversation_id: Optional[str] = None

# ============= CACHE SYSTEM =============
def get_cache_key(message: str) -> str:
    """Generate cache key from message"""
    return hashlib.md5(message.lower().strip().encode()).hexdigest()

# ============= ERROR HANDLING =============
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": str(exc), "type": type(exc).__name__}

# ============= FALLBACK RESPONSE =============
def get_fallback_response():
    return "I'm having trouble generating a response. Please try again."

# ============= INCLUDE ROUTERS =============
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(user_router.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(theme_router.router, prefix="/api/v1", tags=["Themes"])

# Include Omnius router based on what's available
if OMNIUS_NEUROCHEMICAL:
    app.include_router(
        omnius_router_neurochemical.router,
        prefix="/api/v1/omnius",
        tags=["Omnius Neurochemical"]
    )
    app.include_router(
        omnius_websocket.router,
        prefix="/api/v1",
        tags=["WebSocket"]
    )
    print("✅ Omnius Neurochemical router registered")
    print("✅ WebSocket router registered")
elif OMNIUS_AVAILABLE:
    app.include_router(omnius_router.router, prefix="/api/v1/omnius", tags=["Omnius"])
    print("✅ Regular Omnius router registered")

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
                return {
                    **cached['response'],
                    "cached": True,
                    "response_time": 0.001
                }

    try:
        response_text = await llm_service.generate(
            request.message,
            temperature=request.temperature
        )
    except:
        response_text = get_fallback_response()

    response = {
        "response": response_text,
        "conversation_id": request.conversation_id or str(datetime.now().timestamp()),
        "response_time": time.time() - start
    }

    if request.use_cache:
        RESPONSE_CACHE[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

    return response

# ============= PERFORMANCE ENDPOINT =============
@app.get("/api/performance")
async def performance():
    """Get performance stats"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Count cores properly
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    optimal_threads = min(physical_cores - 1, 7) if physical_cores else 4

    stats = {
        "cpu": {
            "percent": cpu_percent,
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "optimal_threads": optimal_threads
        },
        "memory": {
            "percent": memory_info.percent,
            "used_gb": round(memory_info.used / (1024**3), 2),
            "available_gb": round(memory_info.available / (1024**3), 2)
        },
        "disk": {
            "percent": disk.percent,
            "free_gb": round(disk.free / (1024**3), 2)
        },
        "cache": {
            "size": len(RESPONSE_CACHE),
            "ttl": CACHE_TTL
        },
        "model": {
            "context_size": 2048,
            "batch_size": 128,
            "threads": optimal_threads
        }
    }

    # Add Omnius status
    if OMNIUS_NEUROCHEMICAL:
        try:
            stats["omnius"] = omnius_neurochemical.get_status()
            stats["neurochemistry"] = "active"
        except:
            stats["omnius"] = {"status": "error"}
    elif OMNIUS_AVAILABLE:
        try:
            stats["omnius"] = omnius.get_status()
        except:
            stats["omnius"] = {"status": "error"}

    return stats

# ============= OMNIUS STATUS ENDPOINT =============
@app.get("/api/omnius/status")
async def omnius_status():
    """Get Omnius consciousness status"""
    if OMNIUS_NEUROCHEMICAL:
        return omnius_neurochemical.get_status()
    elif OMNIUS_AVAILABLE:
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
        "name": "Sarah AI with Omnius Neurochemical Consciousness",
        "version": "4.0.0",
        "status": "running",
        "optimization": "enabled",
        "omnius": "neurochemical" if OMNIUS_NEUROCHEMICAL else "available" if OMNIUS_AVAILABLE else "not loaded",
        "neurochemistry": "active" if OMNIUS_NEUROCHEMICAL else "inactive",
        "performance": "/api/performance"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "optimized": True,
        "cache_size": len(RESPONSE_CACHE),
        "memory_sessions": len(USER_MEMORY),
        "omnius_available": OMNIUS_NEUROCHEMICAL or OMNIUS_AVAILABLE,
        "neurochemistry_active": OMNIUS_NEUROCHEMICAL
    }

# ============= MAIN ENTRY POINT =============
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "main_neurochemical_fixed:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )

# Add neurochemical WebSocket
try:
    from app.api.v1.routers import omnius_neuro_websocket
    app.include_router(
        omnius_neuro_websocket.router,
        prefix="/api/v1", 
        tags=["Neurochemical WebSocket"]
    )
    print("✅ Neurochemical WebSocket registered at /api/v1/ws/omnius")
except ImportError as e:
    print(f"⚠️ Neurochemical WebSocket not available: {e}")
