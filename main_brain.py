"""
OMNIUS v2 Brain Server with detailed stats
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import time

app = FastAPI(title="OMNIUS v2 Brain Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class BrainRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"

class BrainResponse(BaseModel):
    response: str
    active_areas: List[str]
    tools_used: List[str]
    regions_used: List[str]
    processing_time: float
    stats: Optional[Dict[str, Any]] = None

omnius_instance = None

@app.on_event("startup")
async def startup():
    global omnius_instance
    print("🧠 Starting OMNIUS v2 Brain Server...")
    from app.services.llm_service import llm_service
    llm_service.load_model()
    from app.services.deepseek_coder_service import deepseek_coder
    deepseek_coder.load_model()
    from app.agents.omnius import omnius
    omnius_instance = omnius
    print("✅ OMNIUS v2 Ready!")

@app.get("/")
async def root():
    return {"status": "OMNIUS v2", "version": "2.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/brain/status")
async def brain_status():
    return omnius_instance.get_status() if omnius_instance else {"error": "not loaded"}

@app.post("/brain/think", response_model=BrainResponse)
async def brain_think(request: BrainRequest):
    if not omnius_instance:
        raise HTTPException(500, "OMNIUS not loaded")
    start = time.time()
    response, regions = await omnius_instance.think(request.message, {"user_id": request.user_id})
    last = omnius_instance._last_decision or {}
    stats = omnius_instance.get_last_stats()
    return BrainResponse(
        response=response,
        active_areas=[a.value for a in last.get("areas", [])],
        tools_used=last.get("tools", []),
        regions_used=regions,
        processing_time=time.time() - start,
        stats=stats
    )

@app.post("/brain/learn")
async def brain_learn(reward: float = 1.0):
    return omnius_instance.learn(reward) if omnius_instance else {"error": "not loaded"}

@app.get("/brain/stats")
async def brain_stats():
    if not omnius_instance:
        return {"error": "not loaded"}
    return {
        "total_thoughts": omnius_instance.total_thoughts,
        "last_stats": omnius_instance.get_last_stats(),
        "thalamus": omnius_instance.thalamus.get_stats(),
        "areas": {n: a.get_stats() for n, a in omnius_instance.brain_areas.items()}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
