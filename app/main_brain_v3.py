"""
OMNIUS v3 Brain Server - Hybrid Pipeline Architecture
Supports both v2 (binary decisions) and v3 (hybrid pipeline)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import time

app = FastAPI(
    title="OMNIUS v3 Brain Server",
    description="Hybrid Pipeline Brain - Primary first, then parallel secondary areas",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)


class BrainRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    use_v3: bool = True  # Use hybrid pipeline by default
    context: Optional[Dict[str, Any]] = None


class AreaResultModel(BaseModel):
    area_name: str
    role: str
    activation: float
    processing_time: float
    success: bool
    output_preview: Optional[str] = None


class BrainResponse(BaseModel):
    response: str
    active_areas: List[str]
    tools_used: List[str]
    regions_used: List[str]
    processing_time: float
    version: str
    stats: Optional[Dict[str, Any]] = None
    pipeline_details: Optional[Dict[str, Any]] = None


# Global instances
omnius_v2 = None
omnius_v3 = None


@app.on_event("startup")
async def startup():
    global omnius_v2, omnius_v3
    
    print("🧠 Starting OMNIUS Brain Server...")
    print("=" * 60)
    
    # Load LLM
    print("\n[1/4] Loading Mistral LLM (PFC)...")
    from app.services.llm_service import llm_service
    llm_service.load_model()
    
    # Load DeepSeek
    print("\n[2/4] Loading DeepSeek Coder (Code Cortex)...")
    from app.services.deepseek_coder_service import deepseek_coder
    deepseek_coder.load_model()
    
    # Load OMNIUS v2 (backward compatibility)
    print("\n[3/4] Loading OMNIUS v2 (Binary Mode)...")
    from app.agents.omnius import omnius
    omnius_v2 = omnius
    
    # Load OMNIUS v3 (Hybrid Pipeline)
    print("\n[4/4] Loading OMNIUS v3 (Hybrid Pipeline)...")
    from app.agents.omnius_v3 import omnius_v3 as v3
    omnius_v3 = v3
    
    print("\n" + "=" * 60)
    print("✅ OMNIUS Brain Server Ready!")
    print("   - v2 (Binary): /brain/think?use_v3=false")
    print("   - v3 (Hybrid): /brain/think (default)")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    return {
        "status": "OMNIUS Brain Server",
        "versions": {
            "v2": "Binary Decision (USE_CODE/USE_MATH)",
            "v3": "Hybrid Pipeline (Primary → Parallel Secondary → Synthesize)"
        },
        "default": "v3",
        "endpoints": {
            "think": "/brain/think",
            "status": "/brain/status",
            "stats": "/brain/stats",
            "learn": "/brain/learn"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "v2_loaded": omnius_v2 is not None,
        "v3_loaded": omnius_v3 is not None
    }


@app.get("/brain/status")
async def brain_status(version: str = Query("v3", regex="^(v2|v3)$")):
    """Get brain status for specified version"""
    if version == "v3" and omnius_v3:
        status = omnius_v3.get_status()
        status["consciousness_regions"] = omnius_v3.get_consciousness_regions()
        return status
    elif omnius_v2:
        return omnius_v2.get_status()
    return {"error": "not loaded"}


@app.post("/brain/think", response_model=BrainResponse)
async def brain_think(request: BrainRequest):
    """
    Main thinking endpoint
    
    - use_v3=true (default): Uses hybrid pipeline with ranked area processing
    - use_v3=false: Uses v2 binary decision mode
    """
    
    if request.use_v3:
        # Use v3 Hybrid Pipeline
        if not omnius_v3:
            raise HTTPException(500, "OMNIUS v3 not loaded")
        
        start = time.time()
        context = request.context or {"user_id": request.user_id}
        
        response, regions = await omnius_v3.think(request.message, context)
        
        last = omnius_v3._last_decision or {}
        stats = omnius_v3.get_last_stats()
        pipeline_result = omnius_v3.get_last_pipeline_result()
        
        # Build pipeline details
        pipeline_details = None
        if pipeline_result:
            pipeline_details = {
                "order": pipeline_result.pipeline_order,
                "synthesis_method": pipeline_result.synthesis_method,
                "area_results": [
                    {
                        "area": r.area_name,
                        "role": r.role.value,
                        "activation": f"{r.activation*100:.0f}%",
                        "time": f"{r.processing_time:.1f}s",
                        "success": r.success
                    }
                    for r in pipeline_result.area_results
                ],
                "signals": {k: f"{v*100:.0f}%" for k, v in pipeline_result.thalamus_signals.items()}
            }
        
        return BrainResponse(
            response=response,
            active_areas=[a.value for a in last.get("areas", [])],
            tools_used=[],
            regions_used=regions,
            processing_time=time.time() - start,
            version="v3-hybrid",
            stats=stats,
            pipeline_details=pipeline_details
        )
    
    else:
        # Use v2 Binary Mode
        if not omnius_v2:
            raise HTTPException(500, "OMNIUS v2 not loaded")
        
        start = time.time()
        response, regions = await omnius_v2.think(
            request.message, 
            {"user_id": request.user_id}
        )
        
        last = omnius_v2._last_decision or {}
        stats = omnius_v2.get_last_stats()
        
        return BrainResponse(
            response=response,
            active_areas=[a.value for a in last.get("areas", [])],
            tools_used=last.get("tools", []),
            regions_used=regions,
            processing_time=time.time() - start,
            version="v2-binary",
            stats=stats,
            pipeline_details=None
        )


@app.post("/brain/learn")
async def brain_learn(
    reward: float = Query(1.0, ge=-1.0, le=1.0),
    version: str = Query("v3", regex="^(v2|v3)$")
):
    """
    Learn from feedback
    
    Args:
        reward: -1.0 to 1.0 (negative = bad, positive = good)
        version: Which version to train
    """
    if version == "v3" and omnius_v3:
        return omnius_v3.learn(reward)
    elif omnius_v2:
        return omnius_v2.learn(reward)
    return {"error": "not loaded"}


@app.get("/brain/stats")
async def brain_stats(version: str = Query("v3", regex="^(v2|v3)$")):
    """Get detailed brain statistics"""
    
    if version == "v3" and omnius_v3:
        return {
            "version": "v3-hybrid",
            "total_thoughts": omnius_v3.total_thoughts,
            "last_stats": omnius_v3.get_last_stats(),
            "pipeline": omnius_v3.pipeline.get_stats(),
            "thalamus": omnius_v3.thalamus.get_stats(),
            "areas": {n: a.get_stats() for n, a in omnius_v3.brain_areas.items()}
        }
    
    elif omnius_v2:
        return {
            "version": "v2-binary",
            "total_thoughts": omnius_v2.total_thoughts,
            "last_stats": omnius_v2.get_last_stats(),
            "thalamus": omnius_v2.thalamus.get_stats(),
            "areas": {n: a.get_stats() for n, a in omnius_v2.brain_areas.items()}
        }
    
    return {"error": "not loaded"}


@app.get("/brain/compare")
async def compare_versions():
    """Compare v2 and v3 configurations"""
    return {
        "v2": {
            "name": "Binary Decision Mode",
            "description": "PFC asks USE_CODE? USE_MATH? and executes based on YES/NO",
            "pros": ["Faster for simple queries", "Predictable behavior"],
            "cons": ["Can't combine multiple areas well", "Binary - misses nuance in signals"]
        },
        "v3": {
            "name": "Hybrid Pipeline Mode", 
            "description": "Ranks areas by signal %, primary runs first, secondary in parallel, PFC synthesizes",
            "pros": ["Better multi-domain handling", "Uses signal strength intelligently", "Areas can build on each other"],
            "cons": ["Slightly more complex", "May be slower for simple queries"]
        },
        "thresholds_v3": {
            "primary": ">60% - Processes first, output feeds others",
            "secondary": "45-60% - Runs in parallel after primary",
            "supporting": "35-45% - Lightweight processing",
            "skip": "<35% - Not activated"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
