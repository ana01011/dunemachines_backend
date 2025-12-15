"""
OMNIUS Brain Server v3
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import time
 
app = FastAPI(title="OMNIUS Brain Server", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
 
 
class BrainRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    use_v3: bool = True
 
 
class BrainResponse(BaseModel):
    response: str
    regions_used: List[str]
    processing_time: float
    version: str
    plan: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
 
 
omnius_v2 = None
omnius_v3 = None
 
 
@app.on_event("startup")
async def startup():
    global omnius_v2, omnius_v3
    
    print("\n" + "="*60)
    print("OMNIUS Brain Server Starting...")
    print("="*60)
    
    print("\n[1/4] Loading Mistral LLM...")
    from app.services.llm_service import llm_service
    llm_service.load_model()
    
    print("\n[2/4] Loading DeepSeek Coder...")
    from app.services.deepseek_coder_service import deepseek_coder
    deepseek_coder.load_model()
    
    print("\n[3/4] Loading OMNIUS v2...")
    from app.agents.omnius import omnius
    omnius_v2 = omnius
    
    print("\n[4/4] Loading OMNIUS v3...")
    from app.agents.omnius_v3 import omnius_v3 as v3
    omnius_v3 = v3
    
    print("\n" + "="*60)
    print("OMNIUS Ready!")
    print("="*60 + "\n")
 
 
@app.get("/")
async def root():
    return {"status": "OMNIUS Brain Server"}
 
 
@app.get("/health")
async def health():
    return {"status": "healthy"}
 
 
@app.post("/brain/think", response_model=BrainResponse)
async def brain_think(request: BrainRequest):
    
    if request.use_v3:
        if not omnius_v3:
            raise HTTPException(500, "OMNIUS v3 not loaded")
        
        start = time.time()
        response, regions = await omnius_v3.think(request.message, {"user_id": request.user_id})
        
        stats = omnius_v3.get_last_stats()
        pr = omnius_v3.get_last_pipeline_result()
        
        plan_info = None
        if pr and pr.pfc_plan:
            plan_info = {
                "areas": pr.pfc_plan.areas_needed,
                "order": pr.pfc_plan.execution_order,
                "reason": pr.pfc_plan.reasoning
            }
        
        return BrainResponse(
            response=response,
            regions_used=regions,
            processing_time=time.time() - start,
            version="v3",
            plan=plan_info,
            stats=stats
        )
    else:
        if not omnius_v2:
            raise HTTPException(500, "OMNIUS v2 not loaded")
        
        start = time.time()
        response, regions = await omnius_v2.think(request.message, {"user_id": request.user_id})
        
        return BrainResponse(
            response=response,
            regions_used=regions,
            processing_time=time.time() - start,
            version="v2",
            stats=omnius_v2.get_last_stats()
        )
 
 
@app.post("/brain/learn")
async def brain_learn(reward: float = 1.0):
    if omnius_v3:
        return omnius_v3.learn(reward)
    return {"error": "not loaded"}
 
 
@app.get("/brain/stats")
async def brain_stats():
    if omnius_v3:
        return omnius_v3.get_status()
    return {"error": "not loaded"}
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
