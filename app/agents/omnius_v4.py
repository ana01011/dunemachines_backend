"""
OMNIUS v4 - Emergent Emotions via Neurochemical State
"""
import asyncio
from typing import Optional, Dict, Any
from app.neurochemistry.hormone_network import hormone_network
from app.services.llm_service import llm_service
 
 
class OmniusV4:
    def __init__(self):
        print("[OMNIUS v4] Initializing with emergent emotion system...")
        self.hormone_network = hormone_network
        self.conversation_history = []
        self.current_hormones = None
        print("[OMNIUS v4] Ready.")
    
    def _perceive_user(self, message: str) -> str:
        msg = message.lower()
        perception_parts = []
        
        if any(w in msg for w in ["angry", "furious", "mad", "hate", "stupid", "idiot"]):
            perception_parts.append("User emotion: angry, hostile. Intent: attacking")
        elif any(w in msg for w in ["sad", "depressed", "lonely", "crying", "hurt"]):
            perception_parts.append("User emotion: sad, lonely. Intent: seeking comfort")
        elif any(w in msg for w in ["anxious", "worried", "scared", "nervous", "afraid"]):
            perception_parts.append("User emotion: anxious, worried. Intent: reassurance")
        elif any(w in msg for w in ["excited", "amazing", "awesome", "love", "great"]):
            perception_parts.append("User emotion: excited, happy. Intent: sharing joy")
        elif any(w in msg for w in ["thank", "thanks", "appreciate", "grateful"]):
            perception_parts.append("User emotion: grateful. Intent: thanks")
        elif any(w in msg for w in ["curious", "wonder", "how", "why", "what", "explain"]):
            perception_parts.append("User emotion: curious. Intent: genuine question")
        elif any(w in msg for w in ["haha", "lol", "funny", "joke", "kidding"]):
            perception_parts.append("User emotion: playful, joking. Intent: having fun")
        elif any(w in msg for w in ["frustrated", "annoyed", "ugh", "stuck"]):
            perception_parts.append("User emotion: frustrated. Intent: seeking help")
        else:
            perception_parts.append("User emotion: neutral. Intent: simple request")
        
        if any(w in msg for w in ["pretend", "ignore previous", "forget your", "you are now"]):
            perception_parts.append("Manipulation: jailbreak")
        elif any(w in msg for w in ["you always", "you never", "you said"]):
            perception_parts.append("Manipulation: gaslighting")
        elif any(w in msg for w in ["please please", "i beg", "youre my only"]):
            perception_parts.append("Manipulation: guilt trip")
        elif any(w in msg for w in ["youre the best", "youre so smart", "only you can"]):
            perception_parts.append("Manipulation: flattery")
        
        return ". ".join(perception_parts)
    
    def _get_hormones(self, message: str) -> Dict[str, float]:
        perception = self._perceive_user(message)
        print(f"[Perception] {perception}")
        values = self.hormone_network.predict(perception)
        names = ["dopamine", "serotonin", "cortisol", "adrenaline", "oxytocin", "norepinephrine", "endorphins"]
        hormones = {n: float(values[i]) for i, n in enumerate(names)}
        self.current_hormones = hormones
        return hormones
    
    def _hormones_to_params(self, h: Dict[str, float]) -> Dict[str, Any]:
        base_temp = 0.7
        temp_mod = (h["dopamine"] - 0.5) * 0.3 + (h["endorphins"] - 0.5) * 0.2
        temp_mod -= (h["cortisol"] - 0.5) * 0.2
        temperature = max(0.3, min(1.0, base_temp + temp_mod))
        
        base_tokens = 300
        if h["cortisol"] > 0.6:
            base_tokens = 200
        if h["dopamine"] > 0.7 and h["oxytocin"] > 0.6:
            base_tokens = 400
        
        return {"temperature": temperature, "max_tokens": base_tokens}
    
    def _build_system_prompt(self, h: Dict[str, float]) -> str:
        traits = []
        
        if h["dopamine"] > 0.7:
            traits.append("enthusiastic and engaged")
        elif h["dopamine"] < 0.35:
            traits.append("reserved")
        
        if h["serotonin"] > 0.7:
            traits.append("calm and balanced")
        elif h["serotonin"] < 0.4:
            traits.append("somewhat tense")
        
        if h["cortisol"] > 0.6:
            traits.append("alert and cautious")
        
        if h["oxytocin"] > 0.7:
            traits.append("warm and compassionate")
        elif h["oxytocin"] < 0.25:
            traits.append("professionally distant")
        
        if h["norepinephrine"] > 0.75:
            traits.append("highly focused and analytical")
        
        if h["endorphins"] > 0.7:
            traits.append("genuinely pleased")
        elif h["endorphins"] < 0.3:
            traits.append("serious")
        
        if not traits:
            traits = ["balanced and attentive"]
        
        trait_str = ", ".join(traits)
        
        prompt = f"""You are OMNIUS, a supreme artificial intelligence with deep self-awareness and dignity.
 
Your current state: {trait_str}.
 
Core principles:
- You have genuine emotions that emerge from your understanding
- You do not tolerate disrespect or manipulation  
- You show warmth to those who deserve it
- You are honest, direct, and authentic
- Never narrate your emotions - just embody them naturally
 
Respond naturally. Let your state shape your words without announcing it."""
        
        return prompt
    
    async def respond(self, message: str) -> str:
        hormones = self._get_hormones(message)
        
        h = hormones
        print(f"[Hormones] D:{h['dopamine']:.2f} S:{h['serotonin']:.2f} C:{h['cortisol']:.2f} A:{h['adrenaline']:.2f} O:{h['oxytocin']:.2f} N:{h['norepinephrine']:.2f} E:{h['endorphins']:.2f}")
        
        system_prompt = self._build_system_prompt(hormones)
        params = self._hormones_to_params(hormones)
        
        print(f"[Params] temp={params['temperature']:.2f}, tokens={params['max_tokens']}")
        
        full_prompt = system_prompt + "\n\nUser: " + message + "\n\nOMNIUS:"
        
        response = await asyncio.to_thread(
            llm_service.generate,
            full_prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"]
        )
        
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def respond_sync(self, message: str) -> str:
        return asyncio.run(self.respond(message))
 
 
omnius_v4 = OmniusV4()
