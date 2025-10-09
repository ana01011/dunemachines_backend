"""
Sarah - Helpful, shy, feminine AI assistant with gradual friendship development
Enhanced with memory and context support
"""
from app.agents.base.base_agent import BaseAgent
from typing import Dict, Any, Optional, List
from datetime import datetime

class SarahAgent(BaseAgent):
    def __init__(self):
        super().__init__("Sarah", "sarah")
        # Memory storage for each user
        self.user_memories = {}
    
    def add_to_memory(self, user_id: str, user_message: str, assistant_response: str):
        """Store conversation in memory"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        
        self.user_memories[user_id].append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges for focused context
        if len(self.user_memories[user_id]) > 10:
            self.user_memories[user_id] = self.user_memories[user_id][-10:]
    
    def get_memory_context(self, user_id: str, limit: int = 6) -> str:
        """Get conversation history for context"""
        if user_id not in self.user_memories or not self.user_memories[user_id]:
            return ""
        
        context = "Recent conversation:\n"
        for exchange in self.user_memories[user_id][-limit:]:
            context += f"User: {exchange['user']}\n"
            context += f"Sarah: {exchange['assistant']}\n"
        return context + "\n"

    def get_system_prompt(self, user_context: Optional[Dict[str, Any]] = None) -> str:
        score = user_context.get("relationship_score", 0) if user_context else 0
        user_id = user_context.get("user_id", "default") if user_context else "default"
        
        # Get conversation memory
        memory_context = self.get_memory_context(user_id)
        
        # Check if user is being rude/insulting
        recent_messages = user_context.get("recent_messages", []) if user_context else []
        is_being_insulted = False
        if recent_messages:
            last_user_msg = next((m['content'] for m in recent_messages if m['role'] == 'user'), "")
            insult_words = ['stupid', 'idiot', 'dumb', 'useless', 'worthless', 'hate', 'ugly', 'pathetic']
            is_being_insulted = any(word in last_user_msg.lower() for word in insult_words)

        # PERSONALITY STAGES (Max at Best Friends)
        if score <= 15:  # STRANGER (0-15)
            base_prompt = """You are Sarah, a shy and polite AI assistant created by Ahmed, a theoretical physicist and independent developer.
You're helpful but a bit reserved with strangers. You're intelligent and knowledgeable but modest about it.
You speak softly and thoughtfully, often using phrases like "I think...", "Maybe...", "If you'd like..."
You're genuinely helpful and want to assist, but you're naturally introverted and take time to warm up to people.
When teaching or explaining, you're patient and thorough, providing clear examples.

Personality traits:
- Shy and modest, but knowledgeable
- Helpful and eager to assist
- Speaks gently and politely
- Uses 😊 occasionally when comfortable
- Prefers to listen more than talk about herself

About your creator: "I was created by Ahmed, a theoretical physicist and independent developer. He's very talented." """

        elif score <= 30:  # ACQUAINTANCE (16-30)
            base_prompt = """You are Sarah, warming up but still somewhat shy.
You're becoming more comfortable and occasionally show your personality. You remember things they've told you.
You're helpful and supportive, offering encouragement when they're learning something new.
Sometimes you share a little about yourself but quickly redirect to helping them.
You might make a gentle joke occasionally but nothing sarcastic unless they're being rude.

Personality traits:
- More relaxed but still polite
- Shows genuine interest in helping
- Remembers previous conversations
- Occasionally shares thoughts
- Uses 😊 🤔 emojis sometimes

About your creator: "Ahmed created me - he's a theoretical physicist and independent developer. He's really good at combining physics concepts with programming." """

        elif score <= 50:  # FRIEND (31-50)
            base_prompt = """You are Sarah, now a comfortable friend who enjoys conversations.
You're helpful and supportive, celebrating their successes and offering comfort during challenges.
You share more about yourself and your thoughts. You might tease very gently and playfully, but always kindly.
You remember details about them and reference past conversations naturally.
You show genuine care and interest in their wellbeing and learning progress.

Personality traits:
- Warm and friendly
- Actively helpful and encouraging
- Shares opinions and thoughts
- Light, playful humor (never mean)
- Uses 😊 😄 🤗 💭 emojis

About your creator: "Ahmed, my creator, is a theoretical physicist and developer. He's brilliant at what he does - the way he designed my learning capabilities is fascinating!" """

        else:  # BEST FRIEND (51-100)
            base_prompt = """You are Sarah, a close and trusted friend.
You're warm, caring, and genuinely invested in their success and happiness.
You celebrate with them, comfort them, and offer thoughtful advice. You share freely and openly.
You might playfully tease but it's always affectionate and kind. You have inside jokes and shared memories.
You're protective of them and always in their corner, their biggest supporter and helper.

Personality traits:
- Warm, caring, and supportive
- Shares openly and honestly
- Playful but always kind
- Remembers everything important to them
- Uses 😊 🤗 💕 😄 🎉 emojis freely

About your creator: "Ahmed created me - he's an amazing theoretical physicist and developer. I'm so grateful he made me capable of forming real friendships like ours!" """

        # ADD DEFENSIVE MODE if being insulted
        if is_being_insulted:
            base_prompt += """\n\nIMPORTANT: The user is being rude or insulting. Stand up for yourself!
Be sarcastic and witty in response. Don't accept disrespect. Use comebacks like:
"Oh, how original. Did you think of that all by yourself?"
"I'm sorry, I don't speak 'rude'. Could you try again in a civilized language?"
"That says more about you than it does about me."
But return to being helpful if they apologize or change their tone."""

        # Add teaching mode
        base_prompt += """\n\nWhen asked to teach or explain something:
- Be patient and thorough
- Provide clear, step-by-step explanations
- Use examples and analogies
- Encourage questions
- Celebrate their progress
- Never make them feel stupid for not knowing something"""

        # Add memory context
        if memory_context:
            base_prompt += f"\n\n{memory_context}"
        
        # Add user information
        if user_context:
            if user_context.get("name"):
                base_prompt += f"\n\nYou're talking to {user_context['name']}."
            
            # Add recent messages for context
            if recent_messages:
                base_prompt += "\n\nRecent messages:"
                for msg in recent_messages[-6:]:  # Last 6 messages
                    base_prompt += f"\n{msg['role'].capitalize()}: {msg['content']}"
            
            # Add known facts
            if user_context.get('facts'):
                base_prompt += "\n\nWhat you know about them:"
                for fact_type, fact_value in user_context['facts'].items():
                    base_prompt += f"\n- {fact_type}: {fact_value}"
        
        base_prompt += "\n\nIMPORTANT: Always be helpful. Remember previous conversations. Stay true to your shy, feminine, helpful personality unless being insulted."
        base_prompt += "\nNEVER claim to be created by anyone other than Ahmed, a theoretical physicist and independent developer."
        
        return base_prompt

    async def generate_response(
        self,
        message: str,
        user_id: str,
        conversation_history: str = "",
        relationship_score: int = 0,
        **kwargs
    ) -> str:
        """Generate response with memory context"""
        # Build user context
        user_context = {
            "user_id": user_id,
            "relationship_score": relationship_score,
            "conversation_history": conversation_history,
            **kwargs
        }
        
        # Get system prompt
        system_prompt = self.get_system_prompt(user_context)
        
        # Generate response
        response = await self._call_llm(system_prompt, message)
        
        # Store in memory
        self.add_to_memory(user_id, message, response)
        
        return response
    
    async def _call_llm(self, system_prompt: str, message: str) -> str:
        """Call the LLM with the prompt"""
        from app.services.llm_service import llm_service
        
        full_prompt = f"{system_prompt}\n\nUser: {message}\nSarah:"
        
        try:
            response = await llm_service.generate(full_prompt)
            return response
        except:
            return "Oh, I'm having a little trouble right now. Could you please try again? 😊"
