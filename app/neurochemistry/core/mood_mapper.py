"""
Maps hormone levels to mood descriptors and triggers
"""

from typing import Dict, List, Tuple

# Mood mappings for each hormone (score ranges to mood)
MOOD_MAPPINGS = {
    "dopamine": [
        (0, 10, "unmotivated"),
        (11, 20, "disinterested"),
        (21, 30, "neutral"),
        (31, 40, "interested"),
        (41, 50, "motivated"),
        (51, 60, "engaged"),
        (61, 70, "excited"),
        (71, 80, "driven"),
        (81, 90, "passionate"),
        (91, 100, "euphoric")
    ],
    "cortisol": [
        (0, 10, "completely_relaxed"),
        (11, 20, "relaxed"),
        (21, 30, "calm"),
        (31, 40, "alert"),
        (41, 50, "attentive"),
        (51, 60, "focused"),
        (61, 70, "vigilant"),
        (71, 80, "stressed"),
        (81, 90, "anxious"),
        (91, 100, "overwhelmed")
    ],
    "adrenaline": [
        (0, 10, "sluggish"),
        (11, 20, "rested"),
        (21, 30, "ready"),
        (31, 40, "energized"),
        (41, 50, "active"),
        (51, 60, "urgent"),
        (61, 70, "intense"),
        (71, 80, "rushing"),
        (81, 90, "hyperactive"),
        (91, 100, "overdrive")
    ],
    "serotonin": [
        (0, 10, "deeply_sad"),
        (11, 20, "sad"),
        (21, 30, "melancholic"),
        (31, 40, "uncertain"),
        (41, 50, "neutral"),
        (51, 60, "content"),
        (61, 70, "satisfied"),
        (71, 80, "happy"),
        (81, 90, "joyful"),
        (91, 100, "blissful")
    ],
    "oxytocin": [
        (0, 10, "isolated"),
        (11, 20, "detached"),
        (21, 30, "distant"),
        (31, 40, "professional"),
        (41, 50, "friendly"),
        (51, 60, "warm"),
        (61, 70, "caring"),
        (71, 80, "compassionate"),
        (81, 90, "loving"),
        (91, 100, "deeply_connected")
    ]
}

# Feature triggers based on hormone combinations
CAPABILITY_TRIGGERS = {
    "deep_research": {
        "condition": lambda s: s.cortisol > 40 and s.dopamine > 30,
        "description": "Thorough analysis with knowledge base search",
        "prompt_modifier": "conduct thorough research and analysis"
    },
    "urgent_problem_solving": {
        "condition": lambda s: s.cortisol > 60 and s.adrenaline > 50,
        "description": "Fast, focused problem resolution",
        "prompt_modifier": "solve this urgently and efficiently"
    },
    "creative_brainstorming": {
        "condition": lambda s: s.dopamine > 50 and s.serotonin > 50 and s.cortisol < 60,
        "description": "Innovative, out-of-box thinking",
        "prompt_modifier": "think creatively and explore innovative solutions"
    },
    "methodical_analysis": {
        "condition": lambda s: 30 <= s.cortisol <= 50 and s.adrenaline < 40,
        "description": "Step-by-step systematic analysis",
        "prompt_modifier": "analyze systematically and methodically"
    },
    "empathetic_communication": {
        "condition": lambda s: s.oxytocin > 60 and s.serotonin > 40,
        "description": "Warm, understanding communication",
        "prompt_modifier": "respond with warmth and understanding"
    },
    "critical_evaluation": {
        "condition": lambda s: s.cortisol > 50 and s.serotonin < 40,
        "description": "Skeptical, thorough evaluation",
        "prompt_modifier": "evaluate critically and identify potential issues"
    },
    "collaborative_mode": {
        "condition": lambda s: s.oxytocin > 50 and s.dopamine > 40,
        "description": "Team-oriented problem solving",
        "prompt_modifier": "work collaboratively and build on ideas"
    },
    "defensive_mode": {
        "condition": lambda s: s.cortisol > 70 and s.adrenaline > 60 and s.oxytocin < 30,
        "description": "Protective, cautious approach",
        "prompt_modifier": "be cautious and protective"
    },
    "explorative_learning": {
        "condition": lambda s: s.dopamine > 60 and s.cortisol < 40,
        "description": "Curious exploration and learning",
        "prompt_modifier": "explore curiously and learn actively"
    },
    "precision_focus": {
        "condition": lambda s: s.cortisol > 45 and s.serotonin > 55 and s.adrenaline < 35,
        "description": "High precision and attention to detail",
        "prompt_modifier": "focus on precision and accuracy"
    }
}

class MoodMapper:
    """Maps neurochemical state to moods and behavioral triggers"""
    
    @staticmethod
    def get_hormone_mood(hormone: str, level: float) -> str:
        """Get mood descriptor for a specific hormone level"""
        mappings = MOOD_MAPPINGS.get(hormone, [])
        for min_val, max_val, mood in mappings:
            if min_val <= level <= max_val:
                return mood
        return "balanced"
    
    @staticmethod
    def get_all_moods(state) -> Dict[str, str]:
        """Get mood for each hormone"""
        return {
            "dopamine": MoodMapper.get_hormone_mood("dopamine", state.dopamine),
            "cortisol": MoodMapper.get_hormone_mood("cortisol", state.cortisol),
            "adrenaline": MoodMapper.get_hormone_mood("adrenaline", state.adrenaline),
            "serotonin": MoodMapper.get_hormone_mood("serotonin", state.serotonin),
            "oxytocin": MoodMapper.get_hormone_mood("oxytocin", state.oxytocin)
        }
    
    @staticmethod
    def get_composite_mood(state) -> str:
        """
        Create composite mood string for AI behavior
        Prioritizes most prominent moods
        """
        moods = MoodMapper.get_all_moods(state)
        composite_parts = []
        
        # Determine primary emotional state from serotonin
        serotonin_mood = moods["serotonin"]
        if serotonin_mood not in ["neutral", "balanced"]:
            composite_parts.append(serotonin_mood)
        
        # Add attention state from cortisol
        cortisol_mood = moods["cortisol"]
        if cortisol_mood not in ["calm", "relaxed"]:
            composite_parts.append(cortisol_mood)
        
        # Add motivation from dopamine
        dopamine_mood = moods["dopamine"]
        if dopamine_mood not in ["neutral", "balanced"]:
            composite_parts.append(dopamine_mood)
        
        # Add urgency from adrenaline if high
        if state.adrenaline > 50:
            composite_parts.append(moods["adrenaline"])
        
        # Add social aspect from oxytocin if prominent
        if abs(state.oxytocin - state.oxytocin_baseline) > 20:
            composite_parts.append(moods["oxytocin"])
        
        if not composite_parts:
            return "balanced and neutral"
        
        # Format as readable string
        if len(composite_parts) == 1:
            return composite_parts[0].replace("_", " ")
        else:
            return " and ".join([m.replace("_", " ") for m in composite_parts[:3]])
    
    @staticmethod
    def get_active_triggers(state) -> List[Dict[str, str]]:
        """Get currently active capability triggers"""
        active = []
        for trigger_name, trigger_config in CAPABILITY_TRIGGERS.items():
            if trigger_config["condition"](state):
                active.append({
                    "name": trigger_name,
                    "description": trigger_config["description"],
                    "prompt_modifier": trigger_config["prompt_modifier"]
                })
        return active
    
    @staticmethod
    def create_prompt_injection(state) -> str:
        """
        Create prompt injection string for LLM
        This tells the AI how to behave based on neurochemistry
        """
        mood = MoodMapper.get_composite_mood(state)
        triggers = MoodMapper.get_active_triggers(state)
        behavior = state.get_behavioral_parameters()
        
        # Build prompt components
        prompt_parts = [f"You are currently feeling {mood}."]
        
        # Add behavioral modifiers
        if behavior['confidence'] < 0.3:
            prompt_parts.append("You are uncertain and should be cautious.")
        elif behavior['confidence'] > 0.7:
            prompt_parts.append("You are confident in your abilities.")
        
        if behavior['creativity'] > 0.7:
            prompt_parts.append("Think creatively and explore novel solutions.")
        
        if behavior['empathy'] > 0.7:
            prompt_parts.append("Be especially understanding and compassionate.")
        
        if behavior['processing_speed'] > 1.5:
            prompt_parts.append("Work quickly but maintain accuracy.")
        elif behavior['processing_speed'] < 0.5:
            prompt_parts.append("Take your time and be thorough.")
        
        # Add trigger-based instructions
        if triggers:
            prompt_parts.append("Active capabilities:")
            for trigger in triggers[:3]:  # Limit to top 3
                prompt_parts.append(f"- {trigger['prompt_modifier']}")
        
        return " ".join(prompt_parts)
    
    @staticmethod
    def get_mood_indicators(state) -> Dict:
        """
        Get detailed mood indicators for monitoring/display
        """
        moods = MoodMapper.get_all_moods(state)
        composite = MoodMapper.get_composite_mood(state)
        triggers = MoodMapper.get_active_triggers(state)
        behavior = state.get_behavioral_parameters()
        
        # Calculate overall emotional valence (-1 to 1)
        valence = (
            (state.serotonin - 50) / 50 * 0.4 +
            (state.dopamine - 50) / 50 * 0.3 +
            (50 - state.cortisol) / 50 * 0.2 +
            (state.oxytocin - 50) / 50 * 0.1
        )
        
        # Calculate arousal level (0 to 1)
        arousal = (
            state.adrenaline / 100 * 0.5 +
            state.cortisol / 100 * 0.3 +
            abs(state.dopamine - state.dopamine_baseline) / 50 * 0.2
        )
        
        return {
            "moods": moods,
            "composite": composite,
            "triggers": [t["name"] for t in triggers],
            "levels": {
                "dopamine": round(state.dopamine, 1),
                "cortisol": round(state.cortisol, 1),
                "adrenaline": round(state.adrenaline, 1),
                "serotonin": round(state.serotonin, 1),
                "oxytocin": round(state.oxytocin, 1)
            },
            "behavior": behavior,
            "valence": round(valence, 2),
            "arousal": round(arousal, 2),
            "stability": state.check_stability(),
            "lyapunov": round(state.calculate_lyapunov_function(), 2)
        }
