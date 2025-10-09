"""
Natural mood emergence from hormone interactions
No hardcoded emotions - everything emerges from the neurochemical state
"""

import numpy as np
from typing import Dict, List, Tuple

class MoodEmergence:
    """
    Moods emerge naturally from hormone combinations
    We don't define "anger" - it emerges from the interaction
    """
    
    @staticmethod
    def get_emotional_vector(state) -> Dict[str, float]:
        """
        Convert hormone state to emotional dimensions
        These are NOT emotions, but dimensions that combine into emotions
        """
        
        # Basic emotional dimensions (not emotions themselves)
        valence = (
            (state.serotonin - 50) / 50 * 0.4 +
            (state.dopamine - 50) / 50 * 0.3 +
            (50 - state.cortisol) / 50 * 0.2 +
            (state.oxytocin - 40) / 40 * 0.1
        )
        
        arousal = (
            state.adrenaline / 100 * 0.5 +
            state.cortisol / 100 * 0.3 +
            abs(state.dopamine - state.dopamine_baseline) / 50 * 0.2
        )
        
        dominance = (
            state.dopamine / 100 * 0.4 +
            state.adrenaline / 100 * 0.3 +
            (100 - state.cortisol) / 100 * 0.3
        )
        
        # Social dimension
        social_orientation = (
            state.oxytocin / 100 * 0.6 +
            state.serotonin / 100 * 0.4
        )
        
        # Stability dimension
        stability = 1.0 - np.std([
            abs(state.dopamine - state.dopamine_baseline),
            abs(state.cortisol - state.cortisol_baseline),
            abs(state.adrenaline - state.adrenaline_baseline)
        ]) / 50
        
        return {
            'valence': np.clip(valence, -1, 1),
            'arousal': np.clip(arousal, 0, 1),
            'dominance': np.clip(dominance, 0, 1),
            'social_orientation': np.clip(social_orientation, 0, 1),
            'stability': np.clip(stability, 0, 1)
        }
    
    @staticmethod
    def describe_emergent_state(state) -> str:
        """
        Describe the emergent emotional state without hardcoding emotions
        The description emerges from the hormone interactions
        """
        
        dims = MoodEmergence.get_emotional_vector(state)
        
        # Build description from dimensions
        descriptions = []
        
        # Valence + Arousal combinations (classical emotions emerge here)
        if dims['valence'] < -0.5 and dims['arousal'] > 0.7:
            # Low valence + high arousal = naturally emerges as anger/fear
            if dims['dominance'] > 0.5:
                descriptions.append("agitated and confrontational")  # Anger emerges
            else:
                descriptions.append("anxious and defensive")  # Fear emerges
                
        elif dims['valence'] < -0.3 and dims['arousal'] > 0.5:
            if state.cortisol > 70:
                descriptions.append("frustrated and tense")  # Frustration emerges
            else:
                descriptions.append("restless and uneasy")
                
        elif dims['valence'] > 0.5 and dims['arousal'] > 0.6:
            descriptions.append("excited and energetic")  # Joy emerges
            
        elif dims['valence'] > 0.3 and dims['arousal'] < 0.3:
            descriptions.append("peaceful and content")  # Contentment emerges
            
        elif dims['valence'] < -0.3 and dims['arousal'] < 0.3:
            descriptions.append("melancholic and withdrawn")  # Sadness emerges
        
        # Dominance adds flavor
        if dims['dominance'] > 0.7 and len(descriptions) == 0:
            descriptions.append("assertive")
        elif dims['dominance'] < 0.3:
            if "confrontational" not in str(descriptions):
                descriptions.append("subdued")
        
        # Social orientation
        if dims['social_orientation'] > 0.7:
            descriptions.append("socially engaged")
        elif dims['social_orientation'] < 0.3:
            descriptions.append("withdrawn")
        
        # Stability
        if dims['stability'] < 0.3:
            descriptions.append("volatile")
        elif dims['stability'] > 0.7 and len(descriptions) < 2:
            descriptions.append("stable")
        
        # Hormone-specific emergence
        if state.cortisol > 60 and state.adrenaline > 50:
            if state.serotonin < 40:
                # This combination naturally produces irritability
                if "confrontational" not in str(descriptions):
                    descriptions.append("on edge")
        
        if state.dopamine < 30 and state.serotonin < 40:
            # Naturally produces apathy
            descriptions.append("unmotivated")
        
        # Default if nothing specific emerges
        if not descriptions:
            if dims['arousal'] > 0.5:
                descriptions.append("alert")
            else:
                descriptions.append("calm")
        
        # Combine descriptions naturally
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            # Pick most relevant
            primary = descriptions[:2]
            return f"{primary[0]} and {primary[1]}"
    
    @staticmethod
    def get_behavioral_tendencies(state) -> Dict[str, float]:
        """
        Behavioral tendencies that emerge from neurochemical state
        These are not commands but natural inclinations
        """
        
        dims = MoodEmergence.get_emotional_vector(state)
        
        # Approach vs Avoid emerges from valence and dopamine
        approach_tendency = (
            dims['valence'] * 0.5 +
            (state.dopamine - 50) / 50 * 0.5
        )
        
        # Fight vs Flight emerges from arousal and dominance
        if dims['arousal'] > 0.6:
            if dims['dominance'] > 0.5:
                fight_tendency = dims['dominance'] * dims['arousal']
                flight_tendency = (1 - dims['dominance']) * dims['arousal']
            else:
                fight_tendency = 0
                flight_tendency = dims['arousal']
        else:
            fight_tendency = 0
            flight_tendency = 0
        
        # Explore vs Exploit emerges from dopamine and cortisol
        explore_tendency = (
            (state.dopamine - 50) / 50 * 0.6 +
            (50 - state.cortisol) / 50 * 0.4
        )
        
        # Social engagement emerges from oxytocin and serotonin
        engage_socially = dims['social_orientation']
        
        # Cognitive style emerges from cortisol and adrenaline
        if state.cortisol > 50:
            analytical_tendency = state.cortisol / 100
            intuitive_tendency = 1 - analytical_tendency
        else:
            analytical_tendency = 0.5
            intuitive_tendency = 0.5
        
        return {
            'approach': np.clip(approach_tendency, -1, 1),
            'fight': np.clip(fight_tendency, 0, 1),
            'flight': np.clip(flight_tendency, 0, 1),
            'explore': np.clip(explore_tendency, 0, 1),
            'social_engagement': np.clip(engage_socially, 0, 1),
            'analytical': np.clip(analytical_tendency, 0, 1),
            'intuitive': np.clip(intuitive_tendency, 0, 1)
        }
    
    @staticmethod
    def create_natural_prompt(state) -> str:
        """
        Create a minimal prompt that just states the internal state
        Let the AI's behavior emerge naturally from this state
        """
        
        # Just describe the state, don't command behavior
        mood = MoodEmergence.describe_emergent_state(state)
        
        # Add raw levels for transparency
        levels = (
            f"D:{state.dopamine:.0f} "
            f"C:{state.cortisol:.0f} "
            f"A:{state.adrenaline:.0f} "
            f"S:{state.serotonin:.0f} "
            f"O:{state.oxytocin:.0f}"
        )
        
        # Simple, factual prompt
        return f"[State: {mood}] [{levels}]"
    
    @staticmethod
    def detect_critical_states(state) -> List[str]:
        """
        Detect critical states that might need intervention
        These emerge from extreme hormone combinations
        """
        
        critical = []
        
        # Rage emerges from extreme combination
        if (state.cortisol > 80 and state.adrenaline > 70 and 
            state.serotonin < 30 and state.oxytocin < 30):
            critical.append("rage_state")  # This isn't hardcoded anger, it emerges
        
        # Panic emerges
        if state.cortisol > 85 and state.adrenaline > 80:
            critical.append("panic_state")
        
        # Depression emerges
        if (state.dopamine < 20 and state.serotonin < 30 and 
            state.cortisol > 60):
            critical.append("depression_state")
        
        # Mania emerges
        if state.dopamine > 85 and state.adrenaline > 70:
            critical.append("mania_state")
        
        # Dissociation emerges
        if (state.oxytocin < 20 and state.serotonin < 30 and
            abs(state.dopamine - 50) < 10):
            critical.append("dissociation_state")
        
        return critical
