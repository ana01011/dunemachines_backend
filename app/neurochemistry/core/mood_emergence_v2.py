"""
Natural mood emergence from hormone interactions - REFINED
Emotions emerge from hormone combinations more distinctly
"""

import numpy as np
from typing import Dict, List, Tuple

class MoodEmergence:
    """
    Moods emerge naturally from hormone combinations
    More sensitive to create distinct emotional states
    """
    
    @staticmethod
    def get_emotional_vector(state) -> Dict[str, float]:
        """Convert hormone state to emotional dimensions"""
        
        # Valence: pleasure-displeasure axis
        valence = (
            (state.serotonin - 50) / 50 * 0.4 +
            (state.dopamine - 50) / 50 * 0.3 +
            (50 - state.cortisol) / 50 * 0.2 +
            (state.oxytocin - 40) / 40 * 0.1
        )
        
        # Arousal: activation level
        arousal = (
            state.adrenaline / 100 * 0.5 +
            state.cortisol / 100 * 0.3 +
            abs(state.dopamine - state.dopamine_baseline) / 50 * 0.2
        )
        
        # Dominance: control/power feeling
        dominance = (
            state.dopamine / 100 * 0.4 +
            (100 - state.cortisol) / 100 * 0.3 +
            state.adrenaline / 100 * 0.3
        )
        
        # Social orientation
        social_orientation = (
            state.oxytocin / 100 * 0.6 +
            state.serotonin / 100 * 0.4
        )
        
        # Stability 
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
        Describe emergent emotional state
        More nuanced detection of emotional patterns
        """
        
        dims = MoodEmergence.get_emotional_vector(state)
        
        # Check for specific emotional patterns first
        # These are emergent patterns, not hardcoded emotions
        
        # ANGER PATTERN: High cortisol + adrenaline, low serotonin, low oxytocin
        anger_score = (
            (state.cortisol / 100) * 0.3 +
            (state.adrenaline / 100) * 0.3 +
            (1 - state.serotonin / 100) * 0.2 +
            (1 - state.oxytocin / 100) * 0.2
        )
        
        # FEAR PATTERN: Very high cortisol + adrenaline, low dominance
        fear_score = (
            (state.cortisol / 100) * 0.4 +
            (state.adrenaline / 100) * 0.4 +
            (1 - dims['dominance']) * 0.2
        )
        
        # JOY PATTERN: High dopamine + serotonin, low cortisol
        joy_score = (
            (state.dopamine / 100) * 0.3 +
            (state.serotonin / 100) * 0.3 +
            (1 - state.cortisol / 100) * 0.2 +
            (state.oxytocin / 100) * 0.2
        )
        
        # SADNESS PATTERN: Low dopamine + serotonin, moderate cortisol
        sadness_score = (
            (1 - state.dopamine / 100) * 0.4 +
            (1 - state.serotonin / 100) * 0.4 +
            (state.cortisol / 100) * 0.2
        )
        
        # Find dominant emotional pattern
        emotions = {
            'angry': anger_score,
            'fearful': fear_score,
            'joyful': joy_score,
            'sad': sadness_score
        }
        
        # Get the strongest emotion if above threshold
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        
        if max_emotion[1] > 0.6:  # Strong emotional state
            primary_mood = max_emotion[0]
            
            # Add intensity modifiers
            if max_emotion[1] > 0.8:
                if primary_mood == 'angry':
                    return 'furious' if state.adrenaline > 70 else 'very angry'
                elif primary_mood == 'fearful':
                    return 'terrified' if state.adrenaline > 80 else 'very afraid'
                elif primary_mood == 'joyful':
                    return 'ecstatic' if state.dopamine > 80 else 'very happy'
                elif primary_mood == 'sad':
                    return 'depressed' if state.dopamine < 20 else 'very sad'
            else:
                return primary_mood
        
        # If no strong pattern, describe by dimensions
        descriptions = []
        
        # Arousal-based states
        if dims['arousal'] > 0.7:
            if dims['valence'] < -0.2:
                descriptions.append('agitated')
            elif dims['valence'] > 0.2:
                descriptions.append('excited')
            else:
                descriptions.append('highly aroused')
        elif dims['arousal'] > 0.5:
            if dims['valence'] < -0.2:
                descriptions.append('tense')
            elif dims['valence'] > 0.2:
                descriptions.append('energetic')
            else:
                descriptions.append('alert')
        elif dims['arousal'] < 0.3:
            if dims['valence'] < -0.2:
                descriptions.append('lethargic')
            elif dims['valence'] > 0.2:
                descriptions.append('relaxed')
            else:
                descriptions.append('calm')
        
        # Valence modifiers
        if dims['valence'] < -0.5:
            descriptions.append('distressed')
        elif dims['valence'] > 0.5:
            descriptions.append('content')
        
        # Social dimension
        if dims['social_orientation'] < 0.3:
            descriptions.append('withdrawn')
        elif dims['social_orientation'] > 0.7:
            descriptions.append('socially engaged')
        
        # Stability
        if dims['stability'] < 0.3:
            descriptions.append('unstable')
        
        # Combine descriptions
        if not descriptions:
            return 'neutral'
        elif len(descriptions) == 1:
            return descriptions[0]
        else:
            return ' and '.join(descriptions[:2])
    
    @staticmethod
    def get_behavioral_tendencies(state) -> Dict[str, float]:
        """Behavioral tendencies emerge from neurochemical state"""
        
        dims = MoodEmergence.get_emotional_vector(state)
        
        # Approach vs Avoid
        approach_tendency = dims['valence'] * 0.5 + (state.dopamine - 50) / 100
        
        # Fight/Flight/Freeze responses
        if dims['arousal'] > 0.6:
            # High arousal triggers action
            if state.cortisol > 60 and state.adrenaline > 50:
                if state.oxytocin < 30 and state.serotonin < 40:
                    # Low social hormones + high stress = fight
                    fight_tendency = 0.8
                    flight_tendency = 0.2
                else:
                    # Otherwise flight
                    fight_tendency = 0.2
                    flight_tendency = 0.8
            else:
                fight_tendency = dims['dominance'] * 0.5
                flight_tendency = (1 - dims['dominance']) * 0.5
        else:
            # Low arousal = freeze
            fight_tendency = 0
            flight_tendency = 0
        
        # Exploration
        explore_tendency = (state.dopamine / 100) * (1 - state.cortisol / 100)
        
        # Social engagement
        engage_socially = dims['social_orientation']
        
        # Cognitive style
        analytical_tendency = state.cortisol / 100 if state.cortisol > 40 else 0.3
        intuitive_tendency = 1 - analytical_tendency
        
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
        """Minimal prompt - just internal state"""
        mood = MoodEmergence.describe_emergent_state(state)
        
        # Simplified hormone display
        levels = (
            f"D{int(state.dopamine)}"
            f"C{int(state.cortisol)}"
            f"A{int(state.adrenaline)}"
            f"S{int(state.serotonin)}"
            f"O{int(state.oxytocin)}"
        )
        
        return f"[{mood}][{levels}]"
    
    @staticmethod
    def get_capability_triggers(state) -> List[str]:
        """What capabilities should activate based on state"""
        
        triggers = []
        
        # Deep analysis when cortisol is elevated but not panicked
        if 40 < state.cortisol < 70 and state.adrenaline < 50:
            triggers.append('deep_analysis')
        
        # Creative mode when dopamine high, cortisol low
        if state.dopamine > 60 and state.cortisol < 40:
            triggers.append('creative_exploration')
        
        # Urgent response when both cortisol and adrenaline high
        if state.cortisol > 60 and state.adrenaline > 60:
            triggers.append('urgent_response')
        
        # Empathetic mode when oxytocin high
        if state.oxytocin > 60 and state.serotonin > 50:
            triggers.append('empathetic_engagement')
        
        # Defensive mode when threatened (high cortisol, low social hormones)
        if state.cortisol > 70 and state.oxytocin < 30:
            triggers.append('defensive_mode')
        
        return triggers
