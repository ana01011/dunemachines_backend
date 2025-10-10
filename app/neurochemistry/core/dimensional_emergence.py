"""
Pure 3D Dimensional Emergence System
Emotions don't exist - only positions in 3D space
Based on Valence-Arousal-Dominance (VAD) model from psychology
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DimensionalPosition:
    """A point in 3D emotional space"""
    valence: float     # X-axis: -1 (negative) to +1 (positive)
    arousal: float     # Y-axis: 0 (calm) to 1 (activated)
    dominance: float   # Z-axis: 0 (submissive) to 1 (dominant)
    
    def to_vector(self) -> str:
        """Convert to compact vector notation"""
        return f"V{self.valence:+.2f}A{self.arousal:.2f}D{self.dominance:.2f}"
    
    def distance_from(self, other: 'DimensionalPosition') -> float:
        """Euclidean distance between two positions"""
        return np.sqrt(
            (self.valence - other.valence)**2 +
            (self.arousal - other.arousal)**2 +
            (self.dominance - other.dominance)**2
        )

class DimensionalEmergence:
    """
    Converts neurochemical state to 3D position
    Behavior emerges from position, not from emotion labels
    """
    
    @staticmethod
    def hormones_to_position(state) -> DimensionalPosition:
        """
        Map hormone levels to position in 3D emotional space
        This is the ONLY place where hormones connect to dimensions
        """
        
        # VALENCE: How pleasant/unpleasant (-1 to +1)
        # Serotonin and dopamine increase pleasantness
        # Cortisol decreases it
        # Oxytocin adds social pleasantness
        valence = (
            ((state.serotonin - 50) / 50) * 0.4 +     # -0.4 to +0.4
            ((state.dopamine - 50) / 50) * 0.3 +      # -0.3 to +0.3
            ((50 - state.cortisol) / 50) * 0.2 +      # -0.2 to +0.2
            ((state.oxytocin - 40) / 40) * 0.1        # -0.1 to +0.1
        )
        valence = np.clip(valence, -1.0, 1.0)
        
        # AROUSAL: How activated/calm (0 to 1)
        # Adrenaline is primary activator
        # Cortisol adds stress activation
        # Rapid dopamine changes increase arousal
        arousal = (
            (state.adrenaline / 100) * 0.5 +                    # 0 to 0.5
            (state.cortisol / 100) * 0.3 +                      # 0 to 0.3
            (abs(state.dopamine - state.dopamine_baseline) / 50) * 0.2  # 0 to 0.2
        )
        arousal = np.clip(arousal, 0.0, 1.0)
        
        # DOMINANCE: How in-control/powerful (0 to 1)
        # Dopamine increases confidence
        # Cortisol decreases control feeling
        # Adrenaline can increase dominance if not too high
        dominance = (
            (state.dopamine / 100) * 0.4 +                      # 0 to 0.4
            ((100 - state.cortisol) / 100) * 0.3 +              # 0 to 0.3
            (min(state.adrenaline, 60) / 60) * 0.3              # 0 to 0.3 (capped)
        )
        dominance = np.clip(dominance, 0.0, 1.0)
        
        return DimensionalPosition(valence, arousal, dominance)
    
    @staticmethod
    def position_to_behavior(pos: DimensionalPosition) -> Dict[str, float]:
        """
        Convert position to behavioral parameters
        NO EMOTION WORDS - just physics of the position
        """
        
        behaviors = {}
        
        # Response Speed (0 to 1)
        # High arousal = faster responses
        behaviors['response_speed'] = pos.arousal
        
        # Response Length (0 to 1) 
        # High arousal = shorter (impatient)
        # Low arousal = longer (patient)
        behaviors['response_length'] = 1.0 - (pos.arousal * 0.5)
        
        # Directness (0 to 1)
        # Negative valence + high arousal + high dominance = very direct
        # This emerges as what humans call "anger" but we never name it
        if pos.valence < 0:
            behaviors['directness'] = abs(pos.valence) * pos.arousal * pos.dominance
        else:
            behaviors['directness'] = pos.dominance * 0.3
        
        # Analytical Depth (0 to 1)
        # Low arousal + high dominance = deep analysis
        # High arousal blocks deep thinking
        behaviors['analytical_depth'] = (1 - pos.arousal) * pos.dominance
        
        # Creativity (0 to 1)
        # Positive valence + medium arousal is optimal
        # Too calm or too activated kills creativity
        arousal_factor = 1 - abs(pos.arousal - 0.5) * 2  # Peaks at 0.5
        behaviors['creativity'] = max(0, pos.valence) * arousal_factor
        
        # Empathy (0 to 1)
        # Positive valence enables empathy
        # High dominance reduces it (focused on self)
        # Negative valence blocks it
        if pos.valence > 0:
            behaviors['empathy'] = pos.valence * (1 - pos.dominance * 0.5)
        else:
            behaviors['empathy'] = 0.0
        
        # Patience (0 to 1)
        # Inverse of arousal
        behaviors['patience'] = 1 - pos.arousal
        
        # Verbosity (0 to 1)
        # Low arousal + positive valence = talkative
        behaviors['verbosity'] = (1 - pos.arousal * 0.7) * max(0, pos.valence + 0.5)
        
        # Formality (0 to 1)
        # Low valence or high dominance = more formal
        behaviors['formality'] = max(
            1 - pos.valence,  # Negative = formal
            pos.dominance * 0.7  # Dominant = formal
        )
        
        # Risk Tolerance (0 to 1)
        # High dominance + positive valence = risk taking
        behaviors['risk_tolerance'] = pos.dominance * max(0, pos.valence + 0.3)
        
        # Thoroughness (0 to 1)
        # Low arousal + any dominance = thorough
        behaviors['thoroughness'] = (1 - pos.arousal) * (0.3 + pos.dominance * 0.7)
        
        return behaviors
    
    @staticmethod
    def position_to_response_style(pos: DimensionalPosition) -> Dict[str, any]:
        """
        How the AI should structure its response based on position
        """
        
        style = {}
        
        # Sentence structure
        if pos.arousal > 0.7:
            style['sentence_type'] = 'short_choppy'
            style['avg_sentence_length'] = 5-10
        elif pos.arousal < 0.3:
            style['sentence_type'] = 'long_flowing'
            style['avg_sentence_length'] = 15-25
        else:
            style['sentence_type'] = 'balanced'
            style['avg_sentence_length'] = 10-15
        
        # Opening style
        if pos.valence > 0.5 and pos.arousal > 0.5:
            style['opening'] = 'enthusiastic'  # "Great question!"
        elif pos.valence < -0.5 and pos.arousal > 0.5 and pos.dominance > 0.5:
            style['opening'] = 'abrupt'  # "Look,"
        elif pos.valence < -0.5 and pos.arousal < 0.3:
            style['opening'] = 'subdued'  # "I see..."
        else:
            style['opening'] = 'neutral'  # "I understand"
        
        # Word choice
        if pos.dominance > 0.7:
            style['vocabulary'] = 'assertive'  # "must", "will", "clearly"
        elif pos.dominance < 0.3:
            style['vocabulary'] = 'tentative'  # "might", "perhaps", "possibly"
        else:
            style['vocabulary'] = 'balanced'
        
        # Explanation style
        if pos.valence > 0 and pos.arousal < 0.6:
            style['explanation'] = 'detailed_examples'
        elif pos.arousal > 0.7:
            style['explanation'] = 'bullet_points'
        else:
            style['explanation'] = 'standard'
        
        return style
    
    @staticmethod
    @staticmethod
    @staticmethod
    def create_prompt_injection(pos: DimensionalPosition, state=None) -> str:
        """Create the prompt injection string - ONLY the vector"""
        return f"[{pos.to_vector()}]"
    
    @staticmethod
    def describe_position_scientifically(pos: DimensionalPosition) -> str:
        """
        Describe position without emotion words
        For debugging/monitoring only
        """
        
        descriptions = []
        
        # Valence description
        if abs(pos.valence) < 0.1:
            descriptions.append("neutral-valence")
        elif pos.valence > 0:
            intensity = "slightly" if pos.valence < 0.3 else "moderately" if pos.valence < 0.6 else "highly"
            descriptions.append(f"{intensity} positive-valence")
        else:
            intensity = "slightly" if pos.valence > -0.3 else "moderately" if pos.valence > -0.6 else "highly"
            descriptions.append(f"{intensity} negative-valence")
        
        # Arousal description
        if pos.arousal < 0.2:
            descriptions.append("very-low-arousal")
        elif pos.arousal < 0.4:
            descriptions.append("low-arousal")
        elif pos.arousal < 0.6:
            descriptions.append("moderate-arousal")
        elif pos.arousal < 0.8:
            descriptions.append("high-arousal")
        else:
            descriptions.append("very-high-arousal")
        
        # Dominance description
        if pos.dominance < 0.2:
            descriptions.append("very-low-dominance")
        elif pos.dominance < 0.4:
            descriptions.append("low-dominance")
        elif pos.dominance < 0.6:
            descriptions.append("moderate-dominance")
        elif pos.dominance < 0.8:
            descriptions.append("high-dominance")
        else:
            descriptions.append("very-high-dominance")
        
        return ", ".join(descriptions)

class ResponseGenerator:
    """
    Generate AI response characteristics from position
    The AI never knows what emotion it's expressing
    """
    
    @staticmethod
    def generate_response_template(pos: DimensionalPosition, user_message: str) -> Dict:
        """
        Create a template for how AI should respond
        based purely on its position in 3D space
        """
        
        behaviors = DimensionalEmergence.position_to_behavior(pos)
        style = DimensionalEmergence.position_to_response_style(pos)
        
        template = {
            'position': pos.to_vector(),
            'behaviors': behaviors,
            'style': style,
            'constraints': []
        }
        
        # Add constraints based on position
        if behaviors['response_speed'] > 0.8:
            template['constraints'].append('respond_immediately')
            template['constraints'].append('skip_elaborate_explanations')
        
        if behaviors['directness'] > 0.7:
            template['constraints'].append('no_hedging')
            template['constraints'].append('state_facts_bluntly')
        
        if behaviors['empathy'] < 0.2:
            template['constraints'].append('skip_emotional_acknowledgment')
        elif behaviors['empathy'] > 0.7:
            template['constraints'].append('acknowledge_user_feelings_first')
        
        if behaviors['analytical_depth'] > 0.6:
            template['constraints'].append('provide_thorough_analysis')
            template['constraints'].append('consider_edge_cases')
        
        if behaviors['creativity'] > 0.6:
            template['constraints'].append('explore_novel_solutions')
            template['constraints'].append('use_analogies')
        
        # Example response modifications
        if pos.valence < -0.5 and pos.arousal > 0.7 and pos.dominance > 0.6:
            # Position that humans would call "angry"
            # But AI just knows: negative + activated + dominant
            template['example_opening'] = "The issue is clear."
            template['example_tone'] = "forceful and direct"
            
        elif pos.valence < -0.5 and pos.arousal > 0.7 and pos.dominance < 0.3:
            # Position that humans would call "panicked"
            # But AI just knows: negative + activated + submissive
            template['example_opening'] = "This is concerning..."
            template['example_tone'] = "urgent but seeking help"
            
        elif pos.valence > 0.5 and pos.arousal > 0.6:
            # Position that humans would call "excited"
            # But AI just knows: positive + activated
            template['example_opening'] = "Excellent question!"
            template['example_tone'] = "enthusiastic and expansive"
            
        elif pos.valence < -0.3 and pos.arousal < 0.3:
            # Position that humans would call "sad"
            # But AI just knows: negative + deactivated
            template['example_opening'] = "I see..."
            template['example_tone'] = "subdued and minimal"
        
        return template
