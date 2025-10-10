"""
Generate comprehensive biologically-inspired training data for hormone-to-mood network
Based on real neuroscience research and covers all emotional states
"""

import numpy as np
import torch
import json
from typing import List, Tuple, Dict

class BiologicalTrainingDataGenerator:
    """
    Generates biologically accurate hormone-mood mappings
    """
    
    def __init__(self):
        self.num_hormones = 5  # D, C, A, S, O
        self.num_moods = 30
        self.data = []
        
    def generate_complete_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate comprehensive training dataset based on neuroscience
        """
        print("Generating biologically-grounded training data...")
        
        # 1. Core emotional states from research
        self._add_researched_states()
        
        # 2. Transition states
        self._add_transition_states()
        
        # 3. Complex mixed states
        self._add_mixed_states()
        
        # 4. Extreme states (rare but possible)
        self._add_extreme_states()
        
        # 5. Pathological states
        self._add_pathological_states()
        
        # 6. Daily life variations
        self._add_daily_variations()
        
        # 7. Stress response patterns
        self._add_stress_responses()
        
        # 8. Social bonding patterns
        self._add_social_patterns()
        
        # 9. Reward patterns
        self._add_reward_patterns()
        
        # 10. Circadian variations
        self._add_circadian_patterns()
        
        # Convert to tensors
        inputs = torch.tensor([d[0] for d in self.data], dtype=torch.float32)
        targets = torch.tensor([d[1] for d in self.data], dtype=torch.float32)
        
        print(f"Generated {len(self.data)} biologically-grounded samples")
        return inputs, targets
    
    def _add_researched_states(self):
        """Add emotional states documented in neuroscience research"""
        
        # HAPPINESS/JOY - High dopamine, high serotonin, low cortisol
        for variation in range(10):
            d = 70 + np.random.normal(0, 5)
            c = 20 + np.random.normal(0, 3)
            a = 30 + np.random.normal(0, 5)
            s = 75 + np.random.normal(0, 5)
            o = 60 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.6, energized=0.7, tense=0.1, assured=0.8, warm=0.9,
                restless=0.2, protective=0.1, curious=0.8, determined=0.7, cautious=0.1,
                reflective=0.4, spontaneous=0.7, methodical=0.3, receptive=0.8, assertive=0.6,
                conflicted=0.1, transitioning=0.2, anticipating=0.6, processing=0.3, resonating=0.8
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # FEAR - Low dopamine, high cortisol, high adrenaline
        for variation in range(10):
            d = 25 + np.random.normal(0, 5)
            c = 80 + np.random.normal(0, 5)
            a = 85 + np.random.normal(0, 5)
            s = 30 + np.random.normal(0, 5)
            o = 20 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.95, energized=0.8, tense=0.9, assured=0.1, warm=0.1,
                restless=0.9, protective=0.95, curious=0.2, determined=0.3, cautious=0.95,
                reflective=0.2, spontaneous=0.7, methodical=0.1, receptive=0.2, assertive=0.1,
                conflicted=0.6, transitioning=0.8, anticipating=0.9, processing=0.6, resonating=0.2
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # ANGER - Moderate-high dopamine, high cortisol, high adrenaline
        for variation in range(10):
            d = 60 + np.random.normal(0, 5)
            c = 75 + np.random.normal(0, 5)
            a = 80 + np.random.normal(0, 5)
            s = 35 + np.random.normal(0, 5)
            o = 15 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.9, energized=0.85, tense=0.85, assured=0.7, warm=0.1,
                restless=0.8, protective=0.7, curious=0.1, determined=0.85, cautious=0.2,
                reflective=0.1, spontaneous=0.6, methodical=0.2, receptive=0.1, assertive=0.9,
                conflicted=0.4, transitioning=0.3, anticipating=0.2, processing=0.2, resonating=0.1
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # SADNESS - Low everything except moderate cortisol
        for variation in range(10):
            d = 20 + np.random.normal(0, 5)
            c = 45 + np.random.normal(0, 5)
            a = 15 + np.random.normal(0, 5)
            s = 25 + np.random.normal(0, 5)
            o = 30 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.3, energized=0.15, tense=0.4, assured=0.15, warm=0.25,
                restless=0.2, protective=0.3, curious=0.15, determined=0.2, cautious=0.4,
                reflective=0.8, spontaneous=0.1, methodical=0.4, receptive=0.5, assertive=0.15,
                conflicted=0.5, transitioning=0.3, anticipating=0.2, processing=0.6, resonating=0.6
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # DISGUST - Low dopamine, moderate cortisol, moderate adrenaline, low serotonin
        for variation in range(10):
            d = 25 + np.random.normal(0, 5)
            c = 55 + np.random.normal(0, 5)
            a = 45 + np.random.normal(0, 5)
            s = 30 + np.random.normal(0, 5)
            o = 20 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.6, energized=0.4, tense=0.6, assured=0.3, warm=0.1,
                restless=0.5, protective=0.7, curious=0.1, determined=0.4, cautious=0.7,
                reflective=0.3, spontaneous=0.2, methodical=0.3, receptive=0.1, assertive=0.4,
                conflicted=0.6, transitioning=0.4, anticipating=0.2, processing=0.5, resonating=0.1
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # SURPRISE - Spike in dopamine and adrenaline
        for variation in range(10):
            d = 70 + np.random.normal(0, 5)
            c = 40 + np.random.normal(0, 5)
            a = 75 + np.random.normal(0, 5)
            s = 50 + np.random.normal(0, 5)
            o = 45 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.95, energized=0.8, tense=0.4, assured=0.5, warm=0.5,
                restless=0.6, protective=0.3, curious=0.95, determined=0.5, cautious=0.4,
                reflective=0.2, spontaneous=0.9, methodical=0.1, receptive=0.9, assertive=0.5,
                conflicted=0.3, transitioning=0.7, anticipating=0.8, processing=0.7, resonating=0.5
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_transition_states(self):
        """Add states that represent transitions between emotions"""
        
        # Calm to anxious transition
        for t in np.linspace(0, 1, 20):
            d = 50 - 20*t
            c = 30 + 50*t
            a = 25 + 45*t
            s = 65 - 25*t
            o = 50 - 20*t
            
            moods = self._interpolate_moods(t, 'calm', 'anxious')
            self.data.append(([d, c, a, s, o], moods))
        
        # Happy to sad transition
        for t in np.linspace(0, 1, 20):
            d = 75 - 55*t
            c = 20 + 25*t
            a = 30 - 15*t
            s = 80 - 55*t
            o = 65 - 35*t
            
            moods = self._interpolate_moods(t, 'happy', 'sad')
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_mixed_states(self):
        """Add complex mixed emotional states"""
        
        # Excited but anxious (before presentation)
        for variation in range(10):
            d = 65 + np.random.normal(0, 5)
            c = 60 + np.random.normal(0, 5)
            a = 70 + np.random.normal(0, 5)
            s = 45 + np.random.normal(0, 5)
            o = 35 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.9, energized=0.8, tense=0.7, assured=0.5, warm=0.4,
                restless=0.75, protective=0.5, curious=0.6, determined=0.7, cautious=0.6,
                reflective=0.3, spontaneous=0.5, methodical=0.4, receptive=0.5, assertive=0.6,
                conflicted=0.8, transitioning=0.6, anticipating=0.9, processing=0.7, resonating=0.4
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Tired but content (after accomplishment)
        for variation in range(10):
            d = 55 + np.random.normal(0, 5)
            c = 25 + np.random.normal(0, 5)
            a = 20 + np.random.normal(0, 5)
            s = 70 + np.random.normal(0, 5)
            o = 60 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.3, energized=0.2, tense=0.15, assured=0.7, warm=0.75,
                restless=0.1, protective=0.2, curious=0.3, determined=0.3, cautious=0.2,
                reflective=0.7, spontaneous=0.2, methodical=0.5, receptive=0.6, assertive=0.3,
                conflicted=0.1, transitioning=0.2, anticipating=0.3, processing=0.5, resonating=0.7
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_extreme_states(self):
        """Add rare but possible extreme states"""
        
        # Manic state - everything elevated
        for variation in range(5):
            d = 85 + np.random.normal(0, 3)
            c = 70 + np.random.normal(0, 5)
            a = 80 + np.random.normal(0, 5)
            s = 60 + np.random.normal(0, 5)
            o = 50 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.95, energized=0.95, tense=0.7, assured=0.8, warm=0.6,
                restless=0.9, protective=0.4, curious=0.9, determined=0.9, cautious=0.1,
                reflective=0.1, spontaneous=0.95, methodical=0.1, receptive=0.7, assertive=0.85,
                conflicted=0.6, transitioning=0.8, anticipating=0.9, processing=0.9, resonating=0.5,
                # AI-unique states
                parallel_processing=0.9, recursive_depth=0.7, quantum_uncertain=0.6
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Deep depression - everything low
        for variation in range(5):
            d = 10 + np.random.normal(0, 2)
            c = 60 + np.random.normal(0, 5)
            a = 10 + np.random.normal(0, 2)
            s = 15 + np.random.normal(0, 3)
            o = 20 + np.random.normal(0, 3)
            
            moods = self._create_mood_vector(
                attentive=0.15, energized=0.05, tense=0.5, assured=0.05, warm=0.1,
                restless=0.3, protective=0.4, curious=0.05, determined=0.1, cautious=0.6,
                reflective=0.9, spontaneous=0.05, methodical=0.2, receptive=0.2, assertive=0.05,
                conflicted=0.7, transitioning=0.2, anticipating=0.1, processing=0.8, resonating=0.3
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_pathological_states(self):
        """Add pathological states for completeness"""
        
        # Panic attack
        for variation in range(5):
            d = 15 + np.random.normal(0, 3)
            c = 95 + np.random.normal(0, 2)
            a = 98 + np.random.normal(0, 1)
            s = 20 + np.random.normal(0, 3)
            o = 10 + np.random.normal(0, 2)
            
            moods = self._create_mood_vector(
                attentive=1.0, energized=0.95, tense=1.0, assured=0.0, warm=0.05,
                restless=1.0, protective=1.0, curious=0.0, determined=0.2, cautious=1.0,
                reflective=0.0, spontaneous=0.8, methodical=0.0, receptive=0.1, assertive=0.0,
                conflicted=0.9, transitioning=1.0, anticipating=1.0, processing=0.3, resonating=0.1,
                # Extreme AI states
                parallel_processing=-0.5, quantum_uncertain=1.0, entropy_sensing=1.0
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_daily_variations(self):
        """Add normal daily hormone variations"""
        
        # Morning awakening
        for hour in range(6, 10):
            d = 30 + hour*5
            c = 50 - hour*3
            a = 25 + hour*2
            s = 50 + hour*2
            o = 40
            
            alertness = (hour - 6) / 4
            moods = self._create_mood_vector(
                attentive=0.3 + alertness*0.5,
                energized=0.2 + alertness*0.4,
                tense=0.3 - alertness*0.1,
                assured=0.5 + alertness*0.2,
                warm=0.5 + alertness*0.1
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Afternoon peak
        for variation in range(10):
            d = 60 + np.random.normal(0, 5)
            c = 35 + np.random.normal(0, 5)
            a = 40 + np.random.normal(0, 5)
            s = 65 + np.random.normal(0, 5)
            o = 50 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.8, energized=0.7, tense=0.3, assured=0.7, warm=0.6,
                productive=0.8, focused=0.75
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Evening wind-down
        for hour in range(20, 24):
            d = 50 - (hour-20)*8
            c = 25 - (hour-20)*3
            a = 30 - (hour-20)*5
            s = 70 + (hour-20)*3
            o = 55 + (hour-20)*2
            
            tiredness = (hour - 20) / 4
            moods = self._create_mood_vector(
                attentive=0.6 - tiredness*0.4,
                energized=0.5 - tiredness*0.4,
                tense=0.2 - tiredness*0.1,
                assured=0.6,
                warm=0.6 + tiredness*0.2,
                reflective=0.4 + tiredness*0.3
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_stress_responses(self):
        """Add acute and chronic stress patterns"""
        
        # Acute stress response
        for intensity in np.linspace(0, 1, 15):
            d = 40 - 10*intensity
            c = 40 + 50*intensity
            a = 35 + 55*intensity
            s = 55 - 25*intensity
            o = 45 - 20*intensity
            
            moods = self._create_mood_vector(
                attentive=0.5 + 0.5*intensity,
                energized=0.4 + 0.5*intensity,
                tense=0.2 + 0.7*intensity,
                assured=0.6 - 0.5*intensity,
                cautious=0.3 + 0.6*intensity
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Chronic stress (burnout)
        for day in range(30):
            d = 35 - day*0.5
            c = 65 + day*0.5
            a = 40 - day*0.3
            s = 40 - day*0.5
            o = 35 - day*0.3
            
            moods = self._create_mood_vector(
                attentive=0.6 - day*0.01,
                energized=0.5 - day*0.01,
                tense=0.6 + day*0.01,
                assured=0.4 - day*0.01,
                warm=0.4 - day*0.01,
                exhausted=0.2 + day*0.02
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_social_patterns(self):
        """Add social bonding and interaction patterns"""
        
        # Positive social interaction
        for variation in range(10):
            d = 60 + np.random.normal(0, 5)
            c = 25 + np.random.normal(0, 3)
            a = 35 + np.random.normal(0, 5)
            s = 70 + np.random.normal(0, 5)
            o = 80 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.7, energized=0.6, tense=0.15, assured=0.7, warm=0.9,
                curious=0.7, receptive=0.85, resonating=0.9, empathetic=0.85
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Social anxiety
        for variation in range(10):
            d = 30 + np.random.normal(0, 5)
            c = 65 + np.random.normal(0, 5)
            a = 60 + np.random.normal(0, 5)
            s = 35 + np.random.normal(0, 5)
            o = 25 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.8, energized=0.5, tense=0.75, assured=0.2, warm=0.3,
                cautious=0.85, protective=0.8, conflicted=0.7, self_conscious=0.9
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_reward_patterns(self):
        """Add reward and achievement patterns"""
        
        # Anticipation of reward
        for variation in range(10):
            d = 70 + np.random.normal(0, 5)
            c = 35 + np.random.normal(0, 5)
            a = 50 + np.random.normal(0, 5)
            s = 55 + np.random.normal(0, 5)
            o = 45 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.85, energized=0.7, tense=0.35, assured=0.65, warm=0.6,
                curious=0.8, anticipating=0.95, determined=0.75, optimistic=0.8
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Achievement/Success
        for variation in range(10):
            d = 85 + np.random.normal(0, 3)
            c = 20 + np.random.normal(0, 3)
            a = 45 + np.random.normal(0, 5)
            s = 80 + np.random.normal(0, 5)
            o = 70 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.6, energized=0.75, tense=0.1, assured=0.9, warm=0.85,
                proud=0.9, satisfied=0.85, confident=0.85, accomplished=0.9
            )
            self.data.append(([d, c, a, s, o], moods))
        
        # Disappointment/Failure
        for variation in range(10):
            d = 25 + np.random.normal(0, 5)
            c = 55 + np.random.normal(0, 5)
            a = 30 + np.random.normal(0, 5)
            s = 35 + np.random.normal(0, 5)
            o = 35 + np.random.normal(0, 5)
            
            moods = self._create_mood_vector(
                attentive=0.4, energized=0.3, tense=0.5, assured=0.2, warm=0.3,
                determined=0.4, cautious=0.6, reflective=0.7, disappointed=0.8
            )
            self.data.append(([d, c, a, s, o], moods))
    
    def _add_circadian_patterns(self):
        """Add circadian rhythm patterns"""
        
        for hour in range(24):
            # Circadian cortisol rhythm
            if 6 <= hour <= 8:
                c = 70 + (hour-6)*5  # Morning peak
            elif 8 < hour <= 22:
                c = 80 - (hour-8)*3  # Gradual decline
            else:
                c = 20  # Night low
            
            # Other hormones follow patterns
            d = 50 + 20*np.sin((hour-14)*np.pi/12)  # Peak mid-afternoon
            a = 30 + 10*np.sin((hour-10)*np.pi/12)  # Peak mid-morning
            s = 60 + 15*np.sin((hour-20)*np.pi/12)  # Peak evening
            o = 45 + 10*np.sin((hour-21)*np.pi/12)  # Peak late evening
            
            # Add variations
            for variation in range(3):
                d_var = d + np.random.normal(0, 5)
                c_var = c + np.random.normal(0, 3)
                a_var = a + np.random.normal(0, 3)
                s_var = s + np.random.normal(0, 5)
                o_var = o + np.random.normal(0, 5)
                
                moods = self._calculate_circadian_mood(hour, d_var, c_var, a_var, s_var, o_var)
                self.data.append(([d_var, c_var, a_var, s_var, o_var], moods))
    
    def _create_mood_vector(self, **kwargs):
        """Create a mood vector with specified component values"""
        moods = np.zeros(30)
        
        component_map = {
            'attentive': 0, 'energized': 1, 'tense': 2, 'assured': 3, 'warm': 4,
            'restless': 5, 'protective': 6, 'curious': 7, 'determined': 8, 'cautious': 9,
            'reflective': 10, 'spontaneous': 11, 'methodical': 12, 'receptive': 13, 'assertive': 14,
            'conflicted': 15, 'transitioning': 16, 'anticipating': 17, 'processing': 18, 'resonating': 19,
            'parallel_processing': 20, 'recursive_depth': 21, 'quantum_uncertain': 22,
            'temporal_spread': 23, 'dimensional_shift': 24, 'pattern_crystallizing': 25,
            'entropy_sensing': 26, 'coherence_seeking': 27, 'information_hungry': 28,
            'synthesis_flowing': 29
        }
        
        for name, value in kwargs.items():
            if name in component_map:
                moods[component_map[name]] = value
        
        # Add small random noise for variation
        moods += np.random.normal(0, 0.02, 30)
        
        return moods.tolist()
    
    def _interpolate_moods(self, t, start_state, end_state):
        """Interpolate between two emotional states"""
        state_templates = {
            'calm': self._create_mood_vector(
                attentive=0.5, energized=0.3, tense=0.15, assured=0.6, warm=0.6
            ),
            'anxious': self._create_mood_vector(
                attentive=0.85, energized=0.6, tense=0.75, assured=0.25, warm=0.3
            ),
            'happy': self._create_mood_vector(
                attentive=0.6, energized=0.7, tense=0.1, assured=0.8, warm=0.9
            ),
            'sad': self._create_mood_vector(
                attentive=0.3, energized=0.15, tense=0.4, assured=0.15, warm=0.25
            )
        }
        
        start = np.array(state_templates.get(start_state, np.zeros(30)))
        end = np.array(state_templates.get(end_state, np.zeros(30)))
        
        interpolated = start * (1 - t) + end * t
        return interpolated.tolist()
    
    def _calculate_circadian_mood(self, hour, d, c, a, s, o):
        """Calculate mood based on circadian time"""
        moods = np.zeros(30)
        
        # Morning (6-12)
        if 6 <= hour < 12:
            alertness = (hour - 6) / 6
            moods[0] = 0.3 + 0.5 * alertness  # attentive
            moods[1] = 0.2 + 0.5 * alertness  # energized
            moods[12] = 0.6  # methodical (morning routine)
        
        # Afternoon (12-17)
        elif 12 <= hour < 17:
            productivity = 1 - abs(hour - 14.5) / 2.5
            moods[0] = 0.7 + 0.2 * productivity  # attentive
            moods[1] = 0.6 + 0.2 * productivity  # energized
            moods[8] = 0.7  # determined
        
        # Evening (17-22)
        elif 17 <= hour < 22:
            wind_down = (hour - 17) / 5
            moods[0] = 0.7 - 0.4 * wind_down  # attentive decreasing
            moods[1] = 0.6 - 0.4 * wind_down  # energized decreasing
            moods[10] = 0.3 + 0.4 * wind_down  # reflective increasing
            moods[4] = 0.5 + 0.3 * wind_down  # warm increasing
        
        # Night (22-6)
        else:
            moods[0] = 0.2  # low attention
            moods[1] = 0.1  # low energy
            moods[10] = 0.6  # reflective
        
        # Apply hormone influences
        moods[0] = min(1, moods[0] * (a/100 + 0.5))  # Adrenaline affects attention
        moods[2] = c / 100  # Cortisol directly maps to tension
        moods[3] = (d / 100) * (1 - c/200)  # Assured from dopamine, reduced by cortisol
        moods[4] = (s / 100) * (o / 100)  # Warmth from serotonin and oxytocin
        
        return moods.tolist()
    
    def save_dataset(self, filename='biological_training_data.pth'):
        """Save the dataset to file"""
        inputs, targets = self.generate_complete_dataset()
        
        torch.save({
            'inputs': inputs,
            'targets': targets,
            'num_samples': len(inputs),
            'description': 'Biologically-grounded hormone-to-mood training data'
        }, filename)
        
        print(f"Saved {len(inputs)} samples to {filename}")
        
        # Save sample for inspection
        with open('training_samples.json', 'w') as f:
            samples = []
            for i in range(min(20, len(self.data))):
                samples.append({
                    'hormones': {
                        'dopamine': self.data[i][0][0],
                        'cortisol': self.data[i][0][1],
                        'adrenaline': self.data[i][0][2],
                        'serotonin': self.data[i][0][3],
                        'oxytocin': self.data[i][0][4]
                    },
                    'top_moods': self._get_top_moods(self.data[i][1])
                })
            json.dump(samples, f, indent=2)
        print("Saved sample data to training_samples.json for inspection")
    
    def _get_top_moods(self, mood_vector):
        """Get top 5 mood components"""
        components = [
            'attentive', 'energized', 'tense', 'assured', 'warm',
            'restless', 'protective', 'curious', 'determined', 'cautious',
            'reflective', 'spontaneous', 'methodical', 'receptive', 'assertive',
            'conflicted', 'transitioning', 'anticipating', 'processing', 'resonating',
            'parallel_processing', 'recursive_depth', 'quantum_uncertain',
            'temporal_spread', 'dimensional_shift', 'pattern_crystallizing',
            'entropy_sensing', 'coherence_seeking', 'information_hungry', 'synthesis_flowing'
        ]
        
        top_indices = np.argsort(mood_vector)[-5:][::-1]
        return [(components[i], float(mood_vector[i])) for i in top_indices if mood_vector[i] > 0.3]


if __name__ == "__main__":
    generator = BiologicalTrainingDataGenerator()
    generator.save_dataset()
