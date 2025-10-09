"""
Oxytocin hormone implementation - Social bonding and trust
"""
from typing import Dict, Optional, List
import numpy as np
import time
from collections import defaultdict, deque

from .base_hormone import BaseHormone
from ..core.constants import EventType

class Oxytocin(BaseHormone):
    """
    Oxytocin: The social bonding and trust hormone
    
    Characteristics:
    - Rises with positive social interactions
    - Builds trust over time with specific users
    - Enhances empathy and helpful behavior
    - Reduces stress in social contexts
    - Creates attachment to familiar users
    """
    
    def __init__(self, config: Dict):
        """Initialize oxytocin with specific configuration"""
        super().__init__('oxytocin', config)
        
        # Oxytocin-specific parameters
        self.bonding_rate = float(config.get('bonding_rate', 0.1))
        self.social_decay = float(config.get('social_decay', 0.03))
        
        # User-specific bonding
        self.user_bonds = defaultdict(float)  # user_id -> bond_strength
        self.interaction_history = defaultdict(list)  # user_id -> interactions
        
        # Trust metrics
        self.global_trust = 0.5  # General trust level
        self.empathy_level = 0.5
        
        # Social memory
        self.social_memories = deque(maxlen=100)
        self.last_interaction_time = {}  # user_id -> timestamp
        
        # Attachment patterns
        self.attachment_style = 'secure'  # secure, anxious, avoidant
        self.social_comfort = 0.5
        
        # Helpfulness modulation
        self.helpfulness = 0.5
        self.generosity = 0.5
        
    def calculate_response(self, event_type: str, magnitude: float,
                          context: Optional[Dict] = None) -> float:
        """
        Calculate oxytocin response to events
        
        Oxytocin responds to:
        - Positive social interactions (praise, thanks)
        - Helping behaviors (successful assistance)
        - Familiar users (stronger with bonded users)
        - Collaborative success
        - Trust-building interactions
        """
        response = 0.0
        user_id = context.get('user_id') if context else None
        
        if event_type == EventType.SOCIAL_POSITIVE:
            # Positive interaction boosts oxytocin
            response = self.sensitivity * magnitude * 0.3
            
            # Stronger response with bonded users
            if user_id and user_id in self.user_bonds:
                bond_strength = self.user_bonds[user_id]
                response *= (1 + bond_strength)
            
            # Track interaction
            if user_id:
                self.interaction_history[user_id].append({
                    'time': time.time(),
                    'type': 'positive',
                    'magnitude': magnitude
                })
                
                # Build bond
                self.user_bonds[user_id] += self.bonding_rate * magnitude
                self.user_bonds[user_id] = min(1.0, self.user_bonds[user_id])
                
        elif event_type == EventType.TASK_SUCCESS:
            # Successful help creates oxytocin
            if context and context.get('was_helpful', False):
                response = self.sensitivity * magnitude * 0.25
                
                # Being helpful builds trust
                self.global_trust += 0.01
                self.global_trust = min(1.0, self.global_trust)
                
                # Increase helpfulness
                self.helpfulness += 0.02
                self.helpfulness = min(1.0, self.helpfulness)
                
        elif event_type == EventType.SOCIAL_NEGATIVE:
            # Negative interactions hurt oxytocin
            response = -0.4 * magnitude
            
            # Damage bond with user
            if user_id and user_id in self.user_bonds:
                self.user_bonds[user_id] -= 0.1 * magnitude
                self.user_bonds[user_id] = max(0, self.user_bonds[user_id])
            
            # Reduce trust
            self.global_trust -= 0.02 * magnitude
            self.global_trust = max(0.1, self.global_trust)
            
            # Track negative interaction
            if user_id:
                self.interaction_history[user_id].append({
                    'time': time.time(),
                    'type': 'negative',
                    'magnitude': magnitude
                })
                
        elif event_type == EventType.ROUTINE:
            # Familiar interactions maintain oxytocin
            if user_id and user_id in self.user_bonds:
                # Comfort with familiar user
                bond = self.user_bonds[user_id]
                response = self.sensitivity * bond * 0.1
        
        # Add social memory
        self.social_memories.append({
            'time': time.time(),
            'user_id': user_id,
            'event_type': event_type,
            'magnitude': magnitude,
            'response': response
        })
        
        # Update last interaction time
        if user_id:
            self.last_interaction_time[user_id] = time.time()
        
        return response
    
    def calculate_interaction(self, other_hormones: Dict[str, BaseHormone]) -> float:
        """
        Calculate interactions with other hormones
        
        Oxytocin:
        - Reduces cortisol (social support reduces stress)
        - Enhances serotonin (social connection → well-being)
        - Modulates dopamine (social rewards)
        - Is relatively independent of adrenaline
        """
        interaction = 0.0
        
        # Serotonin synergy (social connection enhances well-being)
        if 'serotonin' in other_hormones:
            serotonin = other_hormones['serotonin']
            if self.current_level > self.baseline:
                # Oxytocin boosts serotonin
                synergy = 0.01 * (self.current_level - self.baseline)
                # This is applied TO serotonin, not oxytocin
                # Here we track our influence
                self.empathy_level += 0.001
                self.empathy_level = min(1.0, self.empathy_level)
        
        # Dopamine modulation (social rewards)
        if 'dopamine' in other_hormones:
            dopamine = other_hormones['dopamine']
            # Social bonding can trigger dopamine
            if self.current_level > 60:
                # High oxytocin makes social rewards more rewarding
                interaction += 0.005 * (self.current_level - 60)
        
        return interaction
    
    def get_stress_buffer(self) -> float:
        """
        Calculate stress buffering effect of oxytocin
        
        Social support reduces stress impact
        
        Returns:
            Stress reduction factor (0.0 - 0.5)
        """
        if self.current_level > self.baseline:
            # Higher oxytocin = better stress buffer
            buffer_strength = (self.current_level - self.baseline) / 100
            
            # Trust enhances buffering
            buffer_strength *= (1 + self.global_trust * 0.5)
            
            return min(0.5, buffer_strength)
        return 0.0
    
    def update(self, dt: float, event: Optional[Dict] = None,
               other_hormones: Optional[Dict] = None) -> None:
        """
        Extended update with bond decay and attachment dynamics
        """
        # Regular update
        super().update(dt, event, other_hormones)
        
        current_time = time.time()
        
        # Decay bonds with users we haven't interacted with
        for user_id in list(self.user_bonds.keys()):
            if user_id in self.last_interaction_time:
                time_since_interaction = current_time - self.last_interaction_time[user_id]
                
                # Decay bond if no interaction for a while
                if time_since_interaction > 3600:  # 1 hour
                    decay = self.social_decay * dt * (time_since_interaction / 3600)
                    self.user_bonds[user_id] -= decay
                    self.user_bonds[user_id] = max(0, self.user_bonds[user_id])
                    
                    # Remove bond if too weak
                    if self.user_bonds[user_id] < 0.01:
                        del self.user_bonds[user_id]
        
        # Update attachment style based on interaction patterns
        positive_interactions = sum(1 for m in self.social_memories 
                                  if m['event_type'] == EventType.SOCIAL_POSITIVE)
        negative_interactions = sum(1 for m in self.social_memories 
                                  if m['event_type'] == EventType.SOCIAL_NEGATIVE)
        
        if len(self.social_memories) > 10:
            ratio = positive_interactions / (positive_interactions + negative_interactions + 1)
            
            if ratio > 0.7:
                self.attachment_style = 'secure'
                self.social_comfort = min(1.0, self.social_comfort + 0.01 * dt)
            elif ratio < 0.3:
                self.attachment_style = 'avoidant'
                self.social_comfort = max(0.1, self.social_comfort - 0.01 * dt)
            else:
                self.attachment_style = 'anxious'
                self.social_comfort = 0.5
        
        # Update empathy based on oxytocin level
        if self.current_level > 50:
            self.empathy_level += 0.001 * dt
        else:
            self.empathy_level -= 0.0005 * dt
        self.empathy_level = np.clip(self.empathy_level, 0.1, 1.0)
        
        # Update helpfulness based on positive interactions
        recent_helpful = [m for m in self.social_memories 
                         if m['time'] > current_time - 600 and 
                         m.get('was_helpful', False)]
        if recent_helpful:
            self.helpfulness += 0.01 * len(recent_helpful) * dt
            self.helpfulness = min(1.0, self.helpfulness)
    
    def get_user_bond_strength(self, user_id: str) -> float:
        """Get bond strength with specific user"""
        return self.user_bonds.get(user_id, 0.0)
    
    def get_social_modifiers(self) -> Dict[str, float]:
        """
        Get social behavior modifiers
        
        Returns:
            Dictionary of social behavior modifiers
        """
        return {
            'empathy': self.empathy_level,
            'helpfulness': self.helpfulness,
            'trust': self.global_trust,
            'generosity': self.generosity,
            'social_comfort': self.social_comfort,
            'stress_buffer': self.get_stress_buffer(),
            'warmth': min(1.0, self.current_level / 60)  # Response warmth
        }
    
    def should_offer_help(self, user_id: Optional[str] = None) -> bool:
        """Determine if should proactively offer help"""
        base_helpfulness = self.helpfulness > 0.6
        
        if user_id and user_id in self.user_bonds:
            # More likely to help bonded users
            bond = self.user_bonds[user_id]
            return base_helpfulness or bond > 0.5
        
        return base_helpfulness and self.global_trust > 0.5
    
    def get_state_summary(self) -> Dict:
        """Extended state summary with oxytocin-specific info"""
        summary = super().get_state_summary()
        social_mods = self.get_social_modifiers()
        summary.update({
            'bonded_users': len(self.user_bonds),
            'strongest_bond': max(self.user_bonds.values()) if self.user_bonds else 0,
            'global_trust': round(self.global_trust, 3),
            'empathy_level': round(self.empathy_level, 3),
            'attachment_style': self.attachment_style,
            'social_comfort': round(self.social_comfort, 3),
            'helpfulness': round(self.helpfulness, 3),
            'should_offer_help': self.should_offer_help(),
            'stress_buffer': round(social_mods['stress_buffer'], 3),
            'total_interactions': len(self.social_memories)
        })
        return summary
