"""
Generates mood indicators for UI display
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MoodIndicator:
    """
    Generates user-friendly mood indicators from neurochemical state
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize mood indicator
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        
        # Mood history for trends
        self.mood_history = []
        self.history_window = 20
        
        # Visual parameters
        self.color_mappings = {
            'very_positive': '#4CAF50',  # Green
            'positive': '#8BC34A',       # Light green
            'neutral': '#FFC107',        # Amber
            'negative': '#FF9800',       # Orange
            'very_negative': '#F44336'   # Red
        }
        
        self.energy_animations = {
            'very_high': 'pulse-fast',
            'high': 'pulse-medium',
            'medium': 'pulse-slow',
            'low': 'fade-slow',
            'very_low': 'fade-very-slow'
        }
        
    def get_current_mood(self) -> Dict:
        """
        Get current mood indicators
        
        Returns:
            Mood indicator dictionary
        """
        # Get raw mood from state
        raw_mood = self.state.get_mood()
        
        # Generate user-friendly indicators
        indicators = {
            'timestamp': datetime.now().isoformat(),
            
            # Numerical indicators (for charts)
            'metrics': {
                'arousal': raw_mood['arousal'],
                'valence': raw_mood['valence'],
                'focus': raw_mood['focus'],
                'confidence': raw_mood['confidence'],
                'warmth': raw_mood['warmth']
            },
            
            # Natural language description
            'description': self._generate_description(raw_mood),
            
            # Visual indicators
            'visual': {
                'color': self._get_mood_color(raw_mood['valence']),
                'animation': self._get_energy_animation(raw_mood['arousal']),
                'intensity': self._get_intensity(raw_mood),
                'icon': self._get_mood_icon(raw_mood)
            },
            
            # Behavioral hints
            'hints': self._generate_hints(raw_mood),
            
            # Trend information
            'trend': self._calculate_trend(),
            
            # Quick status
            'status': self._get_status_summary(raw_mood)
        }
        
        # Add to history
        self.mood_history.append({
            'timestamp': datetime.now(),
            'mood': raw_mood,
            'indicators': indicators
        })
        
        # Trim history
        if len(self.mood_history) > self.history_window:
            self.mood_history.pop(0)
        
        return indicators
    
    def _generate_description(self, mood: Dict) -> str:
        """
        Generate natural language mood description
        
        Args:
            mood: Raw mood dictionary
            
        Returns:
            Natural description string
        """
        descriptions = []
        
        # Energy description
        if mood['arousal'] > 0.5:
            descriptions.append("feeling energetic")
        elif mood['arousal'] > 0:
            descriptions.append("alert")
        elif mood['arousal'] > -0.5:
            descriptions.append("calm")
        else:
            descriptions.append("relaxed")
        
        # Emotional description
        if mood['valence'] > 0.5:
            descriptions.append("very positive")
        elif mood['valence'] > 0:
            descriptions.append("positive")
        elif mood['valence'] > -0.5:
            descriptions.append("neutral")
        else:
            descriptions.append("slightly stressed")
        
        # Focus description
        if mood['focus'] > 0.7:
            descriptions.append("highly focused")
        elif mood['focus'] > 0.4:
            descriptions.append("focused")
        elif mood['focus'] < 0.3:
            descriptions.append("somewhat scattered")
        
        # Confidence description
        if mood['confidence'] > 0.7:
            descriptions.append("very confident")
        elif mood['confidence'] > 0.4:
            descriptions.append("confident")
        elif mood['confidence'] < 0.3:
            descriptions.append("uncertain")
        
        # Combine descriptions
        if len(descriptions) == 1:
            return f"Currently {descriptions[0]}"
        elif len(descriptions) == 2:
            return f"Currently {descriptions[0]} and {descriptions[1]}"
        else:
            return f"Currently {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
    
    def _get_mood_color(self, valence: float) -> str:
        """
        Get color based on emotional valence
        
        Args:
            valence: Emotional valence (-1 to 1)
            
        Returns:
            Hex color code
        """
        if valence > 0.5:
            return self.color_mappings['very_positive']
        elif valence > 0.2:
            return self.color_mappings['positive']
        elif valence > -0.2:
            return self.color_mappings['neutral']
        elif valence > -0.5:
            return self.color_mappings['negative']
        else:
            return self.color_mappings['very_negative']
    
    def _get_energy_animation(self, arousal: float) -> str:
        """
        Get animation style based on arousal
        
        Args:
            arousal: Arousal level (-1 to 1)
            
        Returns:
            Animation class name
        """
        if arousal > 0.7:
            return self.energy_animations['very_high']
        elif arousal > 0.3:
            return self.energy_animations['high']
        elif arousal > -0.3:
            return self.energy_animations['medium']
        elif arousal > -0.7:
            return self.energy_animations['low']
        else:
            return self.energy_animations['very_low']
    
    def _get_intensity(self, mood: Dict) -> float:
        """
        Calculate overall mood intensity
        
        Args:
            mood: Raw mood dictionary
            
        Returns:
            Intensity value (0 to 1)
        """
        # Intensity based on deviation from neutral
        arousal_intensity = abs(mood['arousal'])
        valence_intensity = abs(mood['valence'])
        
        # Weight arousal and valence
        intensity = (arousal_intensity * 0.4 + valence_intensity * 0.6)
        
        return np.clip(intensity, 0, 1)
    
    def _get_mood_icon(self, mood: Dict) -> str:
        """
        Get appropriate icon for mood
        
        Args:
            mood: Raw mood dictionary
            
        Returns:
            Icon identifier
        """
        # Determine primary mood characteristic
        if mood['valence'] > 0.5 and mood['arousal'] > 0.5:
            return 'excited'
        elif mood['valence'] > 0.5 and mood['arousal'] < -0.5:
            return 'content'
        elif mood['valence'] < -0.5 and mood['arousal'] > 0.5:
            return 'stressed'
        elif mood['valence'] < -0.5 and mood['arousal'] < -0.5:
            return 'tired'
        elif mood['focus'] > 0.7:
            return 'focused'
        elif mood['warmth'] > 0.7:
            return 'friendly'
        else:
            return 'neutral'
    
    def _generate_hints(self, mood: Dict) -> List[str]:
        """
        Generate behavioral hints based on mood
        
        Args:
            mood: Raw mood dictionary
            
        Returns:
            List of hint strings
        """
        hints = []
        
        # Response style hints
        if mood['arousal'] > 0.5:
            hints.append("Quick, energetic responses expected")
        elif mood['arousal'] < -0.5:
            hints.append("Thoughtful, measured responses expected")
        
        # Interaction hints
        if mood['valence'] > 0.5:
            hints.append("Open to creative solutions")
        elif mood['valence'] < -0.5:
            hints.append("Being extra careful and thorough")
        
        # Focus hints
        if mood['focus'] > 0.7:
            hints.append("Deep analysis mode active")
        elif mood['focus'] < 0.3:
            hints.append("May benefit from clarification")
        
        # Confidence hints
        if mood['confidence'] > 0.7:
            hints.append("High certainty in responses")
        elif mood['confidence'] < 0.3:
            hints.append("May request additional information")
        
        # Social hints
        if mood['warmth'] > 0.6:
            hints.append("Enhanced empathy active")
        
        return hints
    
    def _calculate_trend(self) -> Dict:
        """
        Calculate mood trend from history
        
        Returns:
            Trend dictionary
        """
        if len(self.mood_history) < 3:
            return {
                'direction': 'stable',
                'strength': 0.0,
                'duration': 0
            }
        
        # Get recent moods
        recent = self.mood_history[-5:]
        
        # Calculate trends for each metric
        arousal_trend = self._calculate_metric_trend([m['mood']['arousal'] for m in recent])
        valence_trend = self._calculate_metric_trend([m['mood']['valence'] for m in recent])
        
        # Determine overall trend
        if valence_trend > 0.1:
            direction = 'improving'
        elif valence_trend < -0.1:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Calculate duration of current trend
        duration = self._calculate_trend_duration(direction)
        
        return {
            'direction': direction,
            'strength': abs(valence_trend),
            'duration': duration,
            'arousal_trend': arousal_trend,
            'valence_trend': valence_trend
        }
    
    def _calculate_metric_trend(self, values: List[float]) -> float:
        """Calculate trend for a single metric"""
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]  # Slope
    
    def _calculate_trend_duration(self, direction: str) -> int:
        """Calculate how long current trend has lasted"""
        if len(self.mood_history) < 2:
            return 0
            
        count = 0
        for i in range(len(self.mood_history) - 1, 0, -1):
            current = self.mood_history[i]['mood']['valence']
            previous = self.mood_history[i-1]['mood']['valence']
            
            if direction == 'improving' and current > previous:
                count += 1
            elif direction == 'declining' and current < previous:
                count += 1
            elif direction == 'stable' and abs(current - previous) < 0.1:
                count += 1
            else:
                break
                
        return count
    
    def _get_status_summary(self, mood: Dict) -> str:
        """
        Get quick status summary
        
        Args:
            mood: Raw mood dictionary
            
        Returns:
            Status string
        """
        # Determine primary state
        if mood['arousal'] > 0.5 and mood['valence'] > 0.5:
            return "Excited and engaged"
        elif mood['arousal'] > 0.5 and mood['valence'] < -0.5:
            return "Stressed but active"
        elif mood['arousal'] < -0.5 and mood['valence'] > 0.5:
            return "Content and relaxed"
        elif mood['arousal'] < -0.5 and mood['valence'] < -0.5:
            return "Low energy, needs support"
        elif mood['focus'] > 0.7:
            return "Deeply focused"
        elif mood['confidence'] > 0.7:
            return "Confident and ready"
        elif mood['warmth'] > 0.7:
            return "Socially engaged"
        else:
            return "Balanced and steady"
    
    def get_mood_summary(self) -> Dict:
        """Get comprehensive mood summary"""
        current = self.get_current_mood()
        
        # Calculate statistics if enough history
        stats = {}
        if len(self.mood_history) > 5:
            recent_moods = [h['mood'] for h in self.mood_history[-10:]]
            stats = {
                'avg_arousal': np.mean([m['arousal'] for m in recent_moods]),
                'avg_valence': np.mean([m['valence'] for m in recent_moods]),
                'avg_focus': np.mean([m['focus'] for m in recent_moods]),
                'volatility': np.std([m['valence'] for m in recent_moods]),
                'stability': 1.0 - np.std([m['arousal'] for m in recent_moods])
            }
        
        return {
            'current': current,
            'statistics': stats,
            'history_length': len(self.mood_history),
            'dominant_state': self._get_dominant_state(),
            'recommendations': self._get_mood_recommendations()
        }
    
    def _get_dominant_state(self) -> str:
        """Get dominant mood state from recent history"""
        if not self.mood_history:
            return 'unknown'
            
        recent = self.mood_history[-5:] if len(self.mood_history) >= 5 else self.mood_history
        
        # Count states
        state_counts = {
            'positive': 0,
            'negative': 0,
            'energetic': 0,
            'calm': 0,
            'focused': 0
        }
        
        for entry in recent:
            mood = entry['mood']
            if mood['valence'] > 0.2:
                state_counts['positive'] += 1
            elif mood['valence'] < -0.2:
                state_counts['negative'] += 1
            
            if mood['arousal'] > 0.3:
                state_counts['energetic'] += 1
            elif mood['arousal'] < -0.3:
                state_counts['calm'] += 1
                
            if mood['focus'] > 0.6:
                state_counts['focused'] += 1
        
        # Find dominant
        return max(state_counts.items(), key=lambda x: x[1])[0]
    
    def _get_mood_recommendations(self) -> List[str]:
        """Get recommendations based on mood patterns"""
        recommendations = []
        
        if not self.mood_history:
            return recommendations
            
        recent = self.mood_history[-5:] if len(self.mood_history) >= 5 else self.mood_history
        recent_moods = [h['mood'] for h in recent]
        
        # Check for concerning patterns
        avg_valence = np.mean([m['valence'] for m in recent_moods])
        avg_arousal = np.mean([m['arousal'] for m in recent_moods])
        
        if avg_valence < -0.3:
            recommendations.append("Consider easier tasks to build confidence")
            
        if avg_arousal > 0.6:
            recommendations.append("High energy detected - good for challenging tasks")
        elif avg_arousal < -0.6:
            recommendations.append("Low energy - consider taking a break")
            
        # Check volatility
        valence_std = np.std([m['valence'] for m in recent_moods])
        if valence_std > 0.4:
            recommendations.append("Mood fluctuating - focus on stability")
            
        return recommendations
