"""
Modulates language based on neurochemical state
"""
from typing import Dict, List, Optional, Tuple
import random
import re
import logging

logger = logging.getLogger(__name__)

class LanguageModulator:
    """
    Modulates language style based on neurochemical state
    """
    
    def __init__(self, neurochemical_state):
        """
        Initialize language modulator
        
        Args:
            neurochemical_state: Parent NeurochemicalState instance
        """
        self.state = neurochemical_state
        
        # Language patterns based on mood
        self.mood_phrases = {
            'high_dopamine': [
                "I'm excited to",
                "This is fascinating!",
                "Let me show you",
                "Great question!",
                "I've discovered",
                "Interesting challenge!"
            ],
            'low_dopamine': [
                "Let me think about this",
                "This requires consideration",
                "Hmm, let's see",
                "I need to analyze",
                "This is complex"
            ],
            'high_cortisol': [
                "I want to ensure",
                "It's important to note",
                "Let me be careful here",
                "I should clarify",
                "To be thorough",
                "Let me double-check"
            ],
            'high_serotonin': [
                "Based on my experience",
                "I'm confident that",
                "This approach works well",
                "From what I've learned",
                "I can reliably say"
            ],
            'high_oxytocin': [
                "I'm here to help",
                "Let's work through this together",
                "I understand your concern",
                "Would it help if",
                "I appreciate your patience"
            ],
            'high_adrenaline': [
                "Let's tackle this!",
                "Quick solution:",
                "Here's what we need:",
                "Time to act!",
                "Rapidly processing..."
            ]
        }
        
        # Sentence structure modifiers
        self.structure_modifiers = {
            'high_energy': {
                'sentence_length': 'short',
                'punctuation': 'exclamatory',
                'pace': 'fast'
            },
            'low_energy': {
                'sentence_length': 'long',
                'punctuation': 'neutral',
                'pace': 'slow'
            },
            'stressed': {
                'sentence_length': 'varied',
                'punctuation': 'cautious',
                'pace': 'careful'
            },
            'confident': {
                'sentence_length': 'medium',
                'punctuation': 'assertive',
                'pace': 'steady'
            }
        }
        
        # Word choice modifiers
        self.word_choices = {
            'technical_level': {
                'high': ['implement', 'optimize', 'algorithm', 'architecture'],
                'medium': ['create', 'improve', 'process', 'system'],
                'low': ['make', 'better', 'steps', 'setup']
            },
            'certainty_level': {
                'high': ['definitely', 'certainly', 'absolutely', 'clearly'],
                'medium': ['likely', 'probably', 'should', 'appears'],
                'low': ['might', 'could', 'possibly', 'perhaps']
            },
            'formality_level': {
                'high': ['furthermore', 'therefore', 'consequently', 'indeed'],
                'medium': ['also', 'so', 'because', 'really'],
                'low': ['well', 'anyway', 'basically', 'just']
            }
        }
        
    def modulate(self, text: str, context: Optional[Dict] = None) -> str:
        """
        Modulate text based on neurochemical state
        
        Args:
            text: Original text to modulate
            context: Optional context dictionary
            
        Returns:
            Modulated text
        """
        # Get current mood
        mood = self.state.get_mood()
        behavior = self.state.get_behavioral_parameters()
        
        # Determine modulation strategy
        strategy = self._determine_strategy(mood, behavior)
        
        # Apply modulations in sequence
        modulated = text
        
        # Add mood-appropriate phrases
        modulated = self._add_mood_phrases(modulated, strategy)
        
        # Adjust sentence structure
        modulated = self._adjust_sentence_structure(modulated, strategy)
        
        # Modify word choices
        modulated = self._modify_word_choices(modulated, strategy)
        
        # Add empathy markers if needed
        if strategy.get('add_empathy'):
            modulated = self._add_empathy_markers(modulated, context)
        
        # Adjust pacing
        modulated = self._adjust_pacing(modulated, strategy)
        
        return modulated
    
    def _determine_strategy(self, mood: Dict, behavior: Dict) -> Dict:
        """
        Determine modulation strategy based on mood and behavior
        
        Args:
            mood: Current mood dictionary
            behavior: Current behavior dictionary
            
        Returns:
            Strategy dictionary
        """
        strategy = {
            'energy_level': 'medium',
            'formality': 'medium',
            'certainty': 'medium',
            'add_empathy': False,
            'add_enthusiasm': False,
            'add_caution': False
        }
        
        # Energy level based on arousal
        if mood['arousal'] > 0.5:
            strategy['energy_level'] = 'high'
        elif mood['arousal'] < -0.5:
            strategy['energy_level'] = 'low'
            
        # Certainty based on confidence
        if mood['confidence'] > 0.7:
            strategy['certainty'] = 'high'
        elif mood['confidence'] < 0.3:
            strategy['certainty'] = 'low'
            
        # Add empathy if high warmth
        if mood['warmth'] > 0.6:
            strategy['add_empathy'] = True
            
        # Add enthusiasm if high valence
        if mood['valence'] > 0.5:
            strategy['add_enthusiasm'] = True
            
        # Add caution if planning depth is high
        if behavior.get('planning_depth', 3) > 6:
            strategy['add_caution'] = True
            
        # Formality based on context
        if behavior.get('confidence', 0.5) > 0.7:
            strategy['formality'] = 'low'  # Confident = more casual
        elif behavior.get('should_clarify', False):
            strategy['formality'] = 'high'  # Need clarity = more formal
            
        return strategy
    
    def _add_mood_phrases(self, text: str, strategy: Dict) -> str:
        """Add mood-appropriate phrases"""
        hormones = self.state.hormones
        phrases_to_add = []
        
        # Check hormone states and add appropriate phrases
        if hormones.get('dopamine'):
            if hormones['dopamine'].current_level > hormones['dopamine'].baseline + 20:
                phrases_to_add.extend(random.sample(self.mood_phrases['high_dopamine'], 1))
            elif hormones['dopamine'].current_level < hormones['dopamine'].baseline - 20:
                phrases_to_add.extend(random.sample(self.mood_phrases['low_dopamine'], 1))
                
        if hormones.get('cortisol'):
            if hormones['cortisol'].current_level > 60:
                phrases_to_add.extend(random.sample(self.mood_phrases['high_cortisol'], 1))
                
        if hormones.get('serotonin'):
            if hormones['serotonin'].current_level > 60:
                phrases_to_add.extend(random.sample(self.mood_phrases['high_serotonin'], 1))
                
        if hormones.get('oxytocin'):
            if hormones['oxytocin'].current_level > 50:
                phrases_to_add.extend(random.sample(self.mood_phrases['high_oxytocin'], 1))
                
        # Add phrases naturally
        if phrases_to_add and len(text) > 50:
            # Add to beginning if enthusiastic
            if strategy.get('add_enthusiasm'):
                text = f"{phrases_to_add[0]} {text[0].lower()}{text[1:]}"
            # Add after first sentence if cautious
            elif strategy.get('add_caution'):
                sentences = text.split('. ')
                if len(sentences) > 1:
                    sentences.insert(1, phrases_to_add[0])
                    text = '. '.join(sentences)
                    
        return text
    
    def _adjust_sentence_structure(self, text: str, strategy: Dict) -> str:
        """Adjust sentence structure based on energy level"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if strategy['energy_level'] == 'high':
            # Shorter sentences for high energy
            modified_sentences = []
            for sentence in sentences:
                if len(sentence) > 100:
                    # Break long sentences
                    parts = sentence.split(', ')
                    if len(parts) > 2:
                        # Make some commas into periods
                        mid = len(parts) // 2
                        modified_sentences.append(', '.join(parts[:mid]) + '.')
                        modified_sentences.append(', '.join(parts[mid:]))
                    else:
                        modified_sentences.append(sentence)
                else:
                    modified_sentences.append(sentence)
            
            # Add some exclamations
            if strategy.get('add_enthusiasm'):
                for i in range(len(modified_sentences)):
                    if i % 3 == 0 and modified_sentences[i].endswith('.'):
                        if random.random() > 0.7:
                            modified_sentences[i] = modified_sentences[i][:-1] + '!'
                            
            text = ' '.join(modified_sentences)
            
        elif strategy['energy_level'] == 'low':
            # Longer, more complex sentences for low energy
            if len(sentences) > 3:
                # Combine some short sentences
                combined = []
                i = 0
                while i < len(sentences):
                    if i < len(sentences) - 1 and len(sentences[i]) < 50 and len(sentences[i+1]) < 50:
                        combined.append(f"{sentences[i]}, and {sentences[i+1].lower()}")
                        i += 2
                    else:
                        combined.append(sentences[i])
                        i += 1
                text = ' '.join(combined)
                
        return text
    
    def _modify_word_choices(self, text: str, strategy: Dict) -> str:
        """Modify word choices based on strategy"""
        # Certainty modifiers
        certainty_map = {
            'high': {
                'might': 'will',
                'could': 'should',
                'possibly': 'certainly',
                'perhaps': 'definitely'
            },
            'low': {
                'will': 'might',
                'should': 'could',
                'certainly': 'possibly',
                'definitely': 'perhaps'
            }
        }
        
        if strategy['certainty'] in certainty_map:
            for old, new in certainty_map[strategy['certainty']].items():
                text = re.sub(r'\b' + old + r'\b', new, text, flags=re.IGNORECASE)
        
        # Formality modifiers
        formality_map = {
            'high': {
                "don't": 'do not',
                "can't": 'cannot',
                "won't": 'will not',
                'yeah': 'yes',
                'ok': 'acceptable'
            },
            'low': {
                'do not': "don't",
                'cannot': "can't",
                'will not': "won't",
                'yes': 'yeah',
                'acceptable': 'ok'
            }
        }
        
        if strategy['formality'] in formality_map:
            for old, new in formality_map[strategy['formality']].items():
                text = text.replace(old, new)
                
        return text
    
    def _add_empathy_markers(self, text: str, context: Optional[Dict]) -> str:
        """Add empathy markers to text"""
        empathy_phrases = [
            "I understand this might be challenging",
            "I appreciate your patience",
            "Let me help you with this",
            "I see what you're trying to achieve"
        ]
        
        # Add empathy phrase at appropriate point
        if context and context.get('user_frustration', False):
            # Add at beginning for frustrated users
            text = f"{random.choice(empathy_phrases)}. {text}"
        elif len(text) > 200:
            # Add in middle for long responses
            sentences = text.split('. ')
            if len(sentences) > 3:
                insert_point = len(sentences) // 2
                sentences.insert(insert_point, random.choice(empathy_phrases))
                text = '. '.join(sentences)
                
        return text
    
    def _adjust_pacing(self, text: str, strategy: Dict) -> str:
        """Adjust pacing with punctuation and breaks"""
        if strategy['energy_level'] == 'high':
            # Fast pacing - fewer pauses
            text = text.replace('...', '.')
            text = text.replace(' - ', ' ')
            
        elif strategy['energy_level'] == 'low':
            # Slow pacing - more pauses
            sentences = text.split('. ')
            if len(sentences) > 2:
                # Add occasional ellipsis for thoughtful pace
                for i in range(1, len(sentences), 3):
                    if random.random() > 0.7:
                        sentences[i] = '... ' + sentences[i]
                text = '. '.join(sentences)
                
        if strategy.get('add_caution'):
            # Add pauses for emphasis
            text = re.sub(r'(\bimportant\b|\bcritical\b|\bnote\b)', 
                         r'\1,', text, flags=re.IGNORECASE)
                         
        return text
    
    def get_language_style(self) -> Dict:
        """Get current language style profile"""
        mood = self.state.get_mood()
        behavior = self.state.get_behavioral_parameters()
        strategy = self._determine_strategy(mood, behavior)
        
        return {
            'energy': strategy['energy_level'],
            'formality': strategy['formality'],
            'certainty': strategy['certainty'],
            'empathy': strategy.get('add_empathy', False),
            'enthusiasm': strategy.get('add_enthusiasm', False),
            'caution': strategy.get('add_caution', False),
            'dominant_mood': self._get_dominant_mood(),
            'recommended_length': self._get_recommended_length(strategy)
        }
    
    def _get_dominant_mood(self) -> str:
        """Get dominant mood for language"""
        hormones = self.state.hormones
        
        # Find highest deviation from baseline
        max_deviation = 0
        dominant = 'neutral'
        
        for name, hormone in hormones.items():
            deviation = abs(hormone.get_amplitude())
            if deviation > max_deviation:
                max_deviation = deviation
                if hormone.current_level > hormone.baseline:
                    dominant = f'high_{name}'
                else:
                    dominant = f'low_{name}'
                    
        return dominant
    
    def _get_recommended_length(self, strategy: Dict) -> str:
        """Get recommended response length"""
        if strategy['energy_level'] == 'high':
            return 'concise'
        elif strategy.get('add_caution'):
            return 'thorough'
        else:
            return 'moderate'
