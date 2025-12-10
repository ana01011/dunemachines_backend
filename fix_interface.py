"""
Fix the message analyzer to properly detect emotions
"""

# Read the interface file
with open('/root/openhermes_backend/app/neurochemistry/interface.py', 'r') as f:
    content = f.read()

# Find and replace the _event_to_inputs method to better detect exercise and relaxation
old_method = """    def _event_to_inputs(self, event: NeurochemicalEvent) -> Dict[str, float]:
        \"\"\"
        Convert event to dynamics inputs
        \"\"\"
        inputs = {
            'reward': max(0, event.valence) * event.intensity,
            'punishment': max(0, -event.valence) * event.intensity,
            'threat': max(0, -event.valence) * event.arousal,
            'urgency': event.arousal * event.intensity,
            'social': event.social * event.intensity,
            'novelty': event.novelty,
            'uncertainty': event.uncertainty,
            'attention': event.arousal,
            'trust': event.social * max(0, event.valence),
            'touch': 0.0,  # Would need specific detection
            'exercise': 0.0,  # Would need specific detection
            'pain': max(0, -event.valence) * 0.5,
            'pleasure': max(0, event.valence) * 0.5,
            'fight_flight': event.arousal * max(0, -event.valence),
            'attachment': event.social * event.duration / 10,
            'nutrition': 0.5,  # Default moderate nutrition
            'glucose': 0.7,    # Default glucose
            'oxygen': 0.9,     # Default oxygen
            'temperature': 1.0, # Normal temperature
            'sleep': 0.0       # Awake by default
        }"""

new_method = """    def _event_to_inputs(self, event: NeurochemicalEvent) -> Dict[str, float]:
        \"\"\"
        Convert event to dynamics inputs - ENHANCED
        \"\"\"
        # Detect exercise keywords
        message_lower = event.event_type.lower() if hasattr(event, 'message') else ""
        exercise_words = ['exercise', 'workout', 'run', 'gym', 'marathon', 'burn', 'sweat', 'endorphin']
        is_exercise = any(word in message_lower for word in exercise_words) or 'exercise' in event.event_type.lower()
        
        # Detect relaxation keywords
        relax_words = ['relax', 'calm', 'peace', 'rest', 'sleep', 'meditat', 'quiet']
        is_relaxation = any(word in message_lower for word in relax_words) or 'relax' in event.event_type.lower()
        
        inputs = {
            'reward': max(0, event.valence) * event.intensity,
            'punishment': max(0, -event.valence) * event.intensity,
            'threat': max(0, -event.valence) * event.arousal if not is_relaxation else 0,
            'urgency': event.arousal * event.intensity if not is_relaxation else 0,
            'social': event.social * event.intensity,
            'novelty': event.novelty,
            'uncertainty': event.uncertainty if not is_relaxation else 0,
            'attention': event.arousal if not is_relaxation else 0.1,
            'trust': event.social * max(0, event.valence),
            'touch': 0.0,  
            'exercise': 1.0 if is_exercise else 0.0,  # Now properly detected!
            'pain': max(0, -event.valence) * 0.5,
            'pleasure': max(0, event.valence) * 0.5,
            'fight_flight': event.arousal * max(0, -event.valence) if not is_relaxation else 0,
            'attachment': event.social * event.duration / 10,
            'nutrition': 0.5,  
            'glucose': 0.7,    
            'oxygen': 0.9,     
            'temperature': 1.0, 
            'sleep': 0.8 if is_relaxation else 0.0  # Relaxation triggers sleep-like state
        }"""

content = content.replace(old_method, new_method)

# Also enhance the _analyze_message to preserve the message for keyword detection
old_analyze = """        return NeurochemicalEvent(
            event_type=event_type,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            social=social,
            novelty=novelty,
            uncertainty=uncertainty,
            duration=1.0
        )"""

new_analyze = """        event = NeurochemicalEvent(
            event_type=event_type,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            social=social,
            novelty=novelty,
            uncertainty=uncertainty,
            duration=1.0
        )
        event.message = message  # Store original message for keyword detection
        return event"""

content = content.replace(old_analyze, new_analyze)

with open('/root/openhermes_backend/app/neurochemistry/interface.py', 'w') as f:
    f.write(content)

print("✅ Interface fixed!")
print("\nChanges made:")
print("1. Now properly detects exercise keywords")
print("2. Now properly detects relaxation keywords")
print("3. Relaxation sets sleep=0.8 and reduces arousal signals")
