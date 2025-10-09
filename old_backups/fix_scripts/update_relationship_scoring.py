with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'r') as f:
    content = f.read()

# Update the scoring values to make progression slower
old_scores = """    # Default small increase for normal interaction
    if score_change == 0:
        score_change = 2
        reason = "normal interaction" """

new_scores = """    # Default small increase for normal interaction
    if score_change == 0:
        score_change = 0.5  # Much slower progression
        reason = "normal interaction" """

content = content.replace(old_scores, new_scores)

# Also update positive interactions to be more modest
content = content.replace('score_change = 3', 'score_change = 1')  # Compliments
content = content.replace('score_change = 4', 'score_change = 1.5')  # Emotional
content = content.replace('score_change = -8', 'score_change = -5')  # Insults less harsh
content = content.replace('score_change = -12', 'score_change = -8')  # Aggression

# Update relationship stages to remove romantic levels
old_stages = """                    WHEN relationships.relationship_score + $3 <= 10 THEN 'stranger'
                    WHEN relationships.relationship_score + $3 <= 25 THEN 'acquaintance'
                    WHEN relationships.relationship_score + $3 <= 45 THEN 'friend'
                    WHEN relationships.relationship_score + $3 <= 65 THEN 'close_friend'
                    WHEN relationships.relationship_score + $3 <= 85 THEN 'romantic_interest'
                    ELSE 'partner'"""

new_stages = """                    WHEN relationships.relationship_score + $3 <= 15 THEN 'stranger'
                    WHEN relationships.relationship_score + $3 <= 30 THEN 'acquaintance'
                    WHEN relationships.relationship_score + $3 <= 50 THEN 'friend'
                    ELSE 'best_friend'"""

content = content.replace(old_stages, new_stages)

with open('/root/openhermes_backend/app/api/v1/routers/chat_router.py', 'w') as f:
    f.write(content)

print("✅ Updated relationship scoring:")
print("- Normal interaction: 0.5 points (was 2)")
print("- Compliments: 1 point (was 3)")
print("- Max level: Best Friend (removed romantic stages)")
print("- Progression is now 4x slower")
