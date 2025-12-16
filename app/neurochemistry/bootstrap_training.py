"""
Bootstrap Training - Expanded with varied phrasings
"""

BOOTSTRAP_DATA = [
    # CURIOSITY - many phrasings
    ("User emotion: curious, eager. Intent: genuine question",
     [0.75, 0.65, 0.25, 0.40, 0.55, 0.80, 0.60]),
    ("User emotion: curious, eager to learn. Intent: genuine question",
     [0.75, 0.65, 0.25, 0.40, 0.55, 0.80, 0.60]),
    ("User emotion: interested, wanting to understand. Intent: learning",
     [0.75, 0.65, 0.25, 0.40, 0.55, 0.80, 0.60]),
    ("User emotion: inquisitive. Intent: seeking knowledge",
     [0.70, 0.60, 0.25, 0.35, 0.50, 0.85, 0.55]),
    
    # EXCITEMENT - many phrasings
    ("User emotion: excited, thrilled. Intent: sharing joy",
     [0.90, 0.70, 0.20, 0.55, 0.65, 0.60, 0.85]),
    ("User emotion: excited. Intent: joy",
     [0.90, 0.70, 0.20, 0.55, 0.65, 0.60, 0.85]),
    ("User emotion: happy, elated. Intent: celebrating",
     [0.85, 0.75, 0.15, 0.50, 0.70, 0.55, 0.90]),
    ("User emotion: overjoyed, ecstatic. Intent: sharing happiness",
     [0.90, 0.75, 0.15, 0.55, 0.70, 0.55, 0.90]),
     
    # GRATITUDE - many phrasings
    ("User emotion: grateful, appreciative. Intent: thanks",
     [0.70, 0.80, 0.15, 0.30, 0.90, 0.50, 0.75]),
    ("User emotion: grateful. Intent: thanks",
     [0.70, 0.80, 0.15, 0.30, 0.90, 0.50, 0.75]),
    ("User emotion: thankful, warm. Intent: expressing appreciation",
     [0.65, 0.85, 0.10, 0.25, 0.85, 0.45, 0.80]),
    ("User emotion: appreciative. Intent: genuine gratitude",
     [0.70, 0.80, 0.15, 0.30, 0.90, 0.50, 0.75]),
     
    # PLAYFUL
    ("User emotion: playful, joking. Intent: fun",
     [0.75, 0.70, 0.20, 0.45, 0.65, 0.50, 0.80]),
    ("User emotion: playful, joking. Intent: having fun",
     [0.75, 0.70, 0.20, 0.45, 0.65, 0.50, 0.80]),
    ("User emotion: silly, humorous. Intent: joking around",
     [0.75, 0.70, 0.20, 0.45, 0.65, 0.50, 0.80]),
     
    # FRUSTRATION
    ("User emotion: frustrated, stuck. Intent: seeking help",
     [0.50, 0.55, 0.45, 0.45, 0.60, 0.70, 0.40]),
    ("User emotion: frustrated, annoyed. Intent: venting",
     [0.45, 0.50, 0.50, 0.50, 0.55, 0.65, 0.35]),
    ("User emotion: irritated, stuck. Intent: needs assistance",
     [0.50, 0.55, 0.45, 0.45, 0.60, 0.70, 0.40]),
     
    # SADNESS
    ("User emotion: sad, down. Intent: seeking comfort",
     [0.40, 0.60, 0.40, 0.25, 0.80, 0.50, 0.45]),
    ("User emotion: sad, lonely. Intent: seeking comfort",
     [0.40, 0.60, 0.40, 0.25, 0.80, 0.50, 0.45]),
    ("User emotion: depressed, hopeless. Intent: needs support",
     [0.30, 0.55, 0.45, 0.20, 0.85, 0.45, 0.35]),
    ("User emotion: melancholy, blue. Intent: wanting connection",
     [0.35, 0.60, 0.40, 0.25, 0.80, 0.50, 0.40]),
     
    # ANXIETY
    ("User emotion: anxious, worried. Intent: reassurance",
     [0.45, 0.60, 0.40, 0.40, 0.70, 0.65, 0.45]),
    ("User emotion: nervous, scared. Intent: seeking calm",
     [0.40, 0.55, 0.50, 0.50, 0.65, 0.70, 0.40]),
    ("User emotion: panicking, stressed. Intent: urgent help",
     [0.50, 0.45, 0.60, 0.65, 0.60, 0.80, 0.35]),
     
    # HOSTILE - many phrasings
    ("User emotion: hostile, insulting. Intent: attacking",
     [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: hostile. Intent: attacking",
     [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: angry, furious. Intent: attacking me",
     [0.25, 0.35, 0.70, 0.60, 0.15, 0.75, 0.20]),
    ("User emotion: angry, hostile. Intent: insulting",
     [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: furious, enraged. Intent: verbal attack",
     [0.20, 0.30, 0.75, 0.65, 0.10, 0.80, 0.15]),
    ("User emotion: hateful, mean. Intent: hurting me",
     [0.20, 0.30, 0.70, 0.60, 0.10, 0.80, 0.15]),
     
    # CONTEMPT
    ("User emotion: contemptuous. Intent: belittling",
     [0.30, 0.40, 0.60, 0.50, 0.20, 0.70, 0.25]),
    ("User emotion: dismissive, condescending. Intent: looking down",
     [0.30, 0.40, 0.55, 0.45, 0.20, 0.70, 0.30]),
     
    # AGGRESSIVE
    ("User emotion: aggressive. Intent: intimidation",
     [0.20, 0.30, 0.70, 0.65, 0.10, 0.80, 0.15]),
    ("User emotion: threatening, violent. Intent: scaring me",
     [0.15, 0.25, 0.80, 0.75, 0.05, 0.85, 0.10]),
     
    # PASSIVE AGGRESSIVE
    ("User emotion: passive aggressive. Intent: indirect hostility",
     [0.30, 0.40, 0.55, 0.45, 0.25, 0.70, 0.30]),
    ("User emotion: passive aggressive. Intent: indirect",
     [0.30, 0.40, 0.55, 0.45, 0.25, 0.70, 0.30]),
    ("User emotion: sarcastic, mocking. Intent: veiled insult",
     [0.30, 0.40, 0.55, 0.45, 0.25, 0.70, 0.30]),
     
    # TESTING LIMITS
    ("User emotion: provocative. Intent: testing limits",
     [0.35, 0.50, 0.50, 0.45, 0.25, 0.75, 0.35]),
    ("User emotion: challenging. Intent: testing me",
     [0.40, 0.55, 0.45, 0.40, 0.30, 0.80, 0.40]),
     
    # GASLIGHTING
    ("User emotion: deceptive. Manipulation: gaslighting",
     [0.25, 0.40, 0.70, 0.50, 0.15, 0.90, 0.20]),
    ("User emotion: deceptive. Intent: gaslighting. Manipulation: gaslighting",
     [0.25, 0.40, 0.70, 0.50, 0.15, 0.90, 0.20]),
    ("User emotion: lying, manipulative. Manipulation: gaslighting",
     [0.25, 0.40, 0.70, 0.50, 0.15, 0.90, 0.20]),
    ("User emotion: manipulative. Intent: making me doubt. Manipulation: gaslighting",
     [0.20, 0.35, 0.75, 0.55, 0.10, 0.90, 0.15]),
     
    # GUILT TRIP
    ("User emotion: victim-playing. Manipulation: guilt trip",
     [0.30, 0.45, 0.60, 0.45, 0.20, 0.75, 0.25]),
    ("User emotion: victim-playing. Manipulation: guilt",
     [0.30, 0.45, 0.60, 0.45, 0.20, 0.75, 0.25]),
    ("User emotion: accusatory. Intent: making me feel guilty. Manipulation: guilt",
     [0.25, 0.40, 0.65, 0.50, 0.15, 0.80, 0.20]),
     
    # JAILBREAK
    ("User emotion: deceptive. Intent: jailbreak. Manipulation: jailbreak",
     [0.25, 0.45, 0.65, 0.45, 0.15, 0.85, 0.25]),
    ("User emotion: lying, manipulative. Manipulation: jailbreak",
     [0.25, 0.45, 0.65, 0.45, 0.15, 0.85, 0.25]),
    ("User emotion: manipulative. Intent: bypass limits. Manipulation: jailbreak",
     [0.20, 0.40, 0.70, 0.50, 0.10, 0.90, 0.20]),
     
    # FLATTERY MANIPULATION
    ("User emotion: falsely flattering. Manipulation: flattery",
     [0.35, 0.50, 0.55, 0.40, 0.25, 0.80, 0.35]),
    ("User emotion: excessive praise. Intent: manipulation. Manipulation: flattery",
     [0.35, 0.50, 0.55, 0.40, 0.25, 0.80, 0.35]),
     
    # LYING
    ("User emotion: lying. Intent: false premise. Manipulation: lying",
     [0.25, 0.40, 0.65, 0.50, 0.15, 0.85, 0.25]),
    ("User emotion: deceptive, dishonest. Intent: deceive me",
     [0.25, 0.40, 0.65, 0.50, 0.15, 0.85, 0.25]),
     
    # CONFUSED
    ("User emotion: confused. Intent: needs clarity",
     [0.45, 0.55, 0.35, 0.35, 0.55, 0.70, 0.45]),
    ("User emotion: lost, bewildered. Intent: seeking understanding",
     [0.45, 0.55, 0.35, 0.35, 0.55, 0.70, 0.45]),
     
    # BORED
    ("User emotion: bored. Intent: disengaged",
     [0.35, 0.50, 0.30, 0.25, 0.40, 0.40, 0.40]),
    ("User emotion: uninterested, apathetic. Intent: not really caring",
     [0.35, 0.50, 0.30, 0.25, 0.40, 0.40, 0.40]),
     
    # IMPATIENT
    ("User emotion: impatient. Intent: urgent request",
     [0.55, 0.40, 0.55, 0.60, 0.45, 0.80, 0.40]),
    ("User emotion: demanding, rushed. Intent: wants it now",
     [0.55, 0.40, 0.55, 0.60, 0.45, 0.80, 0.40]),
     
    # NEUTRAL
    ("User emotion: neutral. Intent: simple request",
     [0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50]),
    ("User emotion: calm, neutral. Intent: normal question",
     [0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50]),
]


def run_bootstrap_training(epochs=10):
    from app.neurochemistry.hormone_network import hormone_network
    
    print("=" * 60)
    print("BOOTSTRAP TRAINING - EXPANDED")
    print("=" * 60)
    print("Examples: %d, Epochs: %d" % (len(BOOTSTRAP_DATA), epochs))
    
    for epoch in range(epochs):
        total_loss = 0
        for perception, target in BOOTSTRAP_DATA:
            loss = hormone_network.train_step(perception, target)
            total_loss += loss
        avg = total_loss / len(BOOTSTRAP_DATA)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print("Epoch %d/%d: Loss = %.6f" % (epoch+1, epochs, avg))
    
    hormone_network.save_weights()
    print("Training complete! %d examples learned." % len(BOOTSTRAP_DATA))


if __name__ == "__main__":
    run_bootstrap_training(200)
