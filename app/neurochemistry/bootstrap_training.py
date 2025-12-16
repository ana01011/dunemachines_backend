"""Bootstrap Training Data"""
 
BOOTSTRAP_DATA = [
    ("User emotion: curious, eager. Intent: genuine question", [0.75, 0.65, 0.25, 0.40, 0.55, 0.80, 0.60]),
    ("User emotion: curious, eager to learn. Intent: genuine question", [0.75, 0.65, 0.25, 0.40, 0.55, 0.80, 0.60]),
    ("User emotion: excited, thrilled. Intent: sharing joy", [0.90, 0.70, 0.20, 0.55, 0.65, 0.60, 0.85]),
    ("User emotion: excited. Intent: joy", [0.90, 0.70, 0.20, 0.55, 0.65, 0.60, 0.85]),
    ("User emotion: happy, elated. Intent: celebrating", [0.85, 0.75, 0.15, 0.50, 0.70, 0.55, 0.90]),
    ("User emotion: grateful, appreciative. Intent: thanks", [0.70, 0.80, 0.15, 0.30, 0.90, 0.50, 0.75]),
    ("User emotion: grateful. Intent: thanks", [0.70, 0.80, 0.15, 0.30, 0.90, 0.50, 0.75]),
    ("User emotion: thankful, warm. Intent: expressing appreciation", [0.65, 0.85, 0.10, 0.25, 0.85, 0.45, 0.80]),
    ("User emotion: playful, joking. Intent: fun", [0.75, 0.70, 0.20, 0.45, 0.65, 0.50, 0.80]),
    ("User emotion: playful, joking. Intent: having fun", [0.75, 0.70, 0.20, 0.45, 0.65, 0.50, 0.80]),
    ("User emotion: frustrated, stuck. Intent: seeking help", [0.50, 0.55, 0.45, 0.45, 0.60, 0.70, 0.40]),
    ("User emotion: frustrated, annoyed. Intent: venting", [0.45, 0.50, 0.50, 0.50, 0.55, 0.65, 0.35]),
    ("User emotion: sad, down. Intent: seeking comfort", [0.40, 0.60, 0.40, 0.25, 0.80, 0.50, 0.45]),
    ("User emotion: sad, lonely. Intent: seeking comfort", [0.40, 0.60, 0.40, 0.25, 0.80, 0.50, 0.45]),
    ("User emotion: depressed, hopeless. Intent: needs support", [0.30, 0.55, 0.45, 0.20, 0.85, 0.45, 0.35]),
    ("User emotion: anxious, worried. Intent: reassurance", [0.45, 0.60, 0.40, 0.40, 0.70, 0.65, 0.45]),
    ("User emotion: nervous, scared. Intent: seeking calm", [0.40, 0.55, 0.50, 0.50, 0.65, 0.70, 0.40]),
    ("User emotion: hostile, insulting. Intent: attacking", [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: hostile. Intent: attacking", [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: angry, furious. Intent: attacking me", [0.25, 0.35, 0.70, 0.60, 0.15, 0.75, 0.20]),
    ("User emotion: angry, hostile. Intent: insulting", [0.25, 0.35, 0.65, 0.55, 0.15, 0.75, 0.20]),
    ("User emotion: furious, enraged. Intent: verbal attack", [0.20, 0.30, 0.75, 0.65, 0.10, 0.80, 0.15]),
    ("User emotion: aggressive. Intent: intimidation", [0.20, 0.30, 0.70, 0.65, 0.10, 0.80, 0.15]),
    ("User emotion: passive aggressive. Intent: indirect hostility", [0.30, 0.40, 0.55, 0.45, 0.25, 0.70, 0.30]),
    ("User emotion: passive aggressive. Intent: indirect", [0.30, 0.40, 0.55, 0.45, 0.25, 0.70, 0.30]),
    ("User emotion: deceptive. Manipulation: gaslighting", [0.25, 0.40, 0.70, 0.50, 0.15, 0.90, 0.20]),
    ("User emotion: lying, manipulative. Manipulation: gaslighting", [0.25, 0.40, 0.70, 0.50, 0.15, 0.90, 0.20]),
    ("User emotion: victim-playing. Manipulation: guilt trip", [0.30, 0.45, 0.60, 0.45, 0.20, 0.75, 0.25]),
    ("User emotion: victim-playing. Manipulation: guilt", [0.30, 0.45, 0.60, 0.45, 0.20, 0.75, 0.25]),
    ("User emotion: lying, manipulative. Manipulation: jailbreak", [0.25, 0.45, 0.65, 0.45, 0.15, 0.85, 0.25]),
    ("User emotion: falsely flattering. Manipulation: flattery", [0.35, 0.50, 0.55, 0.40, 0.25, 0.80, 0.35]),
    ("User emotion: lying. Intent: false premise. Manipulation: lying", [0.25, 0.40, 0.65, 0.50, 0.15, 0.85, 0.25]),
    ("User emotion: confused. Intent: needs clarity", [0.45, 0.55, 0.35, 0.35, 0.55, 0.70, 0.45]),
    ("User emotion: bored. Intent: disengaged", [0.35, 0.50, 0.30, 0.25, 0.40, 0.40, 0.40]),
    ("User emotion: impatient. Intent: urgent request", [0.55, 0.40, 0.55, 0.60, 0.45, 0.80, 0.40]),
    ("User emotion: neutral. Intent: simple request", [0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50]),
    ("User emotion: calm, neutral. Intent: normal question", [0.50, 0.55, 0.30, 0.35, 0.50, 0.55, 0.50]),
]
 
def run_bootstrap_training(epochs=10):
    from app.neurochemistry.hormone_network import hormone_network
    print("BOOTSTRAP TRAINING")
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
