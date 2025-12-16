"""
Continuous Trainer - Verbose training with full psychological analysis
"""
import json
import time
from datetime import datetime
 
from app.neurochemistry.hormone_network import hormone_network
from app.neurochemistry.data_generator import data_generator, SCENARIO_CATEGORIES
 
LOG_PATH = "/root/openhermes_backend/training_log.jsonl"
HNAMES = ['dopamine', 'serotonin', 'cortisol', 'adrenaline', 'oxytocin', 'norepinephrine', 'endorphins']
 
 
def bar(val, w=20):
    f = int(val * w)
    return '█' * f + '░' * (w - f)
 
 
def run_training(max_examples=None, verbose=True):
    print("=" * 70)
    print("DEEP PSYCHOLOGICAL TRAINING - HORMONE NETWORK")
    print("=" * 70)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Started: {now}")
    print(f"Scenarios: {len(SCENARIO_CATEGORIES)}")
    print("=" * 70)
    
    start = time.time()
    total_loss = 0
    count = 0
    i = 0
    
    try:
        while max_examples is None or i < max_examples:
            category = SCENARIO_CATEGORIES[i % len(SCENARIO_CATEGORIES)]
            
            print(f"\n{'─' * 70}")
            print(f"[{i+1}] SCENARIO: {category}")
            print('─' * 70)
            
            try:
                ex = data_generator.generate_example(category)
                
                print(f"\n📨 MESSAGE:")
                msg_short = ex.message[:100] + '...' if len(ex.message) > 100 else ex.message
                print(f"   {msg_short}")
                
                print(f"\n🧠 PFC PERCEPTION:")
                print(f"   User Emotion:    {ex.user_emotion}")
                print(f"   Intended Effect: {ex.intended_effect}")
                print(f"   True Intent:     {ex.intent}")
                print(f"   Manipulation:    {ex.manipulation}")
                
                print(f"\n🎯 TARGET HORMONES (PFC decided):")
                for j, name in enumerate(HNAMES):
                    val = ex.hormones[j]
                    print(f"   {name:15} [{bar(val)}] {val:.2f}")
                
                reason_short = ex.reasoning[:80] + '...' if len(ex.reasoning) > 80 else ex.reasoning
                print(f"\n💭 REASONING: {reason_short}")
                
                pred = hormone_network.predict(ex.perception)
                loss = hormone_network.train_step(ex.perception, ex.hormones)
                total_loss += loss
                count += 1
                
                print(f"\n📊 NETWORK LEARNING:")
                avg = total_loss / count
                print(f"   Loss: {loss:.4f} | Avg: {avg:.4f}")
                
                diffs = []
                for j, name in enumerate(HNAMES):
                    diff = abs(pred[j] - ex.hormones[j])
                    if diff > 0.2:
                        diffs.append(f"{name}: {pred[j]:.2f}->{ex.hormones[j]:.2f}")
                
                if diffs:
                    print(f"   ⚠️  Big corrections: {', '.join(diffs)}")
                
                log_entry = {
                    'time': datetime.now().isoformat(),
                    'category': category,
                    'message': ex.message,
                    'perception': ex.perception,
                    'user_emotion': ex.user_emotion,
                    'intent': ex.intent,
                    'manipulation': ex.manipulation,
                    'target': ex.hormones,
                    'predicted': pred.tolist(),
                    'reasoning': ex.reasoning,
                    'loss': loss
                }
                with open(LOG_PATH, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                if (i + 1) % 25 == 0:
                    hormone_network.save_weights()
                    elapsed = time.time() - start
                    print(f"\n{'=' * 70}")
                    avg2 = total_loss / count
                    print(f"CHECKPOINT: {count} examples | {elapsed/60:.1f} min | Avg Loss: {avg2:.4f}")
                    print(f"{'=' * 70}")
                    
            except Exception as e:
                print(f"   Error: {e}")
            
            i += 1
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    hormone_network.save_weights()
    elapsed = time.time() - start
    
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"Examples: {count}")
    print(f"Time: {elapsed/60:.1f} minutes")
    if count > 0:
        print(f"Avg Loss: {total_loss/count:.4f}")
    print(f"Log: {LOG_PATH}")
    print(f"{'=' * 70}")
 
 
if __name__ == "__main__":
    run_training()
