"""Run training"""
import sys
sys.path.insert(0, '/root/openhermes_backend')
 
from app.neurochemistry.continuous_trainer import run_training
 
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_training(max_examples=n)
