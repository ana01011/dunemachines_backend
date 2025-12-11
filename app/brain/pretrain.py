"""
Pretrain brain neural networks with example patterns
"""
import numpy as np
import json
import os

WEIGHTS_PATH = "/root/openhermes_backend/thalamus_weights.json"

TRAINING_DATA = [
    # CODE - high code signal
    ("write a function", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("implement algorithm", {"code": 0.9, "math": 0.3, "memory": 0.1}),
    ("python script", {"code": 0.95, "math": 0.1, "memory": 0.1}),
    ("code for", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("write python", {"code": 0.95, "math": 0.1, "memory": 0.1}),
    ("create class", {"code": 0.9, "math": 0.1, "memory": 0.1}),
    ("debug this", {"code": 0.85, "math": 0.1, "memory": 0.2}),
    ("fix the bug", {"code": 0.85, "math": 0.1, "memory": 0.2}),
    ("binary search", {"code": 0.9, "math": 0.3, "memory": 0.1}),
    ("quicksort", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("merge sort", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("implement search", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("implement sort", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("linked list", {"code": 0.9, "math": 0.2, "memory": 0.1}),
    ("api endpoint", {"code": 0.9, "math": 0.1, "memory": 0.2}),
    ("database query", {"code": 0.8, "math": 0.1, "memory": 0.4}),
    ("parse json", {"code": 0.85, "math": 0.1, "memory": 0.1}),
    ("web scraper", {"code": 0.9, "math": 0.1, "memory": 0.2}),
    
    # MATH - high math signal
    ("calculate derivative", {"code": 0.2, "math": 0.95, "memory": 0.1}),
    ("integral of", {"code": 0.2, "math": 0.95, "memory": 0.1}),
    ("solve equation", {"code": 0.2, "math": 0.9, "memory": 0.1}),
    ("what is the sum", {"code": 0.1, "math": 0.85, "memory": 0.1}),
    ("percentage of", {"code": 0.1, "math": 0.85, "memory": 0.1}),
    ("probability", {"code": 0.2, "math": 0.9, "memory": 0.1}),
    ("matrix multiplication", {"code": 0.4, "math": 0.9, "memory": 0.1}),
    ("eigenvalue", {"code": 0.3, "math": 0.95, "memory": 0.1}),
    ("quadratic formula", {"code": 0.2, "math": 0.9, "memory": 0.1}),
    
    # MEMORY - high memory signal
    ("remember when", {"code": 0.1, "math": 0.1, "memory": 0.9}),
    ("what did i say", {"code": 0.1, "math": 0.1, "memory": 0.95}),
    ("previous conversation", {"code": 0.1, "math": 0.1, "memory": 0.9}),
    ("last time", {"code": 0.1, "math": 0.1, "memory": 0.85}),
    ("you told me", {"code": 0.1, "math": 0.1, "memory": 0.9}),
    
    # GENERAL - low all signals
    ("explain how", {"code": 0.1, "math": 0.1, "memory": 0.15}),
    ("what is", {"code": 0.1, "math": 0.15, "memory": 0.15}),
    ("tell me about", {"code": 0.1, "math": 0.1, "memory": 0.15}),
    ("describe", {"code": 0.1, "math": 0.1, "memory": 0.1}),
    ("why does", {"code": 0.1, "math": 0.1, "memory": 0.15}),
    ("how does", {"code": 0.1, "math": 0.1, "memory": 0.15}),
    ("photosynthesis", {"code": 0.05, "math": 0.05, "memory": 0.1}),
    ("biology", {"code": 0.05, "math": 0.1, "memory": 0.15}),
    ("history of", {"code": 0.05, "math": 0.05, "memory": 0.3}),
    ("explain", {"code": 0.1, "math": 0.1, "memory": 0.15}),
]


def encode_text(text: str, size: int = 256) -> np.ndarray:
    encoding = np.zeros(size)
    text_bytes = text.lower().encode()[:200]
    for i, char in enumerate(text_bytes):
        encoding[i % size] += (char - 128) / 128.0
    norm = np.linalg.norm(encoding)
    return encoding / norm if norm > 0 else encoding


def pretrain_thalamus(thalamus, epochs: int = 100, lr: float = 0.1):
    print(f"[Pretrain] Training Thalamus ({epochs} epochs)...")
    area_map = {"code": 0, "math": 1, "memory": 2, "physics": 3, "language": 4}
    
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(TRAINING_DATA)
        
        for query, targets in TRAINING_DATA:
            x = encode_text(query)
            hidden = np.maximum(0, x @ thalamus.W1 + thalamus.b1)
            output = 1 / (1 + np.exp(-np.clip(hidden @ thalamus.W2 + thalamus.b2, -500, 500)))
            
            target = np.array([0.3, 0.3, 0.3, 0.2, 0.4])
            for area, value in targets.items():
                if area in area_map:
                    target[area_map[area]] = value
            
            error = target - output
            total_loss += np.mean(error ** 2)
            
            d_output = error * output * (1 - output)
            d_W2 = lr * np.outer(hidden, d_output)
            d_b2 = lr * d_output
            d_hidden = (d_output @ thalamus.W2.T) * (hidden > 0)
            d_W1 = lr * np.outer(x, d_hidden)
            d_b1 = lr * d_hidden
            
            thalamus.W2 += d_W2
            thalamus.b2 += d_b2
            thalamus.W1 += d_W1
            thalamus.b1 += d_b1
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: loss = {total_loss/len(TRAINING_DATA):.4f}")
    
    print(f"[Pretrain] Complete!")
    return thalamus


def save_weights(thalamus, path: str = WEIGHTS_PATH):
    weights = {
        "W1": thalamus.W1.tolist(),
        "b1": thalamus.b1.tolist(),
        "W2": thalamus.W2.tolist(),
        "b2": thalamus.b2.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(weights, f)
    print(f"[Pretrain] Saved to {path}")


def load_weights(thalamus, path: str = WEIGHTS_PATH) -> bool:
    if os.path.exists(path):
        with open(path, 'r') as f:
            weights = json.load(f)
        thalamus.W1 = np.array(weights["W1"])
        thalamus.b1 = np.array(weights["b1"])
        thalamus.W2 = np.array(weights["W2"])
        thalamus.b2 = np.array(weights["b2"])
        print(f"[Pretrain] Loaded weights from {path}")
        return True
    return False


def ensure_pretrained(thalamus):
    """Load weights if exist, otherwise train and save"""
    if not load_weights(thalamus):
        print("[Pretrain] No weights found, training...")
        pretrain_thalamus(thalamus, epochs=150, lr=0.1)
        save_weights(thalamus)
    return thalamus
