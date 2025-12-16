"""Hormone Network"""
import numpy as np
import json
import os
 
class HormoneNetwork:
    WEIGHTS_PATH = "/root/openhermes_backend/hormone_network_weights.json"
    
    def __init__(self, input_size=256, h1=128, h2=64, output_size=7, lr=0.1):
        self.input_size = input_size
        self.lr = lr
        self.W1 = np.random.randn(input_size, h1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, output_size) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_size)
        self.training_steps = 0
        self.load_weights()
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, text):
        enc = np.zeros(self.input_size)
        t = text.lower()
        for i, b in enumerate(t.encode()[:200]):
            enc[i % self.input_size] += (b - 128) / 128.0
        kw = {
            "angry": 200, "furious": 200, "hostile": 201, "insulting": 201,
            "frustrated": 202, "annoyed": 202, "irritated": 202,
            "curious": 203, "inquisitive": 203, "interested": 203,
            "excited": 204, "thrilled": 204, "elated": 204, "happy": 204,
            "grateful": 205, "thankful": 205, "appreciative": 205,
            "sad": 206, "depressed": 206, "lonely": 206, "melancholy": 206,
            "anxious": 207, "worried": 207, "nervous": 207, "scared": 207,
            "manipulation": 210, "manipulative": 210, "deceptive": 210,
            "gaslighting": 211, "lying": 212, "dishonest": 212,
            "jailbreak": 213, "bypass": 213,
            "guilt": 214, "victim": 214, "accusatory": 214,
            "genuine": 220, "sincere": 220, "authentic": 220,
            "testing": 225, "challenging": 225, "provocative": 225,
            "attacking": 227, "attack": 227, "hurting": 227,
            "warmth": 230, "warm": 230, "comfort": 230,
            "joy": 232, "joyful": 232, "celebrating": 232,
            "thanks": 233, "appreciation": 233, "gratitude": 233,
            "playful": 239, "joking": 239, "fun": 239, "silly": 239,
            "bored": 240, "uninterested": 240, "apathetic": 240,
            "neutral": 241, "calm": 241,
            "impatient": 242, "demanding": 242, "urgent": 242,
            "flattery": 243, "praise": 243,
            "passive": 245, "sarcastic": 245, "mocking": 245,
            "aggressive": 246, "threatening": 246, "violent": 246,
            "contempt": 247, "dismissive": 247, "condescending": 247,
        }
        for word, idx in kw.items():
            if word in t:
                enc[idx] = 1.0
        norm = np.linalg.norm(enc)
        return enc / norm if norm > 0 else enc
    
    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        out = self._sigmoid(z3)
        return out, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
    
    def predict(self, perception):
        x = self.encode(perception)
        out, _ = self.forward(x)
        return out
    
    def predict_dict(self, perception):
        out = self.predict(perception)
        names = ["dopamine", "serotonin", "cortisol", "adrenaline", "oxytocin", "norepinephrine", "endorphins"]
        return {n: float(out[i]) for i, n in enumerate(names)}
    
    def train_step(self, perception, target):
        target = np.array(target)
        x = self.encode(perception)
        out, c = self.forward(x)
        loss = np.mean((out - target) ** 2)
        d_out = 2 * (out - target) / 7
        d_z3 = d_out * out * (1 - out)
        self.W3 -= self.lr * np.outer(c["a2"], d_z3)
        self.b3 -= self.lr * d_z3
        d_a2 = d_z3 @ self.W3.T
        d_z2 = d_a2 * (c["z2"] > 0)
        self.W2 -= self.lr * np.outer(c["a1"], d_z2)
        self.b2 -= self.lr * d_z2
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (c["z1"] > 0)
        self.W1 -= self.lr * np.outer(c["x"], d_z1)
        self.b1 -= self.lr * d_z1
        self.training_steps += 1
        return loss
    
    def save_weights(self, path=None):
        path = path or self.WEIGHTS_PATH
        data = {"W1": self.W1.tolist(), "b1": self.b1.tolist(), "W2": self.W2.tolist(), "b2": self.b2.tolist(), "W3": self.W3.tolist(), "b3": self.b3.tolist(), "steps": self.training_steps}
        with open(path, "w") as f:
            json.dump(data, f)
        print("[Network] Saved (%d steps)" % self.training_steps)
    
    def load_weights(self, path=None):
        path = path or self.WEIGHTS_PATH
        if not os.path.exists(path):
            print("[Network] No weights found, using random init")
            return False
        with open(path, "r") as f:
            d = json.load(f)
        self.W1 = np.array(d["W1"])
        self.b1 = np.array(d["b1"])
        self.W2 = np.array(d["W2"])
        self.b2 = np.array(d["b2"])
        self.W3 = np.array(d["W3"])
        self.b3 = np.array(d["b3"])
        self.training_steps = d.get("steps", 0)
        print("[Network] Loaded (%d steps)" % self.training_steps)
        return True
 
hormone_network = HormoneNetwork()
