"""
DeepSeek-Coder Service for code generation
"""
from llama_cpp import Llama
import os
from typing import Optional

class DeepSeekCoderService:
    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_path = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
        
    def load_model(self):
        """Load DeepSeek-Coder model"""
        if not self.model and os.path.exists(self.model_path):
            print("🚀 Loading DeepSeek-Coder...")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Larger context for code
                n_threads=4,
                n_batch=512,
                use_mmap=True,
                verbose=False
            )
            print("✅ DeepSeek-Coder loaded successfully")
        elif not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
    
    def generate_code(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate code using DeepSeek-Coder"""
        if not self.model:
            self.load_model()
            
        if not self.model:
            return "DeepSeek-Coder model not available"
        
        # DeepSeek-Coder specific prompt format
        formatted_prompt = f"""### Instruction:
{prompt}

### Response:
"""
        
        response = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.1,  # Lower temp for precise code
            top_p=0.95,
            stop=["### Instruction:", "### End"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()

# Global instance
deepseek_coder = DeepSeekCoderService()
