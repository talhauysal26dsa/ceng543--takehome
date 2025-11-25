from typing import Optional
import random
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "t5-small",
        max_new_tokens: int = 128,
        num_beams: int = 4,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def generate(self, question: str, context: str) -> str:
        # T5 format: "question: ... context: ..."
        prompt = f"question: {question} context: {context}"
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
            length_penalty=0.6,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

__all__ = ["AnswerGenerator", "set_seed"]
