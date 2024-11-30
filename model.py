import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict

from config import ModelConfig
from config import GenerationConfig

class ChessGPT:
    def __init__(self, config: ModelConfig, tokenizer):
        self.tokenizer = tokenizer
        self.config = self._create_gpt_config(config)
        self.model = GPT2LMHeadModel(self.config)
        
    def _create_gpt_config(self, config: ModelConfig) -> GPT2Config:
        return GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_ctx=config.n_ctx,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head
        )
        
    def save_tokenizer(self, save_dir: str):
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        tokenizer_data = {
            'vocab': self.tokenizer.vocab,
            'id2token': self.tokenizer.id2token,
            'valid_chars': list(self.tokenizer.valid_chars)
        }
        
        with open(os.path.join(save_dir, "tokenizer.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load_tokenizer(cls, load_dir: str) -> Dict:
        import json
        
        with open(os.path.join(load_dir, "tokenizer.json"), 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.model.device)
        
        output = self.model.generate(
            input_ids,
            max_length=generation_config.max_length,
            num_return_sequences=generation_config.num_return_sequences,
            no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
            do_sample=generation_config.do_sample,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            pad_token_id=self.tokenizer.vocab['<PAD>']
        )
        
        return self.tokenizer.decode(output[0].tolist())