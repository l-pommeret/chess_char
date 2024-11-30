from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8

@dataclass
class TrainingConfig:
    output_dir: str = "./models/gpt2-chess"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 200
    warmup_steps: int = 500
    fp16: bool = True

@dataclass
class GenerationConfig:
    max_length: int = 100
    num_return_sequences: int = 1
    no_repeat_ngram_size: int = 2
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.1