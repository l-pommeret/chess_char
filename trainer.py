import os
from transformers import Trainer, TrainingArguments
from typing import Optional
import torch

from config import TrainingConfig

class ChessTrainer:
    def __init__(self, model, train_dataset, test_dataset, config: TrainingConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.trainer = self._setup_trainer()
        
    def _setup_trainer(self) -> Trainer:
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            lr_scheduler_type="linear",
            warmup_steps=self.config.warmup_steps,
            fp16=self.config.fp16
        )
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )
        
    def train(self):
        print("Début de l'entraînement...")
        self.trainer.train()
        print("Entraînement terminé!")
        
    def save_model(self, path: Optional[str] = None):
        save_path = path or self.config.output_dir
        self.trainer.save_model(save_path)
        print(f"Modèle sauvegardé dans {save_path}")