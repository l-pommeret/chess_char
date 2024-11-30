# inference.py

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import os

@dataclass
class InferenceConfig:
    """Configuration pour l'inférence."""
    model_name: str
    max_length: int = 200
    num_return_sequences: int = 1
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_beams: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ChessGenerator:
    """Générateur de parties d'échecs utilisant un modèle HuggingFace."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.model, self.tokenizer = self._load_model()

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_model(self):
        """Charge le modèle et le tokenizer depuis HuggingFace Hub."""
        try:
            self.logger.info(f"Loading model from {self.config.model_name}")
            
            # Chargement du modèle
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            model.to(self.config.device)
            model.eval()
            
            # Téléchargement du fichier tokenizer depuis HF Hub
            tokenizer_path = hf_hub_download(
                repo_id=self.config.model_name,
                filename="tokenizer.json",
                repo_type="model"
            )
            
            # Chargement du tokenizer personnalisé
            with open(tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            
            # Création d'une instance de notre tokenizer personnalisé
            from tokenizer import ChessTokenizer, TokenizerConfig
            tokenizer_config = TokenizerConfig()
            tokenizer = ChessTokenizer(tokenizer_config)
            
            # Configuration du tokenizer avec les données chargées
            tokenizer.vocab = tokenizer_data['vocab']
            tokenizer.id2token = {int(k): v for k, v in tokenizer_data['id2token'].items()}
            tokenizer.valid_chars = set(tokenizer_data['valid_chars'])
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt: str) -> List[str]:
        """Génère une suite à un début de partie d'échecs."""
        try:
            # Encodage du prompt
            inputs = torch.tensor([self.tokenizer.encode(prompt)]).to(self.config.device)
            
            # Configuration de la génération
            gen_kwargs = {
                'max_length': self.config.max_length,
                'num_return_sequences': self.config.num_return_sequences,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'temperature': self.config.temperature,
                'repetition_penalty': self.config.repetition_penalty,
                'do_sample': self.config.do_sample,
                'pad_token_id': self.tokenizer.vocab['<PAD>']
            }
            
            if self.config.num_beams:
                gen_kwargs['num_beams'] = self.config.num_beams
            
            # Génération
            self.logger.info("Generating continuation...")
            with torch.no_grad():
                output_sequences = self.model.generate(
                    inputs,
                    **gen_kwargs
                )
            
            # Décodage des séquences générées
            generated_sequences = []
            for sequence in output_sequences:
                generated_text = self.tokenizer.decode(sequence.tolist())
                generated_sequences.append(generated_text)
            
            return generated_sequences
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            raise

    def validate_game(self, game: str) -> bool:
        """Valide une partie d'échecs générée."""
        try:
            # Vérifie que la partie contient des coups valides
            if not any(char.isdigit() for char in game):
                return False
                
            # Vérifie qu'il y a au moins un coup
            if not '.' in game:
                return False
                
            # Vérifie que la partie se termine correctement
            valid_endings = {'1-0', '0-1', '1/2-1/2', '*'}
            if not any(ending in game for ending in valid_endings):
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating game: {str(e)}")
            return False

    def generate_valid_game(self, prompt: str, max_attempts: int = 5) -> Optional[str]:
        """Génère une partie d'échecs valide."""
        for attempt in range(max_attempts):
            self.logger.info(f"Generation attempt {attempt + 1}/{max_attempts}")
            
            sequences = self.generate(prompt)
            for sequence in sequences:
                if self.validate_game(sequence):
                    return sequence
                    
        self.logger.warning("Failed to generate a valid game")
        return None

def main():
    # Configuration
    config = InferenceConfig(
        model_name="Zual/chess",
        max_length=200,
        temperature=0.8,
        num_return_sequences=1
    )
    
    # Création du générateur
    generator = ChessGenerator(config)
    
    # Exemple de génération
    prompt = "1. e4 e5 2. Nf3 "
    print(f"\nPrompt: {prompt}")
    
    # Génération d'une partie valide
    generated_game = generator.generate_valid_game(prompt)
    
    if generated_game:
        print("\nPartie générée :")
        print(generated_game)
    else:
        print("\nÉchec de la génération")

if __name__ == "__main__":
    main()