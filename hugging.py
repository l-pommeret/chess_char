# huggingface_utils.py

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from huggingface_hub import notebook_login, HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoConfig

@dataclass
class HuggingFaceConfig:
    """Configuration pour l'upload vers HuggingFace Hub."""
    checkpoint_path: str
    tokenizer_path: str
    repo_name: str
    model_name: str = "gpt2-chess"
    tokenizer_file: str = "tokenizer.json"
    use_auth_token: bool = True
    trust_remote_code: bool = True

class HuggingFaceUploader:
    """Gère l'upload des modèles vers HuggingFace Hub."""
    
    def __init__(self, config: HuggingFaceConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.paths = self._setup_paths()
        self.api = HfApi()

    def _setup_logging(self) -> logging.Logger:
        """Configure le logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _setup_paths(self) -> Dict[str, Path]:
        """Configure et vérifie les chemins."""
        paths = {
            'checkpoint': Path(self.config.checkpoint_path).resolve(),
            'tokenizer': Path(self.config.tokenizer_path).resolve(),
            'final_model': Path('final_model').resolve()
        }
        
        # Création des répertoires nécessaires
        paths['final_model'].mkdir(exist_ok=True)
        
        # Vérification des chemins
        for key, path in paths.items():
            if not path.exists() and key != 'final_model':
                raise FileNotFoundError(f"{key} path does not exist: {path}")
            self.logger.info(f"Using {key} path: {path}")
            
        return paths

    def login(self) -> None:
        """Se connecte à HuggingFace Hub."""
        try:
            notebook_login()
            self.logger.info('Successfully logged in to Hugging Face')
        except Exception as e:
            self.logger.error(f"Failed to login: {str(e)}")
            raise

    def prepare_model(self) -> AutoModelForCausalLM:
        """Prépare le modèle pour l'upload."""
        try:
            self.logger.info(f"Loading model from {self.paths['checkpoint']}")
            
            # Chargement de la configuration
            config = AutoConfig.from_pretrained(
                self.paths['checkpoint'],
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Chargement du modèle
            model = AutoModelForCausalLM.from_pretrained(
                self.paths['checkpoint'],
                config=config,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error preparing model: {str(e)}")
            raise

    def upload_tokenizer(self) -> None:
        """Upload le tokenizer vers HuggingFace Hub."""
        tokenizer_file = self.paths['tokenizer'] / self.config.tokenizer_file
        
        if not tokenizer_file.exists():
            self.logger.error(f"Tokenizer file not found: {tokenizer_file}")
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
            
        try:
            self.logger.info(f"Uploading tokenizer from: {tokenizer_file}")
            self.api.upload_file(
                path_or_fileobj=str(tokenizer_file),
                path_in_repo=tokenizer_file.name,
                repo_id=self.config.repo_name,
                repo_type="model",
                use_auth_token=self.config.use_auth_token
            )
            self.logger.info('Tokenizer uploaded successfully')
            
        except Exception as e:
            self.logger.error(f"Error uploading tokenizer: {str(e)}")
            raise

    def verify_upload(self) -> None:
        """Vérifie les fichiers uploadés sur le Hub."""
        try:
            files = self.api.list_repo_files(self.config.repo_name)
            self.logger.info('\nFiles in repository:')
            for file in files:
                self.logger.info(f'- {file}')
                
        except Exception as e:
            self.logger.error(f"Error verifying upload: {str(e)}")
            raise

    def upload(self) -> None:
        """Processus complet d'upload."""
        try:
            # Login
            self.login()
            
            # Préparation et upload du modèle
            model = self.prepare_model()
            self.logger.info(f"Pushing model to hub: {self.config.repo_name}")
            model.push_to_hub(
                self.config.repo_name,
                use_auth_token=self.config.use_auth_token
            )
            
            # Upload du tokenizer
            self.upload_tokenizer()
            
            # Vérification
            self.verify_upload()
            
            self.logger.info("Upload completed successfully")
            
        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            raise

# main.py

def save_to_hub(checkpoint_path: str, tokenizer_path: str, repo_name: str):
    """Fonction principale pour sauvegarder le modèle sur HuggingFace Hub."""
    config = HuggingFaceConfig(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        repo_name=repo_name
    )
    
    uploader = HuggingFaceUploader(config)
    uploader.upload()

if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "./gpt2-chess-games/checkpoint-1000"
    TOKENIZER_PATH = "./gpt2-chess-games/tokenizer"
    REPO_NAME = "Zual/chess"
    
    # Upload
    save_to_hub(CHECKPOINT_PATH, TOKENIZER_PATH, REPO_NAME)