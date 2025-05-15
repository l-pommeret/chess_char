import os
import torch
from model import ChessGPT
from trainer import ChessTrainer
from config import ModelConfig, TrainingConfig, GenerationConfig
from tokenizer import ChessTokenizer
from dataset import create_datasets
from downloader import ChessDataDownloader
from parser import PGNParser

def main():
    # Chargement des données (supposons que train_dataset, test_dataset et tokenizer sont déjà créés)
    
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-09.pgn.zst"
    save_dir = "chess_data"
    max_length = 1024

    # Téléchargement et décompression
    downloader = ChessDataDownloader(url, save_dir)
    zst_file = downloader.download()
    pgn_file = downloader.decompress(zst_file)

    # Initialisation du tokenizer
    tokenizer = ChessTokenizer()

    # Parsing des parties
    parser = PGNParser(tokenizer)
    games, filtered_count, total_count = parser.parse_file(pgn_file)
    print(f"Parties filtrées : {filtered_count}/{total_count}")

    # Création des datasets
    train_dataset, test_dataset = create_datasets(games, tokenizer, max_length)
    print(f"Dataset d'entraînement : {len(train_dataset)} parties")
    print(f"Dataset de test : {len(test_dataset)} parties")

    # Configuration
    model_config = ModelConfig(vocab_size=len(tokenizer))
    training_config = TrainingConfig()
    generation_config = GenerationConfig()
    
    # Création du modèle
    chess_gpt = ChessGPT(model_config, tokenizer)
    
    # Sauvegarde du tokenizer
    chess_gpt.save_tokenizer(os.path.join(training_config.output_dir, "tokenizer"))
    
    # Vérification de la disponibilité du GPU
    if torch.cuda.is_available():
        print("GPU CUDA détecté et sera utilisé pour l'entraînement")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU MPS Apple Silicon détecté et sera utilisé pour l'entraînement")

    # Configuration et lancement de l'entraînement
    trainer = ChessTrainer(
        model=chess_gpt.model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=training_config
    )
    
    # Entraînement
    trainer.train()
    trainer.save_model()
    
    # Test de génération
    prompt = "1.e4 c5 2.Nf3 d6 3.d4"
    generated_game = chess_gpt.generate(prompt, generation_config)
    print(f"Partie générée : {generated_game}")

if __name__ == "__main__":
    main()