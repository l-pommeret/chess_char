import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ChessGPT
from trainer import ChessTrainer
from config import ModelConfig, TrainingConfig, GenerationConfig
from tokenizer import ChessTokenizer
from dataset import create_datasets
from parser import PGNParser

def train_local():
    # Paths
    pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
    output_dir = "./models/gpt2-chess-local"
    
    if not os.path.exists(pgn_path):
        # Try relative to script location if running from parent
        pgn_path = "chess_data/lichess_db_standard_rated_2013-09.pgn"
        if not os.path.exists(pgn_path):
             # Try absolute path based on previous findings
             pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
             if not os.path.exists(pgn_path):
                 print(f"Error: Data file not found at {pgn_path}")
                 return

    print(f"Using data from: {pgn_path}")

    # Initialize tokenizer
    tokenizer = ChessTokenizer()

    # Parse games
    # Limit games for speed if needed, but let's try full dataset first (236MB is smallish)
    # Actually, let's limit to 50k games to be safe and fast for this task
    MAX_GAMES = 50000 
    print(f"Parsing max {MAX_GAMES} games...")
    
    parser = PGNParser(tokenizer)
    games, filtered_count, total_count = parser.parse_file(pgn_path, max_games=MAX_GAMES)
    print(f"Games filtered: {filtered_count}/{total_count}")

    if not games:
        print("No valid games found.")
        return

    # Create datasets
    max_length = 256 # Reduced context for speed? Or keep 1024? 
    # Config says 1024. Let's stick to 1024 but maybe reduce if OOM.
    # 236MB text is ~200M chars. 50k games is ~25MB.
    # 50k games * ~500 chars = 25M chars.
    
    train_dataset, test_dataset = create_datasets(games, tokenizer, max_length=1024)
    print(f"Train dataset: {len(train_dataset)} games")
    print(f"Test dataset: {len(test_dataset)} games")

    # Config
    model_config = ModelConfig(vocab_size=len(tokenizer))
    # Larger model configuration
    model_config.n_layer = 16
    model_config.n_head = 16
    model_config.n_embd = 512
    
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=5, 
        per_device_train_batch_size=32,
        save_steps=1000,
        logging_steps=50
    )
    
    # Create model
    chess_gpt = ChessGPT(model_config, tokenizer)
    
    # Save tokenizer
    chess_gpt.save_tokenizer(os.path.join(training_config.output_dir, "tokenizer"))
    
    # Check GPU
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("Using CPU")

    # Train
    trainer = ChessTrainer(
        model=chess_gpt.model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=training_config
    )
    
    trainer.train()
    trainer.save_model()
    print("Training complete.")

if __name__ == "__main__":
    train_local()
