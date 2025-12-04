import sys
import os
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import json

# Add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import ChessTokenizer

def calculate_llm_entropy():
    # Use the locally trained model - Checkpoint 1 (Epoch 1)
    model_path = "./models/gpt2-chess-local"
    # Tokenizer is in the parent directory's tokenizer folder
    tokenizer_base_path = "./models/gpt2-chess-local"
    pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_local.py first.")
        return

    print(f"Loading local model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading tokenizer from local path...")
    try:
        tokenizer_path = os.path.join(tokenizer_base_path, "tokenizer")
        json_path = os.path.join(tokenizer_path, "tokenizer.json")
        
        try:
            tokenizer = ChessTokenizer.load(json_path)
            print(f"Loaded tokenizer from {json_path}")
        except KeyError:
            print("Standard load failed (missing config), trying manual load...")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            from tokenizer import TokenizerConfig
            config = TokenizerConfig()
            tokenizer = ChessTokenizer(config)
            
            # Rebuild vocab from id2token because vocab might be incomplete or just to be safe
            if 'id2token' in data:
                id2token = {int(k): v for k, v in data['id2token'].items()}
                vocab = {v: k for k, v in id2token.items()}
                tokenizer.vocab = vocab
                tokenizer.id2token = id2token
            else:
                tokenizer.vocab = data['vocab']
                
            tokenizer.valid_chars = set(data['valid_chars'])
            print(f"Loaded tokenizer manually. Vocab size: {len(tokenizer.vocab)}")

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")
        
    model.eval()
    
    total_nll = 0
    total_tokens = 0
    
    batch_size = 1
    max_length = 1024
    
    print(f"Reading {pgn_path}...")
    
    # Use PGNParser to extract moves
    from parser import PGNParser
    
    # We need a dummy tokenizer for parser or use the loaded one if compatible
    # Parser uses tokenizer.is_valid_text
    # Let's just use the loaded tokenizer
    parser = PGNParser(tokenizer)
    
    # Parse file to get games (moves only)
    # We limit to a subset to be fast
    games, _, _ = parser.parse_file(pgn_path, max_games=100)
    
    if not games:
        print("No games found.")
        return
        
    print(f"Evaluated on {len(games)} games.")
    
    print(f"Evaluated on {len(games)} games.")
    
    total_nll = 0
    total_tokens = 0
    
    for game in tqdm(games):
        # Tokenize game
        # We don't pad here, we just evaluate the game content
        input_ids = tokenizer.encode(game)
        
        # Skip empty or too short games
        if len(input_ids) < 2:
            continue
            
        # Truncate if too long (though parser limits moves, encoded might be longer?)
        # Model max position is 1024
        if len(input_ids) > 1024:
            input_ids = input_ids[:1024]
            
        input_tensor = torch.tensor([input_ids])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        with torch.no_grad():
            outputs = model(input_tensor, labels=input_tensor)
            # outputs.loss is the average NLL over the sequence
            # We want the sum
            nll = outputs.loss.item() * len(input_ids)
            
        total_nll += nll
        total_tokens += len(input_ids)
        
    if total_tokens == 0:
        print("No tokens processed.")
        return 0
        
    avg_nll = total_nll / total_tokens
    entropy_bits = avg_nll / math.log(2)
    
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Average NLL: {avg_nll:.4f} nats/token")
    print(f"Level 5 - Entropy of LLM: {entropy_bits:.4f} bits/char")
    
    return entropy_bits

if __name__ == "__main__":
    calculate_llm_entropy()
