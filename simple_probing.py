#!/usr/bin/env python3
"""
Simple probing script that works with the Zual/chess_char model.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import chess

# Import the existing probing infrastructure
from probing import ChessProber, ProbingConfig
from board_utils import ChessBoardEncoder


class SimpleChessModel:
    """Simple wrapper for the chess model that handles tokenization issues."""
    
    def __init__(self):
        print("Loading Zual/chess_char model...")
        
        # Load model directly
        from transformers import GPT2LMHeadModel
        self.model = GPT2LMHeadModel.from_pretrained("Zual/chess_char")
        
        # Create a simple character-level tokenizer based on chess notation
        self.char_to_id = {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
            '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15,
            'R': 16, 'N': 17, 'B': 18, 'Q': 19, 'K': 20, 'P': 21,
            'x': 22, '+': 23, '#': 24, '=': 25, 'O': 26, '-': 27, 
            '.': 28, ' ': 29, '\n': 30, '(': 31, ')': 32, '[': 33, ']': 34,
            '0': 35, '9': 36, '/': 37, "'": 38, '"': 39, '!': 40, '?': 41,
            '<PAD>': 42
        }
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
        print(f"Model loaded successfully!")
        print(f"Model layers: {self.model.config.n_layer}")
        print(f"Hidden size: {self.model.config.n_embd}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return [self.char_to_id.get(char, self.char_to_id['<PAD>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text."""
        return ''.join([self.id_to_char.get(id, '<UNK>') for id in token_ids])


def extract_representations(model: SimpleChessModel, games: List[str], device: str) -> Dict[str, List[Tuple[torch.Tensor, np.ndarray]]]:
    """Extract hidden representations from the model."""
    model.model.eval()
    
    num_layers = model.model.config.n_layer
    representations = {}
    
    # Initialize storage for each layer
    for layer in range(num_layers):
        representations[f'layer_{layer}'] = []
    
    encoder = ChessBoardEncoder()
    
    print(f"Extracting representations from {len(games)} games...")
    
    with torch.no_grad():
        for game_idx, game_text in enumerate(games):
            print(f"Processing game {game_idx + 1}/{len(games)}: {game_text[:50]}...")
            
            # Get board positions
            positions = encoder.moves_string_to_positions(game_text)
            if not positions:
                print(f"  Could not parse game, skipping...")
                continue
            
            try:
                # Tokenize
                token_ids = model.encode(game_text)
                input_ids = torch.tensor([token_ids[:512]]).to(device)  # Truncate to 512 tokens
                
                # Get hidden states
                outputs = model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # Extract for each layer
                for layer_idx in range(num_layers):
                    if layer_idx + 1 < len(hidden_states):
                        layer_hidden = hidden_states[layer_idx + 1]  # +1 because first is embeddings
                        
                        # Use the last token as the game representation
                        game_repr = layer_hidden[0, -1, :].cpu()  # (hidden_dim,)
                        
                        # Associate with each position in the game
                        for position in positions:
                            representations[f'layer_{layer_idx}'].append((game_repr, position))
                
                print(f"  Extracted {len(positions)} positions")
                
            except Exception as e:
                print(f"  Error processing game: {e}")
                continue
    
    return representations


def create_test_games() -> List[str]:
    """Create test games for probing."""
    return [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 c6",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6",
        "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6",
        "1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Be6",
        "1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+",
        "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O",
        "1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6",
        "1. Nf3 d5 2. g3 Nf6 3. Bg2 e6 4. O-O Be7 5. d3 O-O",
        "1. e4 g6 2. d4 Bg7 3. Nc3 d6 4. f4 Nf6 5. Nf3 O-O",
        "1. d4 e6 2. c4 f5 3. g3 Nf6 4. Bg2 Be7 5. Nf3 O-O",
        "1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 c5",
        "1. Nf3 c5 2. c4 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e6",
        "1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 g6 5. Bc4 Nb6",
        "1. d4 d6 2. e4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 O-O"
    ]


def main():
    print("="*60)
    print("SIMPLE CHESS PROBING EXPERIMENT")
    print("="*60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    try:
        chess_model = SimpleChessModel()
        chess_model.model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have transformers and torch installed:")
        print("pip install transformers torch chess")
        return
    
    # Create games
    games = create_test_games()
    print(f"\nUsing {len(games)} test games")
    
    # Extract representations
    print("\n" + "="*50)
    print("EXTRACTING REPRESENTATIONS")
    print("="*50)
    
    try:
        representations = extract_representations(chess_model, games, device)
    except Exception as e:
        print(f"Error extracting representations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check data
    total_samples = sum(len(layer_reprs) for layer_reprs in representations.values())
    if total_samples == 0:
        print("ERROR: No representations extracted!")
        return
    
    print(f"\nExtracted {total_samples} total samples:")
    for layer_key, layer_reprs in representations.items():
        print(f"  {layer_key}: {len(layer_reprs)} samples")
    
    # Configure probing
    print("\n" + "="*50)
    print("CONFIGURING PROBING")
    print("="*50)
    
    hidden_dim = chess_model.model.config.n_embd
    num_layers = chess_model.model.config.n_layer
    
    probing_config = ProbingConfig(
        hidden_dim=hidden_dim,
        probe_layers=list(range(num_layers)),
        batch_size=16,  # Smaller batch for demo
        learning_rate=1e-3,
        num_epochs=8,   # Moderate epochs
        device=device,
        save_dir='./simple_probing_results'
    )
    
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Probing layers: {probing_config.probe_layers}")
    print(f"Training epochs: {probing_config.num_epochs}")
    
    # Create prober and train
    print("\n" + "="*50)
    print("TRAINING PROBES")
    print("="*50)
    
    try:
        prober = ChessProber(probing_config)
        
        # Split data
        train_representations = {}
        test_representations = {}
        
        for layer_key, layer_reprs in representations.items():
            split_idx = int(0.8 * len(layer_reprs))
            train_representations[layer_key] = layer_reprs[:split_idx]
            test_representations[layer_key] = layer_reprs[split_idx:]
            print(f"  {layer_key}: {len(train_representations[layer_key])} train, {len(test_representations[layer_key])} test")
        
        # Train
        prober.train_probes(train_representations)
        
        # Evaluate
        print("\n" + "="*50)
        print("EVALUATING PROBES")
        print("="*50)
        
        results = prober.evaluate_probes(test_representations)
        
        # Display results
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        for layer_key, layer_results in results.items():
            accuracies = list(layer_results.values())
            print(f"\n{layer_key.upper()}:")
            print(f"  Mean accuracy: {np.mean(accuracies):.3f}")
            print(f"  Std deviation: {np.std(accuracies):.3f}")
            print(f"  Min accuracy:  {np.min(accuracies):.3f}")
            print(f"  Max accuracy:  {np.max(accuracies):.3f}")
            
            # Show best squares
            sorted_squares = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3 squares: {sorted_squares[:3]}")
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        prober.visualize_results(results)
        prober.save_results(results)
        prober.save_models()
        
        print(f"Results saved to: {probing_config.save_dir}")
        print("\nTo analyze results:")
        print(f"python analyze_probing.py --results_path {probing_config.save_dir}/probing_results.json")
        
    except Exception as e:
        print(f"Error during probing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()