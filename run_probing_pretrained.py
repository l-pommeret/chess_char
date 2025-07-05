#!/usr/bin/env python3
"""
Script for running chess probing with the pre-trained Zual/chess_char model.
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('.')

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from probing import run_probing_experiment, ProbingConfig
from board_utils import ChessBoardEncoder


class HuggingFaceChessModel:
    """Wrapper for the Hugging Face chess model."""
    
    def __init__(self, model_name: str = "Zual/chess_char"):
        print(f"Loading model {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
        print(f"Model layers: {self.model.config.n_layer}")
        print(f"Hidden size: {self.model.config.n_embd}")
        print(f"Vocab size: {self.model.config.vocab_size}")


def create_extended_demo_games() -> list:
    """Create a larger set of demo chess games for better probing."""
    return [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 c6 6. e3 Bf5 7. Qf3 Bg6",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. f3 h5",
        "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 6. e4 e5 7. Nc3 c6",
        "1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Be6 6. Nf3 Nc6 7. O-O Be7",
        "1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 Qc7",
        "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6",
        "1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7",
        "1. Nf3 d5 2. g3 Nf6 3. Bg2 e6 4. O-O Be7 5. d3 O-O 6. Nbd2 c5 7. e4 Nc6",
        "1. e4 g6 2. d4 Bg7 3. Nc3 d6 4. f4 Nf6 5. Nf3 O-O 6. Bd3 Na6 7. O-O c5",
        "1. d4 e6 2. c4 f5 3. g3 Nf6 4. Bg2 Be7 5. Nf3 O-O 6. O-O d6 7. Nc3 Qe8",
        "1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 c5 6. Bb5+ Bd7 7. e5 Nh5",
        "1. Nf3 c5 2. c4 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e6 6. g3 Qb6 7. Nb3 Ne5",
        "1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 g6 5. Bc4 Nb6 6. Bb3 Nc6 7. exd6 cxd6",
        "1. d4 d6 2. e4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 O-O 6. Bd3 Na6 7. O-O c5",
        "1. c4 Nf6 2. Nc3 e6 3. e4 d5 4. cxd5 exd5 5. e5 Ne4 6. Nf3 Bf5 7. Be2 Nc6",
        "1. e4 c5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. d3 d6 6. f4 e5 7. Nf3 Nge7",
        "1. d4 f5 2. g3 Nf6 3. Bg2 e6 4. Nf3 Be7 5. O-O O-O 6. c4 d6 7. Nc3 Qe8",
        "1. Nf3 Nf6 2. c4 g6 3. g3 Bg7 4. Bg2 O-O 5. O-O d6 6. d4 Nbd7 7. Nc3 e5",
        "1. e4 e5 2. Bc4 Nf6 3. d3 c6 4. Nf3 d5 5. Bb3 a5 6. c3 Bd6 7. Bc2 O-O",
        "1. f4 d5 2. Nf3 g6 3. e3 Bg7 4. Be2 Nf6 5. O-O O-O 6. d3 c5 7. Qe1 Nc6",
        "1. b3 e5 2. Bb2 Nc6 3. e3 d5 4. Bb5 Bd6 5. f4 Qe7 6. Nf3 f6 7. O-O Bd7",
        "1. g3 d5 2. Bg2 Nf6 3. Nf3 e6 4. O-O Be7 5. d3 O-O 6. Nbd2 c5 7. e4 Nc6",
        "1. Nc3 d5 2. e4 d4 3. Nce2 e5 4. Ng3 Be6 5. Bc4 Nc6 6. d3 f5 7. exf5 Bxf5",
        "1. h3 e5 2. e4 Nf6 3. Nc3 Bb4 4. f4 d6 5. fxe5 dxe5 6. Nf3 O-O 7. Bc4 Re8"
    ]


def extract_representations_hf(model: HuggingFaceChessModel, text_games: list, device: str) -> dict:
    """Extract hidden representations from Hugging Face model."""
    model.model.eval()
    
    representations = {}
    num_layers = model.model.config.n_layer
    
    # Initialize storage for each layer
    for layer in range(num_layers):
        representations[f'layer_{layer}'] = []
    
    encoder = ChessBoardEncoder()
    
    with torch.no_grad():
        for game_text in text_games:
            print(f"Processing game: {game_text[:50]}...")
            
            # Get board positions for this game
            positions = encoder.moves_string_to_positions(game_text)
            
            if not positions:
                print(f"  Could not parse game, skipping...")
                continue
            
            # Tokenize the game
            try:
                inputs = model.tokenizer(
                    game_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Get hidden states from model
                outputs = model.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states  # (layer, batch, seq_len, hidden_dim)
                
                # Extract representations for each layer
                for layer_idx in range(num_layers):
                    layer_hidden = hidden_states[layer_idx + 1]  # +1 because layer 0 is embeddings
                    
                    # Use the last non-padded token as the game representation
                    seq_len = attention_mask.sum().item()
                    game_repr = layer_hidden[0, seq_len-1, :].cpu()  # (hidden_dim,)
                    
                    # Associate with each position in the game
                    for position in positions:
                        representations[f'layer_{layer_idx}'].append((game_repr, position))
                        
                print(f"  Extracted {len(positions)} positions")
                        
            except Exception as e:
                print(f"  Error processing game: {e}")
                continue
    
    return representations


def main():
    print("="*60)
    print("CHESS PROBING WITH PRE-TRAINED MODEL")
    print("="*60)
    
    # Load the pre-trained model
    print("1. Loading pre-trained model...")
    chess_model = HuggingFaceChessModel("Zual/chess_char")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chess_model.model.to(device)
    print(f"   Using device: {device}")
    
    # Create demo games
    print("\n2. Loading demo games...")
    games = create_extended_demo_games()
    print(f"   Loaded {len(games)} demo games")
    
    # Extract representations
    print("\n3. Extracting representations...")
    representations = extract_representations_hf(chess_model, games, device)
    
    # Print dataset statistics
    total_positions = 0
    for layer_key, layer_reprs in representations.items():
        print(f"   {layer_key}: {len(layer_reprs)} position-representation pairs")
        total_positions += len(layer_reprs)
    
    print(f"   Total positions: {total_positions}")
    
    if total_positions == 0:
        print("ERROR: No positions extracted. Check game parsing.")
        return
    
    # Configure probing
    print("\n4. Configuring probing experiment...")
    num_layers = chess_model.model.config.n_layer
    hidden_dim = chess_model.model.config.n_embd
    
    probing_config = ProbingConfig(
        hidden_dim=hidden_dim,
        probe_layers=list(range(num_layers)),  # Probe all layers
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=15,  # More epochs for better results
        device=device,
        save_dir='./pretrained_probing_results'
    )
    
    print(f"   Will probe layers: {probing_config.probe_layers}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Training for {probing_config.num_epochs} epochs")
    
    # Train classifiers manually (adapted from probing.py)
    print("\n5. Training probes...")
    
    from probing import ChessProber
    
    # Create custom prober
    prober = ChessProber(probing_config)
    
    # Split data
    train_representations = {}
    test_representations = {}
    
    for layer_key, layer_reprs in representations.items():
        split_idx = int(0.8 * len(layer_reprs))
        train_representations[layer_key] = layer_reprs[:split_idx]
        test_representations[layer_key] = layer_reprs[split_idx:]
        print(f"   {layer_key}: {len(train_representations[layer_key])} train, {len(test_representations[layer_key])} test")
    
    # Train probes
    print("\n6. Training classifiers...")
    prober.train_probes(train_representations)
    
    # Evaluate probes
    print("\n7. Evaluating probes...")
    results = prober.evaluate_probes(test_representations)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for layer_key, layer_results in results.items():
        accuracies = list(layer_results.values())
        print(f"\n{layer_key.upper()}:")
        print(f"  Mean accuracy: {np.mean(accuracies):.3f}")
        print(f"  Std deviation: {np.std(accuracies):.3f}")
        print(f"  Min accuracy:  {np.min(accuracies):.3f}")
        print(f"  Max accuracy:  {np.max(accuracies):.3f}")
        
        # Show best squares
        sorted_squares = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
        print(f"  Best squares:  {sorted_squares[:5]}")
    
    # Save results
    print("\n8. Saving results...")
    prober.visualize_results(results)
    prober.save_results(results)
    prober.save_models()
    
    print(f"\nResults saved to: {probing_config.save_dir}")
    print("Run the analysis script to see detailed visualizations:")
    print(f"python analyze_probing.py --results_path {probing_config.save_dir}/probing_results.json")


if __name__ == '__main__':
    main()