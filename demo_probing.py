#!/usr/bin/env python3
"""
Demo script for chess probing with random model weights.
This script demonstrates the probing system without requiring a pre-trained model.
"""

import torch
import numpy as np
from pathlib import Path
import json

from model import ChessGPT
from tokenizer import ChessTokenizer
from config import ModelConfig
from probing import run_probing_experiment, ProbingConfig


def create_demo_games() -> list:
    """Create a set of demo chess games for testing."""
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
        "1. d4 d6 2. e4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 O-O",
        "1. c4 Nf6 2. Nc3 e6 3. e4 d5 4. cxd5 exd5 5. e5 Ne4",
        "1. e4 c5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. d3 d6",
        "1. d4 f5 2. g3 Nf6 3. Bg2 e6 4. Nf3 Be7 5. O-O O-O",
        "1. Nf3 Nf6 2. c4 g6 3. g3 Bg7 4. Bg2 O-O 5. O-O d6",
        "1. e4 e5 2. Bc4 Nf6 3. d3 c6 4. Nf3 d5 5. Bb3 a5"
    ]


def main():
    print("="*60)
    print("CHESS PROBING DEMO")
    print("="*60)
    print("This demo shows how the probing system works.")
    print("Note: Using random model weights for demonstration.")
    print()
    
    # Initialize tokenizer and model
    print("1. Initializing model...")
    tokenizer = ChessTokenizer()
    model_config = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        n_embd=64,
        n_layer=2,
        n_head=2
    )
    
    chess_model = ChessGPT(model_config, tokenizer)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chess_model.model.to(device)
    print(f"   Using device: {device}")
    print(f"   Model has {model_config.n_layer} layers")
    print(f"   Hidden dimension: {model_config.n_embd}")
    
    # Create demo games
    print("\n2. Loading demo games...")
    games = create_demo_games()
    print(f"   Loaded {len(games)} demo games")
    
    # Configure probing
    print("\n3. Configuring probing experiment...")
    probing_config = ProbingConfig(
        hidden_dim=model_config.n_embd,
        probe_layers=[0, 1],  # Probe both layers
        batch_size=16,        # Smaller batch for demo
        learning_rate=1e-3,
        num_epochs=5,         # Fewer epochs for demo
        device=device,
        save_dir='./demo_probing_results'
    )
    
    print(f"   Will probe layers: {probing_config.probe_layers}")
    print(f"   Training for {probing_config.num_epochs} epochs")
    print(f"   Batch size: {probing_config.batch_size}")
    
    # Run probing experiment
    print("\n4. Running probing experiment...")
    print("   This may take several minutes...")
    
    try:
        results = run_probing_experiment(chess_model, games, probing_config)
        
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
            
            # Show best and worst squares
            sorted_squares = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
            print(f"  Best squares:  {sorted_squares[:3]}")
            print(f"  Worst squares: {sorted_squares[-3:]}")
        
        print(f"\nDetailed results saved to: {probing_config.save_dir}")
        print("Run the analysis script to see visualizations:")
        print(f"python analyze_probing.py --results_path {probing_config.save_dir}/probing_results.json")
        
    except Exception as e:
        print(f"Error during probing: {e}")
        import traceback
        traceback.print_exc()
        
        # Show what we would expect to see
        print("\n" + "="*60)
        print("EXPECTED BEHAVIOR")
        print("="*60)
        print("With a trained model, you would typically see:")
        print("• Higher accuracy on center squares (e4, e5, d4, d5)")
        print("• Different patterns between layers")
        print("• Accuracy correlating with piece activity")
        print("• Better performance on frequently occupied squares")
        print("• Layer 1 often showing more chess-specific patterns than layer 0")


if __name__ == '__main__':
    main()