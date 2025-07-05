#!/usr/bin/env python3
"""
Script for running chess model probing experiments.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import torch

from model import ChessGPT
from tokenizer import ChessTokenizer
from config import ModelConfig
from probing import run_probing_experiment, ProbingConfig
from dataset import ChessDataset


def load_chess_model(model_path: str = None, config_path: str = None) -> ChessGPT:
    """Load a trained chess model."""
    
    # If no model_path provided, use Hugging Face model
    if model_path is None or model_path == "auto":
        print("Using Hugging Face model: Zual/chess_char")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        class HFChessModel:
            def __init__(self):
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    self.tokenizer = AutoTokenizer.from_pretrained("Zual/chess_char")
                    self.model = AutoModelForCausalLM.from_pretrained("Zual/chess_char")
                except:
                    # Fallback to GPT2
                    self.tokenizer = GPT2Tokenizer.from_pretrained("Zual/chess_char", use_fast=False)
                    self.model = GPT2LMHeadModel.from_pretrained("Zual/chess_char")
                
                # Add pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Add encode/decode methods for compatibility
                self.encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
                self.decode = lambda tokens: self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return HFChessModel()
    
    # Original logic for local model files
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = ChessTokenizer()
    
    # Load or create model config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            model_config = ModelConfig(**config_dict)
    else:
        # Use default config
        model_config = ModelConfig(vocab_size=len(tokenizer.vocab))
    
    # Initialize model
    chess_model = ChessGPT(model_config, tokenizer)
    
    # Load trained weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        chess_model.model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    else:
        print("WARNING: Model weights not found. Using random initialization.")
    
    return chess_model


def load_games_from_dataset(dataset_path: str, max_games: int = 1000) -> List[str]:
    """Load games from the training dataset."""
    print(f"Loading games from {dataset_path} (max {max_games} games)")
    
    games = []
    
    try:
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                for game_data in data[:max_games]:
                    if isinstance(game_data, dict) and 'moves' in game_data:
                        games.append(game_data['moves'])
                    elif isinstance(game_data, str):
                        games.append(game_data)
        else:
            # Try to load as text file
            with open(dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_games:
                        break
                    line = line.strip()
                    if line:
                        games.append(line)
    
    except Exception as e:
        print(f"Error loading games: {e}")
        # Create some dummy games for testing
        games = [
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
            "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
            "1. e4 c5 2. Nf3 d6 3. d4 cxd4",
            "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7",
            "1. c4 e5 2. Nc3 Nf6 3. g3 d5"
        ]
        print(f"Using {len(games)} dummy games for testing")
    
    print(f"Loaded {len(games)} games")
    return games


def extract_representations_hf(model, text_games: List[str], probe_layers: List[int], device: str) -> Dict[str, List[Tuple[torch.Tensor, np.ndarray]]]:
    """Extract representations from Hugging Face model."""
    from board_utils import ChessBoardEncoder
    
    model.model.eval()
    representations = {}
    
    # Initialize storage for each layer
    for layer in probe_layers:
        representations[f'layer_{layer}'] = []
    
    encoder = ChessBoardEncoder()
    
    with torch.no_grad():
        for game_text in text_games:
            print(f"Processing: {game_text[:50]}...")
            
            # Get board positions
            positions = encoder.moves_string_to_positions(game_text)
            if not positions:
                continue
            
            try:
                # Tokenize with proper handling
                if hasattr(model, 'encode'):
                    # Custom tokenizer
                    input_ids = torch.tensor([model.encode(game_text)]).to(device)
                else:
                    # Hugging Face tokenizer
                    inputs = model.tokenizer(game_text, return_tensors="pt", 
                                           padding=True, truncation=True, max_length=512)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs.get("attention_mask", None)
                
                # Get hidden states
                if attention_mask is not None:
                    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, 
                                        output_hidden_states=True)
                    seq_len = attention_mask.sum().item()
                    last_token_idx = seq_len - 1
                else:
                    outputs = model.model(input_ids, output_hidden_states=True)
                    last_token_idx = -1
                
                hidden_states = outputs.hidden_states
                
                # Extract for each layer
                for layer_idx in probe_layers:
                    if layer_idx < len(hidden_states) - 1:  # -1 because first is embeddings
                        layer_hidden = hidden_states[layer_idx + 1]
                        game_repr = layer_hidden[0, last_token_idx, :].cpu()
                        
                        for position in positions:
                            representations[f'layer_{layer_idx}'].append((game_repr, position))
                
            except Exception as e:
                print(f"Error processing game: {e}")
                continue
    
    return representations


def main():
    parser = argparse.ArgumentParser(description='Run chess model probing experiments')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default="auto",
                       help='Path to trained model weights (default: auto - uses Hugging Face model)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config JSON file')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset file (JSON or text)')
    parser.add_argument('--max_games', type=int, default=50,
                       help='Maximum number of games to use')
    
    # Probing arguments
    parser.add_argument('--probe_layers', type=int, nargs='+', default=None,
                       help='Which layers to probe (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training probes')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for probe training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./probing_results',
                       help='Directory to save results')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained probe models')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    chess_model = load_chess_model(args.model_path, args.config_path)
    chess_model.model.to(device)
    
    # Auto-detect layers if not specified
    if args.probe_layers is None:
        if hasattr(chess_model.model.config, 'n_layer'):
            num_layers = chess_model.model.config.n_layer
        else:
            num_layers = 2  # Default fallback
        args.probe_layers = list(range(num_layers))
    
    print(f"Will probe layers: {args.probe_layers}")
    
    # Load games
    if args.dataset_path:
        games = load_games_from_dataset(args.dataset_path, args.max_games)
    else:
        # Extended test games
        games = [
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
        print(f"Using {len(games)} test games")
    
    # Get hidden dimension
    if hasattr(chess_model.model.config, 'n_embd'):
        hidden_dim = chess_model.model.config.n_embd
    elif hasattr(chess_model.model.config, 'hidden_size'):
        hidden_dim = chess_model.model.config.hidden_size
    else:
        hidden_dim = 64  # Default fallback
    
    print(f"Hidden dimension: {hidden_dim}")
    
    # Create probing config
    probing_config = ProbingConfig(
        hidden_dim=hidden_dim,
        probe_layers=args.probe_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.output_dir
    )
    
    # Extract representations
    print("\n" + "="*50)
    print("EXTRACTING REPRESENTATIONS")
    print("="*50)
    
    representations = extract_representations_hf(chess_model, games, args.probe_layers, device)
    
    # Check if we got any data
    total_samples = sum(len(layer_reprs) for layer_reprs in representations.values())
    if total_samples == 0:
        print("ERROR: No representations extracted. Check your games and model.")
        return
    
    print(f"Extracted {total_samples} total samples")
    for layer_key, layer_reprs in representations.items():
        print(f"  {layer_key}: {len(layer_reprs)} samples")
    
    # Train and evaluate probes manually
    print("\n" + "="*50)
    print("TRAINING PROBES")
    print("="*50)
    
    from probing import ChessProber
    
    # Create prober and train
    prober = ChessProber(probing_config)
    
    # Split data
    train_representations = {}
    test_representations = {}
    
    for layer_key, layer_reprs in representations.items():
        split_idx = int(0.8 * len(layer_reprs))
        train_representations[layer_key] = layer_reprs[:split_idx]
        test_representations[layer_key] = layer_reprs[split_idx:]
    
    # Train
    prober.train_probes(train_representations)
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATING PROBES")
    print("="*50)
    
    results = prober.evaluate_probes(test_representations)
    
    # Print final summary
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    
    for layer_key, layer_results in results.items():
        accuracies = list(layer_results.values())
        print(f"\n{layer_key}:")
        print(f"  Mean accuracy: {np.mean(accuracies):.3f}")
        print(f"  Std accuracy:  {np.std(accuracies):.3f}")
        print(f"  Min accuracy:  {np.min(accuracies):.3f}")
        print(f"  Max accuracy:  {np.max(accuracies):.3f}")
    
    # Save results
    prober.visualize_results(results)
    prober.save_results(results)
    if args.save_models:
        prober.save_models()
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Run: python analyze_probing.py --results_path {args.output_dir}/probing_results.json")


if __name__ == '__main__':
    import chess  # Add this import for the chess library
    main()