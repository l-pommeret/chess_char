"""
Probing module for chess model analysis.
Trains 64 classifiers to predict board state from model representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import chess

from model import ChessGPT
from board_utils import ChessBoardEncoder, PGNPositionExtractor


@dataclass
class ProbingConfig:
    """Configuration for probing experiments."""
    hidden_dim: int = 64  # Model hidden dimension
    num_squares: int = 64  # Chess board squares
    num_classes: int = 13  # 12 pieces + empty
    probe_layers: List[int] = None  # Which layers to probe (None = all)
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./probing_results"
    
    def __post_init__(self):
        if self.probe_layers is None:
            self.probe_layers = [0, 1]  # Default to all layers for 2-layer model


class SquareClassifier(nn.Module):
    """Individual classifier for one chess square."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 13):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ChessProber:
    """Main probing class that manages 64 square classifiers."""
    
    def __init__(self, config: ProbingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.encoder = ChessBoardEncoder()
        
        # Create 64 classifiers (one per square)
        self.classifiers = {}
        for layer in config.probe_layers:
            self.classifiers[f'layer_{layer}'] = {}
            for square in range(64):
                classifier = SquareClassifier(config.hidden_dim, config.num_classes)
                self.classifiers[f'layer_{layer}'][square] = classifier.to(self.device)
        
        # Initialize optimizers
        self.optimizers = {}
        for layer_key, layer_classifiers in self.classifiers.items():
            self.optimizers[layer_key] = {}
            for square, classifier in layer_classifiers.items():
                self.optimizers[layer_key][square] = torch.optim.Adam(
                    classifier.parameters(), 
                    lr=config.learning_rate
                )
    
    def extract_representations(self, model: ChessGPT, text_games: List[str]) -> Dict[str, List[Tuple[torch.Tensor, np.ndarray]]]:
        """Extract hidden representations and corresponding board states."""
        model.model.eval()
        
        representations = {}
        for layer in self.config.probe_layers:
            representations[f'layer_{layer}'] = []
        
        with torch.no_grad():
            for game_text in tqdm(text_games, desc="Extracting representations"):
                # Get board positions for this game
                positions = self.encoder.moves_string_to_positions(game_text)
                
                if not positions:
                    continue
                
                # Tokenize the game
                try:
                    input_ids = torch.tensor([model.tokenizer.encode(game_text)]).to(self.device)
                    
                    # Get hidden states from model
                    outputs = model.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states  # (layer, batch, seq_len, hidden_dim)
                    
                    # Extract representations for each layer
                    for layer_idx in self.config.probe_layers:
                        layer_hidden = hidden_states[layer_idx + 1]  # +1 because layer 0 is embeddings
                        
                        # Use the last token representation as the game representation
                        game_repr = layer_hidden[0, -1, :].cpu()  # (hidden_dim,)
                        
                        # Associate with each position in the game
                        for position in positions:
                            representations[f'layer_{layer_idx}'].append((game_repr, position))
                            
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue
        
        return representations
    
    def train_probes(self, representations: Dict[str, List[Tuple[torch.Tensor, np.ndarray]]]):
        """Train all 64 classifiers for each layer."""
        
        for layer_key, layer_reprs in representations.items():
            print(f"\nTraining probes for {layer_key}")
            
            # Prepare data for this layer
            X = torch.stack([repr_tensor for repr_tensor, _ in layer_reprs])  # (N, hidden_dim)
            y = np.array([board_state for _, board_state in layer_reprs])  # (N, 64)
            
            # Train each square classifier
            for square in range(64):
                classifier = self.classifiers[layer_key][square]
                optimizer = self.optimizers[layer_key][square]
                
                square_labels = torch.tensor(y[:, square], dtype=torch.long).to(self.device)
                
                # Training loop for this square
                classifier.train()
                for epoch in range(self.config.num_epochs):
                    total_loss = 0
                    correct = 0
                    total = 0
                    
                    # Mini-batch training
                    for i in range(0, len(X), self.config.batch_size):
                        batch_X = X[i:i+self.config.batch_size].to(self.device)
                        batch_y = square_labels[i:i+self.config.batch_size]
                        
                        optimizer.zero_grad()
                        outputs = classifier(batch_X)
                        loss = F.cross_entropy(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    
                    if epoch % 5 == 0:
                        accuracy = 100 * correct / total
                        print(f"  Square {self.encoder.get_square_name(square)} - Epoch {epoch}, Loss: {total_loss:.4f}, Acc: {accuracy:.2f}%")
    
    def evaluate_probes(self, representations: Dict[str, List[Tuple[torch.Tensor, np.ndarray]]]) -> Dict[str, Dict[str, float]]:
        """Evaluate all probes and return accuracy metrics."""
        results = {}
        
        for layer_key, layer_reprs in representations.items():
            print(f"\nEvaluating probes for {layer_key}")
            
            X = torch.stack([repr_tensor for repr_tensor, _ in layer_reprs])
            y = np.array([board_state for _, board_state in layer_reprs])
            
            layer_results = {}
            
            for square in range(64):
                classifier = self.classifiers[layer_key][square]
                classifier.eval()
                
                square_labels = torch.tensor(y[:, square], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    outputs = classifier(X.to(self.device))
                    _, predicted = torch.max(outputs, 1)
                    
                    accuracy = accuracy_score(square_labels.cpu().numpy(), predicted.cpu().numpy())
                    square_name = self.encoder.get_square_name(square)
                    layer_results[square_name] = accuracy
                    
            results[layer_key] = layer_results
            
            # Print summary
            avg_accuracy = np.mean(list(layer_results.values()))
            print(f"  Average accuracy for {layer_key}: {avg_accuracy:.3f}")
            
        return results
    
    def visualize_results(self, results: Dict[str, Dict[str, float]]):
        """Create visualizations of probing results."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        for layer_key, layer_results in results.items():
            # Create 8x8 heatmap of accuracies
            accuracy_matrix = np.zeros((8, 8))
            
            for square_name, accuracy in layer_results.items():
                square_idx = chess.parse_square(square_name)
                row = square_idx // 8
                col = square_idx % 8
                accuracy_matrix[7-row, col] = accuracy  # Flip row for chess board orientation
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(accuracy_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       xticklabels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                       yticklabels=['8', '7', '6', '5', '4', '3', '2', '1'],
                       vmin=0, vmax=1)
            
            plt.title(f'Probing Accuracy Heatmap - {layer_key}')
            plt.xlabel('File')
            plt.ylabel('Rank')
            
            save_path = os.path.join(self.config.save_dir, f'heatmap_{layer_key}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved heatmap for {layer_key} to {save_path}")
    
    def save_results(self, results: Dict[str, Dict[str, float]]):
        """Save probing results to JSON."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(self.config.save_dir, 'probing_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary statistics
        summary = {}
        for layer_key, layer_results in results.items():
            accuracies = list(layer_results.values())
            summary[layer_key] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'median_accuracy': np.median(accuracies)
            }
        
        summary_path = os.path.join(self.config.save_dir, 'summary_stats.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {results_path}")
        print(f"Summary statistics saved to {summary_path}")
    
    def save_models(self):
        """Save trained classifier models."""
        models_dir = os.path.join(self.config.save_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for layer_key, layer_classifiers in self.classifiers.items():
            layer_dir = os.path.join(models_dir, layer_key)
            os.makedirs(layer_dir, exist_ok=True)
            
            for square, classifier in layer_classifiers.items():
                square_name = self.encoder.get_square_name(square)
                model_path = os.path.join(layer_dir, f'probe_{square_name}.pth')
                torch.save(classifier.state_dict(), model_path)
        
        print(f"Trained models saved to {models_dir}")


def run_probing_experiment(chess_model: ChessGPT, 
                          text_games: List[str], 
                          config: ProbingConfig) -> Dict[str, Dict[str, float]]:
    """Run a complete probing experiment."""
    
    print(f"Starting probing experiment with {len(text_games)} games")
    print(f"Probing layers: {config.probe_layers}")
    print(f"Using device: {config.device}")
    
    # Initialize prober
    prober = ChessProber(config)
    
    # Extract representations
    print("\n=== Extracting Representations ===")
    representations = prober.extract_representations(chess_model, text_games)
    
    # Print dataset statistics
    for layer_key, layer_reprs in representations.items():
        print(f"{layer_key}: {len(layer_reprs)} position-representation pairs")
    
    # Split data into train/test
    train_representations = {}
    test_representations = {}
    
    for layer_key, layer_reprs in representations.items():
        split_idx = int(0.8 * len(layer_reprs))
        train_representations[layer_key] = layer_reprs[:split_idx]
        test_representations[layer_key] = layer_reprs[split_idx:]
    
    # Train probes
    print("\n=== Training Probes ===")
    prober.train_probes(train_representations)
    
    # Evaluate probes
    print("\n=== Evaluating Probes ===")
    results = prober.evaluate_probes(test_representations)
    
    # Visualize and save results
    print("\n=== Saving Results ===")
    prober.visualize_results(results)
    prober.save_results(results)
    prober.save_models()
    
    return results