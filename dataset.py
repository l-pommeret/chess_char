import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List
from tqdm import tqdm

from tokenizer import ChessTokenizer

class ChessDataset(Dataset):
    """Dataset PyTorch pour les parties d'échecs."""
    
    def __init__(self, games: List[str], tokenizer: ChessTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.games = self._validate_games(games)

    def _validate_games(self, games: List[str]) -> List[str]:
        """Vérifie que chaque partie peut être tokenizée."""
        valid_games = []
        for game in tqdm(games, desc="Validation des parties"):
            try:
                self.tokenizer.encode(game)
                valid_games.append(game)
            except ValueError:
                continue
        print(f"Parties valides : {len(valid_games)}/{len(games)}")
        return valid_games

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retourne une partie encodée avec padding."""
        game = self.games[idx]
        encoded = self.tokenizer.encode(game)
        
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded += [self.tokenizer.vocab['<PAD>']] * (self.max_length - len(encoded))
            
        return {
            'input_ids': torch.tensor(encoded),
            'labels': torch.tensor(encoded)
        }

def create_datasets(games: List[str], tokenizer: ChessTokenizer,
                   max_length: int, test_ratio: float = 0.0005) -> Tuple[Dataset, Dataset]:
    """Crée les datasets d'entraînement et de test."""
    test_size = int(test_ratio * len(games))
    train_size = len(games) - test_size
    
    train_data = ChessDataset(games[:train_size], tokenizer, max_length)
    test_data = ChessDataset(games[train_size:], tokenizer, max_length)
    
    return train_data, test_data