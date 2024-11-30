import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

@dataclass
class ChessGame:
    """Stocke les informations d'une partie d'échecs."""
    moves: str
    white_elo: int
    black_elo: int
    time_control: float
    num_moves: int

class PGNParser:
    """Parse et filtre les fichiers PGN avec logging détaillé."""
    
    def __init__(self, tokenizer, min_elo: int = 1500,  # Elo min
                 min_moves: int = 10,    
                 max_moves: int = 300, 
                 min_time_control: float = 3.0): 
        self.tokenizer = tokenizer
        self.min_elo = min_elo
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.min_time_control = min_time_control
        self.stats = {
            'total_games': 0,
            'elo_filtered': 0,
            'moves_filtered': 0,
            'time_filtered': 0,
            'invalid_format': 0,
            'valid_games': 0
        }

    def clean_pgn(self, text: str) -> str:
        """Nettoie le texte PGN de manière plus permissive."""
        # Supprime les commentaires et variantes
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        # Optimise les espaces mais garde la lisibilité
        text = re.sub(r'(\d+\.)\s*', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def parse_time_control(line: str) -> Optional[float]:
        """Parse le contrôle du temps avec plus de formats acceptés."""
        # Format standard: "TimeControl "300+0""
        if match := re.search(r'\[TimeControl "(\d+)(?:\+\d+)?"\]', line):
            return int(match.group(1)) / 60
        return None

    @staticmethod
    def parse_elo(line: str) -> int:
        """Parse l'Elo avec valeur par défaut plus basse."""
        if match := re.search(r'\[(\w+)Elo "(\d+)"\]', line):
            return int(match.group(2))
        return 1500  # Valeur par défaut plus permissive

    def is_valid_game(self, game: ChessGame) -> bool:
        """Vérifie si une partie répond aux critères avec logging."""
        if game.white_elo < self.min_elo or game.black_elo < self.min_elo:
            self.stats['elo_filtered'] += 1
            return False
            
        if not (self.min_moves <= game.num_moves <= self.max_moves):
            self.stats['moves_filtered'] += 1
            return False
            
        if game.time_control < self.min_time_control:
            self.stats['time_filtered'] += 1
            return False
            
        return True

    def parse_file(self, filepath: str, max_games: Optional[int] = None) -> Tuple[List[str], int, int]:
        """Parse un fichier PGN avec logging détaillé."""
        games = []
        current_game = []
        current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)
        in_game = False

        print(f"Critères de filtrage:")
        print(f"- Elo minimum: {self.min_elo}")
        print(f"- Nombre de coups: {self.min_moves}-{self.max_moves}")
        print(f"- Temps de contrôle minimum: {self.min_time_control} minutes")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing PGN"):
                line = line.strip()
                
                if line.startswith('['):
                    if not in_game:
                        in_game = True
                    if "WhiteElo" in line:
                        current.white_elo = self.parse_elo(line)
                    elif "BlackElo" in line:
                        current.black_elo = self.parse_elo(line)
                    elif "TimeControl" in line:
                        current.time_control = self.parse_time_control(line) or 0
                elif line:
                    current_game.append(line)
                    current.num_moves += line.count('.')
                elif in_game and current_game:  # Fin d'une partie
                    self.stats['total_games'] += 1
                    
                    if self.is_valid_game(current):
                        clean_game = self.clean_pgn(' '.join(current_game))
                        try:
                            if self.tokenizer.is_valid_text(clean_game):
                                games.append(clean_game)
                                self.stats['valid_games'] += 1
                                if max_games and len(games) >= max_games:
                                    break
                            else:
                                self.stats['invalid_format'] += 1
                        except ValueError:
                            self.stats['invalid_format'] += 1
                    
                    current_game = []
                    current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)
                    in_game = False

        print("\nStatistiques du parsing:")
        print(f"Parties totales: {self.stats['total_games']}")
        print(f"Filtrées par Elo: {self.stats['elo_filtered']}")
        print(f"Filtrées par nombre de coups: {self.stats['moves_filtered']}")
        print(f"Filtrées par temps de contrôle: {self.stats['time_filtered']}")
        print(f"Format invalide: {self.stats['invalid_format']}")
        print(f"Parties valides: {self.stats['valid_games']}")

        return games, self.stats['valid_games'], self.stats['total_games']