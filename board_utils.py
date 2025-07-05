"""
Utilities for chess board representation and PGN parsing for probing.
"""

import chess
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional, Dict
import io

class ChessBoardEncoder:
    """Encodes chess positions into numerical representations for probing."""
    
    # Mapping from piece symbols to indices
    PIECE_TO_IDX = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
        '.': 12  # Empty square
    }
    
    IDX_TO_PIECE = {v: k for k, v in PIECE_TO_IDX.items()}
    
    def __init__(self):
        self.num_classes = 13  # 12 pieces + empty
        
    def board_to_array(self, board: chess.Board) -> np.ndarray:
        """Convert chess board to 64-element array with piece indices."""
        position = np.zeros(64, dtype=np.int32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                position[square] = self.PIECE_TO_IDX['.']
            else:
                position[square] = self.PIECE_TO_IDX[piece.symbol()]
                
        return position
    
    def fen_to_array(self, fen: str) -> np.ndarray:
        """Convert FEN string to 64-element array."""
        board = chess.Board(fen)
        return self.board_to_array(board)
    
    def pgn_to_positions(self, pgn_string: str) -> List[Tuple[np.ndarray, str]]:
        """Extract all positions from a PGN game with their move context."""
        positions = []
        
        try:
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                return positions
            
            board = game.board()
            
            # Initial position
            positions.append((self.board_to_array(board), "initial"))
            
            # Play through all moves
            for move_num, move in enumerate(game.mainline_moves()):
                board.push(move)
                move_context = f"move_{move_num + 1}"
                positions.append((self.board_to_array(board), move_context))
                
        except Exception as e:
            print(f"Error parsing PGN: {e}")
            return []
            
        return positions
    
    def moves_string_to_positions(self, moves_string: str) -> List[np.ndarray]:
        """Convert a moves string like '1. e4 e5 2. Nf3' to board positions."""
        positions = []
        
        try:
            # Create a minimal PGN
            pgn_string = f"[Event \"Analysis\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n{moves_string} *"
            
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                return positions
            
            board = game.board()
            positions.append(self.board_to_array(board))
            
            for move in game.mainline_moves():
                board.push(move)
                positions.append(self.board_to_array(board))
                
        except Exception as e:
            print(f"Error parsing moves: {e}")
            return []
            
        return positions
    
    def get_square_name(self, square_idx: int) -> str:
        """Get square name (e.g., 'a1', 'h8') from square index."""
        return chess.square_name(square_idx)
    
    def get_piece_name(self, piece_idx: int) -> str:
        """Get piece symbol from piece index."""
        return self.IDX_TO_PIECE.get(piece_idx, '?')


class PGNPositionExtractor:
    """Extract positions from PGN files for probing dataset creation."""
    
    def __init__(self):
        self.encoder = ChessBoardEncoder()
    
    def extract_from_pgn_file(self, file_path: str, max_games: Optional[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Extract positions from a PGN file with metadata."""
        positions = []
        games_processed = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    game_positions = self.encoder.pgn_to_positions(str(game))
                    
                    # Add metadata
                    metadata = {
                        'game_id': games_processed,
                        'white_elo': game.headers.get('WhiteElo', 'Unknown'),
                        'black_elo': game.headers.get('BlackElo', 'Unknown'),
                        'event': game.headers.get('Event', 'Unknown'),
                        'result': game.headers.get('Result', 'Unknown')
                    }
                    
                    for pos, move_context in game_positions:
                        positions.append((pos, {**metadata, 'move_context': move_context}))
                    
                    games_processed += 1
                    
                    if max_games and games_processed >= max_games:
                        break
                        
                    if games_processed % 1000 == 0:
                        print(f"Processed {games_processed} games, {len(positions)} positions")
                        
        except Exception as e:
            print(f"Error reading PGN file: {e}")
            
        return positions
    
    def extract_from_text_games(self, text_games: List[str]) -> List[np.ndarray]:
        """Extract positions from a list of text games (move sequences)."""
        all_positions = []
        
        for game_text in text_games:
            positions = self.encoder.moves_string_to_positions(game_text)
            all_positions.extend(positions)
            
        return all_positions