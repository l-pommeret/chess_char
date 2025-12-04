import sys
import os
import math
import chess.pgn
from tqdm import tqdm

def calculate_legal_entropy():
    pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
    
    if not os.path.exists(pgn_path):
        print(f"Error: File {pgn_path} not found.")
        return

    print(f"Reading {pgn_path}...")
    pgn = open(pgn_path)
    
    total_log_legal = 0
    total_chars = 0
    games_processed = 0
    
    # Process a subset of games to be feasible
    MAX_GAMES = 1000
    
    with tqdm(total=MAX_GAMES, desc="Processing games") as pbar:
        while games_processed < MAX_GAMES:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
                
            if game is None:
                break
                
            board = game.board()
            
            # Iterate through moves
            for move in game.mainline_moves():
                # Get number of legal moves in current position
                num_legal = board.legal_moves.count()
                
                # Calculate entropy contribution: log2(num_legal)
                # If num_legal is 0 (checkmate/stalemate), entropy is 0 for that step (no choice)
                if num_legal > 0:
                    total_log_legal += math.log2(num_legal)
                
                # Get SAN string length for character count
                # We need to be careful: SAN depends on context (disambiguation)
                # python-chess board.san(move) gives the standard algebraic notation
                san_move = board.san(move)
                
                # Add length of move string + 1 for space (or move number chars?)
                # The user wants entropy of the PGN file.
                # In PGN, a move is like "1. e4 e5"
                # The "1. " and " " are structural overhead.
                # If we consider "Entropy when we know legal moves", we are asking:
                # How many bits do I need to specify the next move, given I know it's a legal move?
                # That is log2(num_legal).
                # And this information is spread over len(san_move) characters (plus overhead).
                # So we sum log2(num_legal) and divide by the total characters used to represent these moves.
                
                # Let's approximate the character count as len(san_move) + 1 (for space)
                # We ignore move numbers for now as they are deterministic given the game state
                # (except for the formatting).
                # Actually, move numbers "1." are redundant if we know we are at start.
                # So maybe we should count the actual characters in the PGN string for this game?
                # But extracting exact string from python-chess game object is tricky without re-exporting.
                # Let's use len(san_move) + 1 (space).
                
                total_chars += len(san_move) + 1
                
                board.push(move)
            
            games_processed += 1
            pbar.update(1)
            
    if total_chars == 0:
        print("No moves processed.")
        return 0
        
    entropy = total_log_legal / total_chars
    print(f"\nProcessed {games_processed} games.")
    print(f"Total bits (sum log2 legal): {total_log_legal:.2f}")
    print(f"Total characters (approx): {total_chars}")
    print(f"Level 4 - Entropy with known legal moves: {entropy:.4f} bits/char")
    
    return entropy

if __name__ == "__main__":
    calculate_legal_entropy()
