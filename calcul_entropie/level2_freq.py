import sys
import os
import math
from collections import Counter
from tqdm import tqdm

# Add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import ChessTokenizer

def calculate_freq_entropy():
    tokenizer = ChessTokenizer()
    pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
    
    if not os.path.exists(pgn_path):
        print(f"Error: File {pgn_path} not found.")
        return

    # Count character frequencies
    char_counts = Counter()
    total_chars = 0
    
    print(f"Reading {pgn_path}...")
    with open(pgn_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting chars"):
            # We should probably clean the text as the tokenizer does, 
            # or at least filter for valid chars to be consistent with the alphabet
            cleaned_line = tokenizer.clean_text(line)
            char_counts.update(cleaned_line)
            total_chars += len(cleaned_line)
            
    print(f"Total characters processed: {total_chars}")
    
    entropy = 0
    print("\nCharacter Frequencies:")
    for char, count in char_counts.most_common():
        prob = count / total_chars
        entropy -= prob * math.log2(prob)
        if prob > 0.01: # Print only common chars
             print(f"'{char}': {prob:.4f}")
             
    print(f"\nLevel 2 - Entropy with known frequency: {entropy:.4f} bits/char")
    return entropy

if __name__ == "__main__":
    calculate_freq_entropy()
