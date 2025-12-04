import sys
import os
import math

# Add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import ChessTokenizer

def calculate_max_entropy():
    tokenizer = ChessTokenizer()
    vocab_size = len(tokenizer.valid_chars)
    
    # Max entropy is log2 of the alphabet size (assuming uniform distribution)
    max_entropy = math.log2(vocab_size)
    
    print(f"Alphabet size: {vocab_size}")
    print(f"Valid characters: {sorted(list(tokenizer.valid_chars))}")
    print(f"Level 1 - Max Entropy (Uniform): {max_entropy:.4f} bits/char")
    
    return max_entropy

if __name__ == "__main__":
    calculate_max_entropy()
