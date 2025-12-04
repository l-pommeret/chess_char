import sys
import os
import math
from collections import Counter, defaultdict
from tqdm import tqdm

# Add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import ChessTokenizer

def get_context(char, prev_char, state):
    """
    Determine the context based on previous character and current state.
    State can be: 'TAG', 'MOVES', 'COMMENT'
    """
    if state == 'TAG':
        if char == ']':
            return 'TAG_END', 'MOVES' # Transition to MOVES (potentially)
        return f"TAG_{prev_char}", 'TAG'
    
    if state == 'COMMENT':
        if char == '}':
            return 'COMMENT_END', 'MOVES'
        return 'COMMENT_INSIDE', 'COMMENT'

    # State is MOVES
    if char == '[':
        return 'TAG_START', 'TAG'
    if char == '{':
        return 'COMMENT_START', 'COMMENT'
    
    # Specific move syntax contexts
    if prev_char == '.':
        return 'AFTER_DOT', 'MOVES' # Expect space
    if prev_char.isdigit():
        return 'AFTER_DIGIT', 'MOVES' # Expect digit or dot
    if prev_char in 'NBRQK':
        return 'AFTER_PIECE', 'MOVES' # Expect file or x
    if prev_char in 'abcdefgh':
        return 'AFTER_FILE', 'MOVES' # Expect rank or x
    if prev_char in '12345678':
        return 'AFTER_RANK', 'MOVES' # Expect space, +, #
    if prev_char == ' ':
        return 'AFTER_SPACE', 'MOVES' # Expect move number or piece
    if prev_char == 'x':
        return 'AFTER_CAPTURE', 'MOVES'
    if prev_char == '+':
        return 'AFTER_CHECK', 'MOVES'
    
    return 'DEFAULT', 'MOVES'

def calculate_syntax_entropy():
    tokenizer = ChessTokenizer()
    pgn_path = "../chess_data/lichess_db_standard_rated_2013-09.pgn"
    
    if not os.path.exists(pgn_path):
        print(f"Error: File {pgn_path} not found.")
        return

    # Counts: context -> char -> count
    context_counts = defaultdict(Counter)
    total_chars = 0
    
    print(f"Reading {pgn_path}...")
    
    # We need to process character by character to maintain state
    # Reading line by line is easier for file handling
    
    state = 'MOVES' # Start assuming moves or tags (handled by [ check)
    prev_char = '\n' # Dummy start
    
    with open(pgn_path, 'r', encoding='utf-8') as f:
        # Read a subset to be faster? Or full file?
        # Let's read the first 1M lines or so to get a good estimate without taking forever
        # Or just use tqdm on lines
        for i, line in tqdm(enumerate(f), desc="Processing syntax"):
            cleaned_line = tokenizer.clean_text(line)
            # Add newline as explicit character if we want to model line breaks, 
            # but tokenizer usually strips them. 
            # Let's stick to what tokenizer produces but maybe add a space if it joins lines?
            # The tokenizer.clean_text replaces \n with space or removes it.
            # Let's process the cleaned line.
            
            for char in cleaned_line:
                context, new_state = get_context(char, prev_char, state)
                context_counts[context][char] += 1
                state = new_state
                prev_char = char
                total_chars += 1
                
            # Add a space between lines effectively
            if not cleaned_line.endswith(' '):
                 context, new_state = get_context(' ', prev_char, state)
                 context_counts[context][' '] += 1
                 state = new_state
                 prev_char = ' '
                 total_chars += 1

            if i > 100000: # Limit to 100k lines for speed
                break
                
    print(f"Total characters processed: {total_chars}")
    
    total_entropy = 0
    
    # H(X|C) = sum P(c) * H(X|c)
    # P(c) = count(c) / total_chars
    
    print("\nContext Entropies:")
    for context, counts in context_counts.items():
        context_total = sum(counts.values())
        context_prob = context_total / total_chars
        
        context_entropy = 0
        for char, count in counts.items():
            prob = count / context_total
            context_entropy -= prob * math.log2(prob)
            
        print(f"Context '{context}': p={context_prob:.4f}, H={context_entropy:.4f} bits")
        total_entropy += context_prob * context_entropy
        
    print(f"\nLevel 3 - Entropy with known syntax: {total_entropy:.4f} bits/char")
    return total_entropy

if __name__ == "__main__":
    calculate_syntax_entropy()
