#!/usr/bin/env python3
"""
Quick test script to verify the probing system works.
"""

import sys
sys.path.append('.')

def test_board_utils():
    """Test board utilities."""
    print("Testing board utilities...")
    
    from board_utils import ChessBoardEncoder
    
    encoder = ChessBoardEncoder()
    
    # Test basic encoding
    test_games = [
        "1. e4 e5 2. Nf3 Nc6",
        "1. d4 d5 2. c4 e6"
    ]
    
    for game in test_games:
        positions = encoder.moves_string_to_positions(game)
        print(f"Game: {game}")
        print(f"  Extracted {len(positions)} positions")
        if positions:
            print(f"  First position shape: {positions[0].shape}")
            print(f"  Sample squares: {positions[0][:8]}")  # First rank
    
    print("✓ Board utilities working")


def test_model_loading():
    """Test model loading."""
    print("\nTesting model loading...")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        model_name = "Zual/chess_char"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        print(f"✓ Model loaded successfully")
        print(f"  Layers: {model.config.n_layer}")
        print(f"  Hidden size: {model.config.n_embd}")
        print(f"  Vocab size: {model.config.vocab_size}")
        
        # Test tokenization
        test_text = "1. e4 e5 2. Nf3"
        tokens = tokenizer.encode(test_text)
        print(f"  Test tokenization: '{test_text}' -> {len(tokens)} tokens")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    return True


def test_probing_setup():
    """Test probing setup."""
    print("\nTesting probing setup...")
    
    try:
        from probing import ProbingConfig, ChessProber
        
        config = ProbingConfig(
            hidden_dim=64,
            probe_layers=[0, 1],
            batch_size=4,
            num_epochs=1,
            device='cpu'
        )
        
        prober = ChessProber(config)
        print(f"✓ Prober created successfully")
        print(f"  Probing layers: {config.probe_layers}")
        print(f"  Total classifiers: {len(config.probe_layers) * 64}")
        
    except Exception as e:
        print(f"✗ Error setting up probing: {e}")
        return False
    
    return True


def main():
    print("="*50)
    print("CHESS PROBING SYSTEM TEST")
    print("="*50)
    
    # Test components
    test_board_utils()
    
    model_ok = test_model_loading()
    if not model_ok:
        print("\n⚠️  Model loading failed. Make sure you have:")
        print("   pip install transformers torch chess")
        print("   And internet connection for downloading the model")
        return
    
    prober_ok = test_probing_setup()
    if not prober_ok:
        print("\n⚠️  Probing setup failed. Check dependencies.")
        return
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)
    print("The probing system is ready to use.")
    print("\nTo run the full experiment:")
    print("  python run_probing_pretrained.py")
    print("\nTo run a quick demo:")
    print("  python demo_probing.py")


if __name__ == '__main__':
    main()