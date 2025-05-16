import torch
import os
import argparse
import glob
from model import ChessGPT
from tokenizer import ChessTokenizer, TokenizerConfig
from config import ModelConfig, GenerationConfig

def main():
    # Configurer les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Inférence avec un modèle ChessGPT")
    parser.add_argument("--model", type=str, default="./models/gpt2-chess",
                        help="Chemin vers le dossier du modèle")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Nom spécifique du checkpoint à utiliser (ex: 'checkpoint-500')")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Chemin vers le fichier tokenizer.json")
    parser.add_argument("--prompt", type=str, default="1.e4 e5 2.Nf3 ",
                        help="Prompt de départ pour la génération")
    args = parser.parse_args()
    
    # Initialiser le tokenizer
    print("Initialisation du tokenizer...")
    tokenizer = ChessTokenizer()
    
    # Déterminer le chemin du tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    else:
        tokenizer_path = os.path.join(args.model, "tokenizer/tokenizer.json")
    
    print(f"Chargement du tokenizer depuis {tokenizer_path}...")
    
    try:
        # Charger directement le fichier tokenizer.json sans utiliser la méthode load
        import json
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        # Configurer manuellement le tokenizer
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id2token = {int(k): v for k, v in tokenizer_data['id2token'].items()}
        tokenizer.valid_chars = set(tokenizer_data['valid_chars'])
    except Exception as e:
        print(f"Erreur lors du chargement du tokenizer: {e}")
        return
    
    # Configurer et créer le modèle
    print("Configuration du modèle...")
    model_config = ModelConfig(vocab_size=len(tokenizer))
    chess_gpt = ChessGPT(model_config, tokenizer)
    
    # Détecter le meilleur périphérique disponible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilisation du GPU CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Utilisation du GPU MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Utilisation du CPU")
    
    # Déterminer le chemin du modèle
    model_path = args.model
    
    # Si un checkpoint spécifique est demandé
    if args.checkpoint:
        # Si c'est un nom de dossier complet
        if os.path.isdir(os.path.join(model_path, args.checkpoint)):
            checkpoint_path = os.path.join(model_path, args.checkpoint)
        else:
            # Si c'est juste un nom de checkpoint comme "checkpoint-500"
            matching_dirs = glob.glob(os.path.join(model_path, f"{args.checkpoint}*"))
            if matching_dirs:
                checkpoint_path = matching_dirs[0]
            else:
                print(f"Aucun checkpoint correspondant à '{args.checkpoint}' trouvé dans {model_path}")
                return
        model_path = checkpoint_path
    
    print(f"Chargement du modèle depuis {model_path}...")
    
    try:
        # Cette partie dépend de comment le modèle a été sauvegardé
        # Si le modèle a été sauvegardé avec transformers, utiliser:
        from transformers import GPT2LMHeadModel
        chess_gpt.model = GPT2LMHeadModel.from_pretrained(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Essai d'une méthode alternative de chargement...")
        
        try:
            # Si le modèle a été sauvegardé avec torch.save
            checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(checkpoint_file):
                chess_gpt.model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            else:
                # Chercher d'autres fichiers de modèle
                model_files = glob.glob(os.path.join(model_path, "*.bin")) + glob.glob(os.path.join(model_path, "*.pt"))
                if model_files:
                    print(f"Tentative de chargement du fichier {model_files[0]}...")
                    chess_gpt.model.load_state_dict(torch.load(model_files[0], map_location=device))
                else:
                    print(f"Aucun fichier de modèle trouvé dans {model_path}")
                    return
        except Exception as e:
            print(f"Échec du chargement du modèle: {e}")
            return
    
    # Déplacer le modèle sur le périphérique approprié
    chess_gpt.model.to(device)
    chess_gpt.model.eval()
    
    # Configuration pour la génération
    gen_config = GenerationConfig()
    
    # Générer une continuation pour un prompt donné
    prompt = args.prompt
    print(f"\nPrompt: {prompt}")
    
    try:
        with torch.no_grad():
            generated = chess_gpt.generate(prompt, gen_config)
            print(f"\nPartie générée:\n{generated}")
    except Exception as e:
        print(f"Erreur lors de la génération: {e}")

if __name__ == "__main__":
    main()