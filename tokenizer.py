import re
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import json
import os
from tqdm import tqdm

@dataclass
class TokenizerConfig:
    pad_token: str = '<PAD>'
    initial_tokens: List[str] = None
    special_tokens: List[str] = None
    save_path: Optional[str] = None

class ChessTokenizer:
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {self.config.pad_token: 0}
        self.id2token: Dict[int, str] = {0: self.config.pad_token}
        self.token_id: int = 1
        self.valid_chars: Set[str] = set()
        self._initialize_vocab()
        print(f"Taille du vocabulaire: {len(self.vocab)}")
        print(f"Nombre de caractères valides: {len(self.valid_chars)}")

    def _initialize_vocab(self) -> None:
        """Initialise le vocabulaire avec tous les tokens nécessaires."""
        # 1. Squares (e4, a1, etc.)
        for file in 'abcdefgh':
            for rank in '12345678':
                square = file + rank
                self._add_token(square)
                # Ajout des caractères individuels au vocabulaire
                self._add_token(file)
                self._add_token(rank)
                self.valid_chars.add(file)
                self.valid_chars.add(rank)

        # 2. Pièces et symboles de base
        basic_chars = set('RNBQKP')  # Pièces
        self.valid_chars.update(basic_chars)
        for char in basic_chars:
            self._add_token(char)

        # 3. Symboles de notation
        notation_chars = set('x+#=O-0123456789.')
        self.valid_chars.update(notation_chars)
        for char in notation_chars:
            self._add_token(char)

        # 4. Caractères de formatage
        format_chars = set(' \n\t(),[]"\'/!')
        self.valid_chars.update(format_chars)
        for char in format_chars:
            self._add_token(char)

        # 5. Tokens spéciaux
        if self.config.special_tokens:
            for token in self.config.special_tokens:
                self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token not in self.vocab:
            self.vocab[token] = self.token_id
            self.id2token[self.token_id] = token
            self.token_id += 1

    def clean_text(self, text: str) -> str:
        # Normalisation des caractères spéciaux
        replacements = {
            "'": "'",
            """: '"',
            """: '"',
            "–": "-",
            "—": "-",
            "…": "...",
            "\r": "\n",
            "\u2026": "..."
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Suppression des caractères invalides
        cleaned = ''.join(char for char in text if char in self.valid_chars)
        
        # Normalisation des espaces
        return ' '.join(cleaned.split())

    def tokenize(self, text: str) -> List[str]:
        """Tokenize le texte avec gestion des cases d'échecs et des caractères individuels."""
        text = self.clean_text(text)
        tokens = []
        i = 0
        while i < len(text):
            # Vérifier si c'est une case d'échecs (e.g., e4, a1)
            if i + 1 < len(text) and text[i] in 'abcdefgh' and text[i+1] in '12345678':
                tokens.append(text[i:i+2])
                i += 2
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def is_valid_text(self, text: str) -> bool:
        text = self.clean_text(text)
        
        # Vérifier les caractères invalides
        invalid_chars = set(char for char in text if char not in self.valid_chars)
        if invalid_chars:
            return False
            
        # Vérifier que le texte n'est pas vide
        if not text.strip():
            return False
            
        # Vérifier la présence de coups d'échecs
        if not re.search(r'\d+\.', text):
            return False
            
        return True

    def encode(self, text: str) -> List[int]:
        """Encode le texte en gérant les erreurs potentielles."""
        try:
            return [self.vocab[token] for token in self.tokenize(text)]
        except KeyError as e:
            print(f"Erreur d'encodage - token non trouvé: {e}")
            print(f"Texte problématique: {text}")
            raise

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id2token[id] for id in ids 
                      if id != self.vocab[self.config.pad_token])

    def save(self, path: Optional[str] = None) -> None:
        save_path = path or self.config.save_path
        if save_path is None:
            raise ValueError("Aucun chemin de sauvegarde spécifié")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'valid_chars': list(self.valid_chars),
            'config': {
                'pad_token': self.config.pad_token,
                'special_tokens': self.config.special_tokens
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer sauvegardé dans {save_path}")

    @classmethod
    def load(cls, path: str) -> 'ChessTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        config = TokenizerConfig(
            pad_token=data['config']['pad_token'],
            special_tokens=data['config']['special_tokens'],
            save_path=path
        )
        
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.id2token = {int(k): v for k, v in data['id2token'].items()}
        tokenizer.valid_chars = set(data['valid_chars'])
        tokenizer.token_id = max(map(int, data['id2token'].keys())) + 1
        
        return tokenizer

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                    show_progress: bool = True) -> List[List[int]]:
        iterator = tqdm(texts) if show_progress else texts
        encoded = []
        
        for text in iterator:
            try:
                tokens = self.encode(text)
                if max_length:
                    tokens = tokens[:max_length]
                    if len(tokens) < max_length:
                        tokens += [self.vocab[self.config.pad_token]] * (max_length - len(tokens))
                encoded.append(tokens)
            except Exception as e:
                print(f"Erreur lors de l'encodage du texte: {e}")
                continue
                
        return encoded

    def debug_text(self, text: str) -> None:
        """Affiche des informations de debug sur le texte."""
        print("\nDEBUG TEXT:")
        print("Texte original:", text)
        print("Longueur:", len(text))
        print("Caractères uniques:", sorted(set(text)))
        
        cleaned = self.clean_text(text)
        print("\nTexte nettoyé:", cleaned)
        print("Longueur après nettoyage:", len(cleaned))
        print("Caractères uniques après nettoyage:", sorted(set(cleaned)))
        
        print("\nTokens:", self.tokenize(cleaned))
        
        try:
            encoded = self.encode(cleaned)
            print("Encodage réussi:", encoded)
            print("Décodage:", self.decode(encoded))
        except Exception as e:
            print(f"Erreur d'encodage: {e}")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size

if __name__ == "__main__":
    config = TokenizerConfig(
        pad_token='<PAD>',
        special_tokens=['<START>', '<END>'],
        save_path='./models/chess_tokenizer.json'
    )
    
    tokenizer = ChessTokenizer(config)
    
    # Test
    game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    print("\nTest de tokenization:")
    print("Texte:", game)
    tokens = tokenizer.tokenize(game)
    print("Tokens:", tokens)
    ids = tokenizer.encode(game)
    print("IDs:", ids)
    decoded = tokenizer.decode(ids)
    print("Décodé:", decoded)
    
    tokenizer.save()