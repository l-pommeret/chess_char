from downloader import ChessDataDownloader
from tokenizer import ChessTokenizer
from parser import PGNParser
from dataset import create_datasets

def main():
    # Configuration
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-09.pgn.zst"
    save_dir = "chess_data"
    max_length = 1024

    # Téléchargement et décompression
    downloader = ChessDataDownloader(url, save_dir)
    zst_file = downloader.download()
    pgn_file = downloader.decompress(zst_file)

    # Initialisation du tokenizer
    tokenizer = ChessTokenizer()

    # Parsing des parties
    parser = PGNParser(tokenizer)
    games, filtered_count, total_count = parser.parse_file(pgn_file)
    print(f"Parties filtrées : {filtered_count}/{total_count}")

    open('chess_data/filtered_games.pgn', 'w').write('\n\n'.join(str(game) for game in games))

    # Création des datasets
    train_dataset, test_dataset = create_datasets(games, tokenizer, max_length)
    print(f"Dataset d'entraînement : {len(train_dataset)} parties")
    print(f"Dataset de test : {len(test_dataset)} parties")

if __name__ == "__main__":
    main()