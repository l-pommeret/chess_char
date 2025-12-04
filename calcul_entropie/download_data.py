import sys
import os

# Add parent directory to path to import downloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloader import ChessDataDownloader

def download_data():
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-09.pgn.zst"
    save_dir = "../chess_data"  # Save in parent directory's chess_data folder
    
    print(f"Downloading data from {url}...")
    downloader = ChessDataDownloader(url, save_dir)
    zst_file = downloader.download()
    print(f"Downloaded to {zst_file}")
    
    print("Decompressing...")
    pgn_file = downloader.decompress(zst_file)
    print(f"Decompressed to {pgn_file}")
    
    return pgn_file

if __name__ == "__main__":
    download_data()
