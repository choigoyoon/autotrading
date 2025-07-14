import pandas as pd
import sys

def peek_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"--- {file_path} ---")
        print("Shape:", df.shape)
        print("Columns:", df.columns)
        print("Head:\n", df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        peek_data(sys.argv[1])
    else:
        # 여기에 보고 싶은 기본 파일 경로를 넣으세요.
        peek_data('data/processed/btc_usdt_kst/super_divergence_dataset.parquet') 