import pandas as pd
import sys
from pandas import DataFrame
from pathlib import Path

def peek_data(file_path: str | Path) -> None:
    """
    Parquet 파일의 기본적인 정보를 출력합니다.

    Args:
        file_path (str | Path): 확인할 Parquet 파일의 경로.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ Error: 파일 또는 경로를 찾을 수 없습니다 -> {file_path}")
            return
            
        print(f"--- 🔍 {path.name} 파일 정보 ---")
        df: DataFrame = pd.read_parquet(path)
        print("📊 Shape:", df.shape)
        print("📋 Columns:", df.columns.tolist())
        print("👀 Head:\n", df.head())
        
    except Exception as e:
        print(f"❌ Error: '{file_path}' 파일을 읽는 중 오류 발생: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        peek_data(sys.argv[1])
    else:
        default_file = 'data/processed/btc_usdt_kst/super_divergence_dataset.parquet'
        print(f"ℹ️  인자가 제공되지 않았습니다. 기본 파일로 실행합니다: {default_file}")
        peek_data(default_file) 