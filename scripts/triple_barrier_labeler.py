import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count
import functools

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- 타임프레임별 Triple-Barrier 설정 ---
# 각 타임프레임의 변동성과 특성에 맞춰 장벽 및 시간 제한을 다르게 설정합니다.
# profit_pct: 수익 실현 장벽 (%)
# loss_pct: 손실 제한 장벽 (%)
# time_limit_candles: 최대 보유 기간 (캔들 수)
TIMEFRAME_CONFIGS = {
    "default":  {"profit_pct": 2.0, "loss_pct": 2.0, "time_limit_candles": 50},
    "1min":     {"profit_pct": 0.5, "loss_pct": 0.5, "time_limit_candles": 60},    # 1시간
    "3min":     {"profit_pct": 1.0, "loss_pct": 1.0, "time_limit_candles": 20},    # 1시간
    "5min":     {"profit_pct": 1.2, "loss_pct": 1.2, "time_limit_candles": 12},    # 1시간
    "10min":    {"profit_pct": 1.5, "loss_pct": 1.5, "time_limit_candles": 12},    # 2시간
    "15min":    {"profit_pct": 1.8, "loss_pct": 1.8, "time_limit_candles": 8},     # 2시간
    "30min":    {"profit_pct": 2.0, "loss_pct": 2.0, "time_limit_candles": 6},     # 3시간
    "1h":       {"profit_pct": 2.5, "loss_pct": 2.5, "time_limit_candles": 4},     # 4시간
    "2h":       {"profit_pct": 3.0, "loss_pct": 3.0, "time_limit_candles": 4},     # 8시간
    "4h":       {"profit_pct": 3.5, "loss_pct": 3.5, "time_limit_candles": 6},     # 1일
    "6h":       {"profit_pct": 4.0, "loss_pct": 4.0, "time_limit_candles": 4},     # 1일
    "8h":       {"profit_pct": 4.5, "loss_pct": 4.5, "time_limit_candles": 6},     # 2일
    "12h":      {"profit_pct": 5.0, "loss_pct": 5.0, "time_limit_candles": 4},     # 2일
    "1day":     {"profit_pct": 7.0, "loss_pct": 7.0, "time_limit_candles": 7},     # 1주
    "3day":     {"profit_pct": 10.0, "loss_pct": 10.0, "time_limit_candles": 5},    # 15일
    "1week":    {"profit_pct": 12.0, "loss_pct": 12.0, "time_limit_candles": 4},    # 1달
    "1month":   {"profit_pct": 20.0, "loss_pct": 20.0, "time_limit_candles": 3},    # 3달
}

def label_chunk(df_chunk: pd.DataFrame, config: dict) -> pd.Series:
    """
    데이터프레임의 작은 덩어리(chunk)에 대해 Triple-Barrier 라벨링을 수행합니다.
    (이 함수는 멀티프로세싱의 개별 작업자(worker)가 실행합니다)
    """
    prices = df_chunk['close'].values
    labels = np.full(len(prices), -1, dtype=np.int8)
    
    profit_pct = config['profit_pct'] / 100.0
    loss_pct = config['loss_pct'] / 100.0
    time_limit_candles = config['time_limit_candles']
    
    for i in range(len(prices) - time_limit_candles):
        entry_price = prices[i]
        upper_barrier = entry_price * (1 + profit_pct)
        lower_barrier = entry_price * (1 - loss_pct)
        
        for j in range(i + 1, i + 1 + time_limit_candles):
            # 경계 검사: j가 prices 배열의 유효한 인덱스 내에 있는지 확인
            if j >= len(prices):
                break
            future_price = prices[j]
            
            if future_price >= upper_barrier:
                labels[i] = 1
                break
            elif future_price <= lower_barrier:
                labels[i] = 0
                break
        
        if labels[i] == -1:
            final_price = prices[i + time_limit_candles]
            labels[i] = 1 if final_price > entry_price else 0
            
    return pd.Series(labels, index=df_chunk.index)

def get_labels_parallel(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    멀티프로세싱을 사용하여 Triple-Barrier 라벨링을 병렬로 수행합니다.
    """
    n_cores = cpu_count() - 1 or 1 # 사용 가능한 코어 수 (최소 1개)
    chunk_size = int(np.ceil(len(df) / n_cores))
    
    # 데이터프레임을 여러 개의 덩어리로 분할
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # functools.partial을 사용하여 config 인자를 label_chunk 함수에 고정
    worker_func = functools.partial(label_chunk, config=config)

    print(f"{n_cores}개의 코어를 사용하여 병렬 라벨링을 시작합니다...")
    
    with Pool(n_cores) as pool:
        # tqdm을 Pool.imap에 적용하여 진행 상황 추적
        results = list(tqdm(pool.imap(worker_func, chunks), total=len(chunks), desc="병렬 라벨링 진행률"))
    
    # 모든 결과를 하나로 합침
    combined_labels = pd.concat(results)
    return combined_labels.rename("label_tb")


def process_all_timeframes(target_timeframe: str | None = None):
    """모든 타임프레임에 대해 Triple-Barrier 라벨링을 수행합니다."""
    
    input_dir = project_root / 'data/processed/btc_usdt_kst/resampled_ohlcv'
    output_dir = project_root / 'data/labels_triple_barrier/btc_usdt_kst'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timeframes = list(TIMEFRAME_CONFIGS.keys())
    if "default" in timeframes:
        timeframes.remove("default")
    
    if target_timeframe:
        if target_timeframe not in timeframes:
            print(f"오류: '{target_timeframe}'는 유효한 타임프레임이 아닙니다.")
            return
        timeframes = [target_timeframe]

    for tf in timeframes:
        print(f"\n--- '{tf}' 타임프레임 처리 시작 ---")
        input_file = input_dir / f"{tf}.parquet"
        if not input_file.exists():
            print(f"파일 없음: {input_file}")
            continue
            
        df = pd.read_parquet(input_file)
        print(f"데이터 로드 완료: {len(df):,}개 캔들")
        
        config = TIMEFRAME_CONFIGS.get(tf, TIMEFRAME_CONFIGS["default"])
        print(f"사용된 설정: {config}")
        
        # 병렬 라벨링 실행
        labels = get_labels_parallel(df, config)
        
        # 라벨링 결과(라벨 != -1)만 필터링
        valid_labels = labels[labels != -1].copy()
        
        # 라벨 분포 확인
        label_counts = valid_labels.value_counts()
        buy_count = label_counts.get(1, 0)
        sell_count = label_counts.get(0, 0)
        total_valid = len(valid_labels)
        
        if total_valid > 0:
            print("라벨 분포:")
            print(f"  매수 (1): {buy_count:,}개 ({(buy_count/total_valid)*100:.2f}%)")
            print(f"  매도 (0): {sell_count:,}개 ({(sell_count/total_valid)*100:.2f}%)")
        else:
            print("생성된 유효 라벨이 없습니다.")

        # 라벨 데이터만 저장
        output_file = output_dir / f"triple_barrier_{tf}.parquet"
        valid_labels.to_frame(name='label').to_parquet(output_file)
        print(f"저장 완료: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="미래 데이터 누수 없는 Triple-Barrier 라벨링 스크립트")
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="특정 타임프레임만 처리합니다 (예: '1min', '1h'). 지정하지 않으면 모두 처리합니다."
    )
    args = parser.parse_args()
    
    print("==============================================")
    print("=== Triple-Barrier 라벨링 시스템 시작 ===")
    print("==============================================")
    
    process_all_timeframes(target_timeframe=args.timeframe)
    
    print("\n==============================================")
    print("=== 모든 라벨링 작업 완료 ===")
    print("==============================================")

if __name__ == "__main__":
    main() 