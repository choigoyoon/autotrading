import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

# --- 타입 정의 ---
OhlcvData = Dict[str, npt.NDArray[Any]]
ChunkArgs = Tuple[int, str, pd.DataFrame, npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]

# --- 설정 클래스 ---
class Config:
    """분석 스크립트의 설정을 관리합니다."""
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    LABELS_FILE: Path = PROJECT_ROOT / 'data' / 'labels_macd' / 'btc_usdt_kst' / 'btc_usdt_kst_merged_labels.parquet'
    OHLCV_DIR: Path = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv'
    OUTPUT_FILE: Path = PROJECT_ROOT / 'results' / 'label_performance' / 'performance_simple_fast.csv'
    
    CPU_COUNT: int = os.cpu_count() or 1
    MAX_WORKERS: int = min(CPU_COUNT * 2, 32)
    CHUNK_SIZE_LARGE: int = 20000
    CHUNK_SIZE_SMALL: int = 10000

    TF_MAP: Dict[str, str] = {
        '1D': '1day', '2D': '2day', '3D': '3day', '1W': '1week',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1min': '1min', '3min': '3min', '5min': '5min', '10min': '10min', 
        '15min': '15min', '30min': '30min'
    }

# --- Numba 최적화 함수 ---
@numba.njit(parallel=True, fastmath=True)
def analyze_labels(
    label_timestamps: npt.NDArray[np.int64], 
    label_prices: npt.NDArray[np.float64], 
    label_types: npt.NDArray[np.int64], 
    ohlcv_timestamps: npt.NDArray[np.int64], 
    ohlcv_high: npt.NDArray[np.float64], 
    ohlcv_low: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """CPU 병렬 최적화된 핵심 분석 함수"""
    num_labels = len(label_timestamps)
    results = np.zeros((num_labels, 3), dtype=np.float64)
    
    for i in numba.prange(num_labels):
        label_ts = label_timestamps[i]
        label_price = label_prices[i]
        label_type = label_types[i]
        
        start_idx = np.searchsorted(ohlcv_timestamps, label_ts, side='left')
        
        if start_idx >= len(ohlcv_high):
            continue
        
        future_high = ohlcv_high[start_idx:]
        future_low = ohlcv_low[start_idx:]

        if len(future_high) == 0:
            continue
            
        profit_pct: float = 0.0
        if label_type == 1:  # 매수
            max_price = np.max(future_high)
            if label_price > 0:
                profit_pct = ((max_price - label_price) / label_price) * 100
        else:  # 매도
            min_price = np.min(future_low)
            if label_price > 0:
                profit_pct = ((label_price - min_price) / label_price) * 100
            
        results[i, 0] = label_ts
        results[i, 1] = profit_pct
        results[i, 2] = label_type
        
    return results

# --- 병렬 처리 워커 함수 ---
def process_chunk(args: ChunkArgs) -> Optional[pd.DataFrame]:
    """데이터 청크를 받아 Numba 함수를 호출하고 결과를 DataFrame으로 반환합니다."""
    chunk_id, timeframe, chunk_df, ohlcv_timestamps, ohlcv_high, ohlcv_low = args
    
    try:
        label_timestamps = chunk_df.index.view('int64')
        label_prices = chunk_df['close'].to_numpy(dtype=np.float64)
        label_types = chunk_df['label'].to_numpy(dtype=np.int64)
        
        results_array = analyze_labels(
            label_timestamps, label_prices, label_types,
            ohlcv_timestamps, ohlcv_high, ohlcv_low
        )
        
        df = pd.DataFrame(results_array, columns=['timestamp', 'profit_pct', 'label_type'])
        # 0인 타임스탬프 (분석 안 된 데이터) 제거
        df = df[df['timestamp'] > 0]
        if df.empty:
            return None

        df['timeframe'] = timeframe
        df['entry_time'] = pd.to_datetime(df['timestamp'], unit='ns')
        return df
        
    except Exception as e:
        print(f"[오류] {timeframe}_청크{chunk_id}: {e}")
        return None

# --- 메인 실행 로직 ---
def main() -> None:
    """성과 분석 메인 실행 함수"""
    config = Config()
    os.environ['NUMBA_NUM_THREADS'] = str(config.CPU_COUNT)
    config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"CPU 코어 수: {config.CPU_COUNT}, 사용할 워커 수: {config.MAX_WORKERS}")
    
    print("1. 데이터 로드 중...")
    signals_df = pd.read_parquet(config.LABELS_FILE)
    print(f"   총 {len(signals_df):,}개 라벨")
    
    print("2. OHLCV 로드 중...")
    ohlcv_data: Dict[str, OhlcvData] = {}
    for tf_obj in signals_df['timeframe'].unique():
        tf = str(tf_obj)
        file_name = config.TF_MAP.get(tf, tf)
        ohlcv_file = config.OHLCV_DIR / f"{file_name}.parquet"
        if ohlcv_file.exists():
            ohlcv_df = pd.read_parquet(ohlcv_file)
            ohlcv_data[tf] = {
                'timestamps': ohlcv_df.index.view('int64'),
                'high': ohlcv_df['high'].to_numpy(dtype=np.float64),
                'low': ohlcv_df['low'].to_numpy(dtype=np.float64)
            }
    
    print("3. 청크 생성 중...")
    all_chunks: List[ChunkArgs] = []
    for tf, group in signals_df.groupby('timeframe'):
        tf_str = str(tf)
        if tf_str not in ohlcv_data:
            continue
        
        chunk_size = config.CHUNK_SIZE_LARGE if len(group) > 100000 else config.CHUNK_SIZE_SMALL
        for i in range(0, len(group), chunk_size):
            chunk = group.iloc[i:i+chunk_size]
            ohlcv = ohlcv_data[tf_str]
            all_chunks.append((
                i // chunk_size, tf_str, chunk,
                ohlcv['timestamps'], ohlcv['high'], ohlcv['low']
            ))
    print(f"   총 {len(all_chunks)}개 청크 생성")
    
    print("4. 병렬 분석 시작...")
    results: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in all_chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 분석"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"청크 처리 중 메인 스레드 예외: {e}")

    print("5. 결과 정리...")
    if not results:
        print("❌ 결과 없음")
        return
        
    final_df = pd.concat(results, ignore_index=True)
    final_df['signal_type'] = final_df['label_type'].map({1.0: 'Buy', -1.0: 'Sell'})
    final_df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"✅ 완료: {len(final_df):,}개 결과")
    print(f"평균 수익률: {final_df['profit_pct'].mean():.2f}%")

if __name__ == "__main__":
    main()