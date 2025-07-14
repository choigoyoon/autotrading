import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import numba
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 🔥 핵심: numba parallel=True로 CPU 활용률 극대화
@numba.njit(parallel=True, fastmath=True)
def analyze_labels(label_timestamps, label_prices, label_types, 
                   ohlcv_timestamps, ohlcv_high, ohlcv_low):
    """CPU 병렬 최적화된 핵심 함수"""
    num_labels = len(label_timestamps)
    results = np.zeros((num_labels, 3), dtype=np.float64)
    
    # 🔥 numba.prange = 모든 CPU 코어 활용!
    for i in numba.prange(num_labels):
        label_ts = label_timestamps[i]
        label_price = label_prices[i]
        label_type = label_types[i]
        
        start_idx = np.searchsorted(ohlcv_timestamps, label_ts)
        
        if start_idx >= len(ohlcv_high):
            continue
            
        if label_type == 1:  # 매수
            max_price = np.max(ohlcv_high[start_idx:])
            profit_pct = ((max_price - label_price) / label_price) * 100
        else:  # 매도
            min_price = np.min(ohlcv_low[start_idx:])
            profit_pct = ((label_price - min_price) / label_price) * 100
            
        results[i, 0] = label_ts
        results[i, 1] = profit_pct
        results[i, 2] = label_type
        
    return results

def process_chunk(args):
    """청크 처리 (단순화)"""
    chunk_id, timeframe, chunk_df, ohlcv_timestamps, ohlcv_high, ohlcv_low = args
    
    try:
        # 데이터 준비
        label_timestamps = chunk_df.index.view('int64')
        label_prices = chunk_df['close'].values.astype(np.float64)
        label_types = chunk_df['label'].values.astype(np.int64)
        
        # 분석 실행 (내부적으로 모든 CPU 코어 사용)
        results = analyze_labels(label_timestamps, label_prices, label_types,
                               ohlcv_timestamps, ohlcv_high, ohlcv_low)
        
        # DataFrame 생성
        df = pd.DataFrame(results, columns=['timestamp', 'profit_pct', 'label_type'])
        df['timeframe'] = timeframe
        df['entry_time'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        print(f"[오류] {timeframe}_청크{chunk_id}: {e}")
        return None

def main():
    """초간단 메인 함수"""
    # 🔥 CPU 코어 최대 활용 설정
    os.environ['NUMBA_NUM_THREADS'] = str(os.cpu_count())
    
    # 경로 설정
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    labels_file = PROJECT_ROOT / 'data' / 'labels_macd' / 'btc_usdt_kst' / 'btc_usdt_kst_merged_labels.parquet'
    ohlcv_dir = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv'
    output_file = PROJECT_ROOT / 'results' / 'label_performance' / 'performance_simple_fast.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"CPU 코어 수: {os.cpu_count()}")
    
    # 데이터 로드
    print("1. 데이터 로드 중...")
    signals_df = pd.read_parquet(labels_file)
    print(f"   총 {len(signals_df):,}개 라벨")
    
    # OHLCV 로드
    print("2. OHLCV 로드 중...")
    TF_MAP = {
        '1D': '1day', '2D': '2day', '3D': '3day', '1W': '1week',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1min': '1min', '3min': '3min', '5min': '5min', '10min': '10min', 
        '15min': '15min', '30min': '30min'
    }
    
    ohlcv_data = {}
    for tf in signals_df['timeframe'].unique():
        file_name = TF_MAP.get(tf, tf)
        ohlcv_file = ohlcv_dir / f"{file_name}.parquet"
        if ohlcv_file.exists():
            ohlcv_df = pd.read_parquet(ohlcv_file)
            ohlcv_data[tf] = {
                'timestamps': ohlcv_df.index.view('int64'),
                'high': ohlcv_df['high'].values.astype(np.float64),
                'low': ohlcv_df['low'].values.astype(np.float64)
            }
    
    # 🔥 간단한 청크 생성
    print("3. 청크 생성 중...")
    all_chunks = []
    
    for tf, group in signals_df.groupby('timeframe'):
        if tf not in ohlcv_data:
            continue
            
        # 🔥 간단한 청크 크기 결정
        chunk_size = 20000 if len(group) > 100000 else 10000
        
        # 청크 분할
        for i in range(0, len(group), chunk_size):
            chunk = group.iloc[i:i+chunk_size]
            ohlcv = ohlcv_data[tf]
            chunk_args = (i//chunk_size, tf, chunk, 
                         ohlcv['timestamps'], ohlcv['high'], ohlcv['low'])
            all_chunks.append(chunk_args)
    
    print(f"   총 {len(all_chunks)}개 청크 생성")
    
    # 🔥 최대 병렬 처리
    print("4. 병렬 분석 시작...")
    results = []
    
    # 🔥 워커 수 대폭 증가
    max_workers = min(os.cpu_count() * 2, 32)  # CPU 코어 수 x 2
    print(f"   워커 수: {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk_args) for chunk_args in all_chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 분석"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                continue
    
    # 결과 정리
    print("5. 결과 정리...")
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df['signal_type'] = final_df['label_type'].map({1: 'Buy', -1: 'Sell'})
        final_df.to_csv(output_file, index=False)
        
        print(f"✅ 완료: {len(final_df):,}개 결과")
        print(f"평균 수익률: {final_df['profit_pct'].mean():.2f}%")
    else:
        print("❌ 결과 없음")

if __name__ == "__main__":
    main()