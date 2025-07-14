import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numba
import os

@numba.njit(fastmath=True)
def analyze_single_signal(signal_ts, signal_price, signal_label, 
                         ohlcv_timestamps, ohlcv_high, ohlcv_low, ohlcv_close):
    """단일 신호 분석 (numba 최적화)"""
    # 신호 이후 데이터만 사용
    start_idx = np.searchsorted(ohlcv_timestamps, signal_ts)
    
    if start_idx >= len(ohlcv_high):
        return 0.0, -1, 0, -1  # 수익률, 최고점idx, 상태, 청산idx
    
    future_high = ohlcv_high[start_idx:]
    future_low = ohlcv_low[start_idx:]
    future_close = ohlcv_close[start_idx:]
    
    if len(future_high) == 0:
        return 0.0, -1, 0, -1
    
    # 최대 수익률 계산
    max_profit_pct = 0.0
    peak_idx = -1
    
    if signal_label == 1:  # 매수
        peak_idx = np.argmax(future_high)
        peak_price = future_high[peak_idx]
        max_profit_pct = ((peak_price - signal_price) / signal_price) * 100
    else:  # 매도
        peak_idx = np.argmin(future_low)
        trough_price = future_low[peak_idx]
        max_profit_pct = ((signal_price - trough_price) / signal_price) * 100
    
    # 본절 복귀 체크 (최고점 이후)
    status = 1  # 1: Holding, 2: Breakeven Exit
    exit_idx = -1
    
    if peak_idx < len(future_high) - 1:
        after_peak_high = future_high[peak_idx + 1:]
        after_peak_low = future_low[peak_idx + 1:]
        
        if signal_label == 1:  # 매수 → 저점이 진입가 이하로
            for i in range(len(after_peak_low)):
                if after_peak_low[i] <= signal_price:
                    status = 2
                    exit_idx = peak_idx + 1 + i
                    break
        else:  # 매도 → 고점이 진입가 이상으로
            for i in range(len(after_peak_high)):
                if after_peak_high[i] >= signal_price:
                    status = 2
                    exit_idx = peak_idx + 1 + i
                    break
    
    return max_profit_pct, start_idx + peak_idx, status, start_idx + exit_idx if exit_idx != -1 else -1

def process_timeframe_chunk(args):
    """타임프레임 청크 처리"""
    timeframe, chunk_df, ohlcv_data = args
    
    try:
        # 데이터 준비
        signal_timestamps = chunk_df.index.view('int64')
        signal_prices = chunk_df['close'].values.astype(np.float64)
        signal_labels = chunk_df['label'].values.astype(np.int64)
        
        ohlcv_timestamps = ohlcv_data['timestamps']
        ohlcv_high = ohlcv_data['high']
        ohlcv_low = ohlcv_data['low']
        ohlcv_close = ohlcv_data['close']
        
        results = []
        
        # 각 신호 분석
        for i in range(len(signal_timestamps)):
            signal_ts = signal_timestamps[i]
            signal_price = signal_prices[i]
            signal_label = signal_labels[i]
            
            max_profit, peak_idx, status, exit_idx = analyze_single_signal(
                signal_ts, signal_price, signal_label,
                ohlcv_timestamps, ohlcv_high, ohlcv_low, ohlcv_close
            )
            
            # 결과 저장
            peak_timestamp = ohlcv_timestamps[peak_idx] if peak_idx != -1 else None
            exit_timestamp = ohlcv_timestamps[exit_idx] if exit_idx != -1 else None
            candles_to_exit = exit_idx - np.searchsorted(ohlcv_timestamps, signal_ts) if exit_idx != -1 else np.nan
            
            results.append({
                '진입시점': pd.to_datetime(signal_ts),
                '타임프레임': timeframe,
                '진입가격': signal_price,
                '신호타입': 'Buy' if signal_label == 1 else 'Sell',
                '최대수익률(%)': max_profit,
                '최고(저)점시점': pd.to_datetime(peak_timestamp) if peak_timestamp else None,
                '최종상태': 'Holding' if status == 1 else 'Breakeven Exit',
                '청산시점': pd.to_datetime(exit_timestamp) if exit_timestamp else None,
                '청산까지캔들수': candles_to_exit,
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"[오류] {timeframe}: {e}")
        return None

def analyze_performance():
    """병렬화된 성과 분석"""
    # 🔥 경로 수정
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    labels_dir = PROJECT_ROOT / 'data' / 'labels_macd' / 'btc_usdt_kst'
    ohlcv_dir = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv'
    results_dir = PROJECT_ROOT / 'results' / 'label_performance'
    results_dir.mkdir(parents=True, exist_ok=True)

    merged_labels_file = labels_dir / 'btc_usdt_kst_merged_labels.parquet'
    output_file = results_dir / 'performance_analysis_parallel.csv'
    
    print(f"CPU 코어 수: {os.cpu_count()}")
    
    # 데이터 로드
    if not merged_labels_file.exists():
        print(f"❌ 오류: 병합된 라벨 파일을 찾을 수 없습니다.")
        return

    print("1. 병합된 라벨 데이터 로드 중...")
    signals_df = pd.read_parquet(merged_labels_file)
    print(f"   총 {len(signals_df):,}개의 신호를 분석합니다.")

    # OHLCV 로드
    print("2. OHLCV 데이터 로드 중...")
    timeframes = signals_df['timeframe'].unique()
    
    # 🔥 파일명 매핑 (기존 코드 호환)
    TF_MAP = {
        '1D': '1day', '2D': '2day', '3D': '3day', '1W': '1week',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1min': '1min', '3min': '3min', '5min': '5min', '10min': '10min', 
        '15min': '15min', '30min': '30min'
    }
    
    ohlcv_data = {}
    for tf in timeframes:
        file_name = TF_MAP.get(tf, tf)
        ohlcv_file = ohlcv_dir / f"{file_name}.parquet"
        
        # 파일명 변형 시도
        if not ohlcv_file.exists():
            ohlcv_file = ohlcv_dir / f"{tf.replace('min', 'm')}.parquet"
        
        if ohlcv_file.exists():
            try:
                ohlcv_df = pd.read_parquet(ohlcv_file)
                ohlcv_data[tf] = {
                    'timestamps': ohlcv_df.index.view('int64'),
                    'high': ohlcv_df['high'].values.astype(np.float64),
                    'low': ohlcv_df['low'].values.astype(np.float64),
                    'close': ohlcv_df['close'].values.astype(np.float64)
                }
                print(f"   {tf}: {ohlcv_df.shape}")
            except Exception as e:
                print(f"⚠️ {tf} 로드 실패: {e}")
        else:
            print(f"⚠️ {tf} 파일 없음")

    # 🔥 병렬 처리
    print("3. 병렬 성과 분석 시작...")
    
    # 타임프레임별 청크 생성
    chunks = []
    for tf, group in signals_df.groupby('timeframe'):
        if tf in ohlcv_data:
            # 청크 크기 조정 (큰 타임프레임은 작게 나누기)
            chunk_size = 50000 if len(group) > 100000 else len(group)
            
            for i in range(0, len(group), chunk_size):
                chunk = group.iloc[i:i+chunk_size]
                chunks.append((tf, chunk, ohlcv_data[tf]))
    
    print(f"   총 {len(chunks)}개 청크로 분할")
    
    # 병렬 실행
    results = []
    max_workers = min(os.cpu_count(), 16)  # 최대 16개 워커
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_timeframe_chunk, chunk) for chunk in chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 분석"):
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"[에러] 청크 처리 실패: {e}")
                continue

    # 결과 정리
    print("4. 결과 정리...")
    if results:
        results_df = pd.concat(results, ignore_index=True)
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n🎉 성과 분석 완료!")
        print(f"✅ 결과 저장: {output_file}")
        print(f"   - 총 분석된 신호: {len(results_df):,}개")
        print(f"   - 평균 최대 수익률: {results_df['최대수익률(%)'].mean():.2f}%")
        
        # 🔥 간단 통계
        print(f"\n📊 빠른 통계:")
        print(f"   - Holding: {(results_df['최종상태'] == 'Holding').sum():,}개")
        print(f"   - Breakeven Exit: {(results_df['최종상태'] == 'Breakeven Exit').sum():,}개")
        print(f"   - 양수 수익률: {(results_df['최대수익률(%)'] > 0).sum():,}개 ({(results_df['최대수익률(%)'] > 0).mean()*100:.1f}%)")
        
        # 타임프레임별 요약
        print(f"\n📈 타임프레임별 평균 수익률:")
        tf_summary = results_df.groupby('타임프레임')['최대수익률(%)'].agg(['count', 'mean']).round(2)
        for tf, row in tf_summary.iterrows():
            print(f"   {tf}: {row['count']:,}개, 평균 {row['mean']:.2f}%")
            
    else:
        print("❌ 결과 없음")

if __name__ == "__main__":
    analyze_performance()