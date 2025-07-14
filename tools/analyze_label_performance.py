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
AnalysisResult = Dict[str, Any]

# --- 설정 클래스 ---
class Config:
    """분석 스크립트의 설정을 관리합니다."""
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    LABELS_DIR: Path = PROJECT_ROOT / 'data' / 'labels_macd' / 'btc_usdt_kst'
    OHLCV_DIR: Path = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv'
    RESULTS_DIR: Path = PROJECT_ROOT / 'results' / 'label_performance'
    MERGED_LABELS_FILE: Path = LABELS_DIR / 'btc_usdt_kst_merged_labels.parquet'
    OUTPUT_FILE: Path = RESULTS_DIR / 'performance_analysis_parallel.csv'
    
    # CPU 코어 수의 절반 또는 최대 16개 중 작은 값을 사용
    MAX_WORKERS: int = min(os.cpu_count() or 1, 16)
    CHUNK_SIZE: int = 50000

    TF_MAP: Dict[str, str] = {
        '1D': '1day', '2D': '2day', '3D': '3day', '1W': '1week',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1min': '1min', '3min': '3min', '5min': '5min', '10min': '10min',
        '15min': '15min', '30min': '30min'
    }

# --- Numba 최적화 함수 ---
@numba.njit(fastmath=True)
def analyze_single_signal(
    signal_ts: np.int64, 
    signal_price: np.float64, 
    signal_label: np.int64, 
    ohlcv_timestamps: npt.NDArray[np.int64], 
    ohlcv_high: npt.NDArray[np.float64], 
    ohlcv_low: npt.NDArray[np.float64], 
) -> Tuple[float, int, int, int]:
    """단일 신호 분석 (numba 최적화)"""
    start_idx = np.searchsorted(ohlcv_timestamps, signal_ts, side='left')
    
    if start_idx >= len(ohlcv_high):
        return 0.0, -1, 0, -1

    future_high = ohlcv_high[start_idx:]
    future_low = ohlcv_low[start_idx:]
    
    if len(future_high) == 0:
        return 0.0, -1, 0, -1
    
    max_profit_pct: float = 0.0
    peak_idx: int = -1
    
    if signal_label == 1:  # 매수
        peak_idx = int(np.argmax(future_high))
        peak_price = future_high[peak_idx]
        max_profit_pct = ((peak_price - signal_price) / signal_price) * 100
    else:  # 매도
        peak_idx = int(np.argmin(future_low))
        trough_price = future_low[peak_idx]
        max_profit_pct = ((signal_price - trough_price) / signal_price) * 100
    
    status: int = 1  # 1: Holding, 2: Breakeven Exit
    exit_idx: int = -1
    
    if peak_idx < len(future_high) - 1:
        after_peak_high = future_high[peak_idx + 1:]
        after_peak_low = future_low[peak_idx + 1:]
        
        if signal_label == 1:
            indices = np.where(after_peak_low <= signal_price)[0]
            if len(indices) > 0:
                status = 2
                exit_idx = peak_idx + 1 + int(indices[0])
        else:
            indices = np.where(after_peak_high >= signal_price)[0]
            if len(indices) > 0:
                status = 2
                exit_idx = peak_idx + 1 + int(indices[0])
    
    final_peak_idx = start_idx + peak_idx if peak_idx != -1 else -1
    final_exit_idx = start_idx + exit_idx if exit_idx != -1 else -1
    return max_profit_pct, int(final_peak_idx), status, int(final_exit_idx)

# --- 병렬 처리 워커 함수 ---
def process_timeframe_chunk(
    timeframe: str,
    chunk_df: pd.DataFrame,
    ohlcv_data: OhlcvData
) -> Optional[pd.DataFrame]:
    """타임프레임 청크를 처리하고 분석 결과를 데이터프레임으로 반환합니다."""
    try:
        signal_timestamps: npt.NDArray[np.int64] = chunk_df.index.view('int64')
        signal_prices: npt.NDArray[np.float64] = chunk_df['close'].to_numpy(dtype=np.float64)
        signal_labels: npt.NDArray[np.int64] = chunk_df['label'].to_numpy(dtype=np.int64)
        
        ohlcv_ts = ohlcv_data['timestamps']
        ohlcv_h = ohlcv_data['high']
        ohlcv_l = ohlcv_data['low']
        
        results: List[AnalysisResult] = []
        
        for i in range(len(signal_timestamps)):
            max_profit, peak_idx, status, exit_idx = analyze_single_signal(
                signal_timestamps[i], signal_prices[i], signal_labels[i],
                ohlcv_ts, ohlcv_h, ohlcv_l
            )
            
            peak_timestamp = ohlcv_ts[peak_idx] if peak_idx != -1 else np.nan
            exit_timestamp = ohlcv_ts[exit_idx] if exit_idx != -1 else np.nan
            
            candles_to_exit = np.nan
            if exit_idx != -1:
                entry_idx = np.searchsorted(ohlcv_ts, signal_timestamps[i], side='left')
                candles_to_exit = exit_idx - entry_idx
            
            results.append({
                '진입시점': pd.to_datetime(signal_timestamps[i]),
                '타임프레임': timeframe,
                '진입가격': signal_prices[i],
                '신호타입': 'Buy' if signal_labels[i] == 1 else 'Sell',
                '최대수익률(%)': max_profit,
                '최고(저)점시점': pd.to_datetime(peak_timestamp, errors='coerce'),
                '최종상태': 'Holding' if status == 1 else 'Breakeven Exit',
                '청산시점': pd.to_datetime(exit_timestamp, errors='coerce'),
                '청산까지캔들수': float(candles_to_exit),
            })
        
        return pd.DataFrame(results) if results else None
    except Exception as e:
        print(f"[오류] {timeframe} 청크 처리 중 예외 발생: {e}")
        return None

# --- 메인 분석 로직 ---
def load_ohlcv_data(config: Config, timeframes: npt.NDArray[Any]) -> Dict[str, OhlcvData]:
    """필요한 모든 OHLCV 데이터를 로드합니다."""
    ohlcv_data: Dict[str, OhlcvData] = {}
    print("2. OHLCV 데이터 로드 중...")
    for tf in timeframes:
        file_name = config.TF_MAP.get(tf, tf)
        ohlcv_file = config.OHLCV_DIR / f"{file_name}.parquet"
        
        if not ohlcv_file.exists():
            ohlcv_file = config.OHLCV_DIR / f"{tf.replace('min', 'm')}.parquet"
        
        if ohlcv_file.exists():
            try:
                ohlcv_df = pd.read_parquet(ohlcv_file)
                ohlcv_data[tf] = {
                    'timestamps': ohlcv_df.index.view('int64'),
                    'high': ohlcv_df['high'].to_numpy(dtype=np.float64),
                    'low': ohlcv_df['low'].to_numpy(dtype=np.float64)
                }
                print(f"   {tf}: {ohlcv_df.shape}")
            except Exception as e:
                print(f"⚠️ {tf} 로드 실패: {e}")
        else:
            print(f"⚠️ {tf} 파일 없음")
    return ohlcv_data

def analyze_performance() -> None:
    """성과 분석을 병렬로 실행하고 결과를 저장합니다."""
    config = Config()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CPU 코어 수: {os.cpu_count()}, 사용할 워커 수: {config.MAX_WORKERS}")

    if not config.MERGED_LABELS_FILE.exists():
        print(f"❌ 오류: 병합된 라벨 파일을 찾을 수 없습니다: {config.MERGED_LABELS_FILE}")
        return

    print("1. 병합된 라벨 데이터 로드 중...")
    signals_df = pd.read_parquet(config.MERGED_LABELS_FILE)
    print(f"   총 {len(signals_df):,}개의 신호를 분석합니다.")

    ohlcv_data = load_ohlcv_data(config, signals_df['timeframe'].unique())

    print("3. 병렬 성과 분석 시작...")
    tasks: List[Tuple[str, pd.DataFrame, OhlcvData]] = []
    for tf_obj, group in signals_df.groupby(by='timeframe'):
        tf = str(tf_obj)
        if tf in ohlcv_data:
            for i in range(0, len(group), config.CHUNK_SIZE):
                chunk = group.iloc[i:i + config.CHUNK_SIZE]
                tasks.append((tf, chunk, ohlcv_data[tf]))
    
    print(f"   총 {len(tasks)}개 청크로 분할")
    
    results: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(process_timeframe_chunk, *task): task for task in tasks
        }
        for future in tqdm(as_completed(future_to_chunk), total=len(tasks), desc="병렬 분석"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"청크 처리 중 메인 스레드에서 예외 발생: {e}")

    if not results:
        print("❌ 분석 결과 없음")
        return

    print("4. 결과 정리 및 저장...")
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉 성과 분석 완료!")
    print(f"✅ 결과 저장: {config.OUTPUT_FILE}")
    print_summary_statistics(results_df)

def print_summary_statistics(df: pd.DataFrame) -> None:
    """최종 분석 결과에 대한 요약 통계를 출력합니다."""
    print(f"   - 총 분석된 신호: {len(df):,}개")
    print(f"   - 평균 최대 수익률: {df['최대수익률(%)'].mean():.2f}%")
    
    print(f"\n📊 빠른 통계:")
    print(f"   - Holding: {(df['최종상태'] == 'Holding').sum():,}개")
    print(f"   - Breakeven Exit: {(df['최종상태'] == 'Breakeven Exit').sum():,}개")
    print(f"   - 양수 수익률: {(df['최대수익률(%)'] > 0).sum():,}개 ({(df['최대수익률(%)'] > 0).mean()*100:.1f}%)")
    
    print(f"\n📈 타임프레임별 평균 수익률:")
    tf_summary = df.groupby('타임프레임')['최대수익률(%)'].agg(['count', 'mean']).round(2)
    with pd.option_context('display.max_rows', None):
        print(tf_summary)

if __name__ == "__main__":
    analyze_performance()