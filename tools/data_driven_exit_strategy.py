import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

# --- 설정 클래스 ---
class Config:
    """데이터 기반 익절 전략 분석을 위한 설정"""
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    LABELS_PATH: Path = PROJECT_ROOT / 'data' / 'labels' / 'btc_usdt_kst_merged_labels.parquet'
    OHLCV_PATH: Path = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet'
    LOOKAHEAD_WINDOW_MIN: int = 240  # 4시간
    BACKTEST_SAMPLE_SIZE: int = 10000
    RATIO_GRID_STEP: int = 10

# --- 타입 정의 ---
ExitTargets = Dict[str, float]
ExitRatios = Tuple[float, ...]

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """분석에 필요한 라벨과 가격 데이터를 로드합니다."""
    print("--- 데이터 로딩 ---")
    if not config.LABELS_PATH.exists() or not config.OHLCV_PATH.exists():
        raise FileNotFoundError("오류: 필수 데이터 파일(라벨 또는 OHLCV)을 찾을 수 없습니다.")
        
    df_labels = pd.read_parquet(config.LABELS_PATH)
    df_buy_signals = df_labels[df_labels['label'] == 1].copy()
    
    df_ohlcv = pd.read_parquet(config.OHLCV_PATH)
    
    print(f"매수 신호(L값) 데이터 로드 완료: {len(df_buy_signals)}개")
    print(f"가격 데이터 로드 완료: {len(df_ohlcv)}개")
    return df_buy_signals, df_ohlcv

def analyze_profit_distribution_vectorized(
    df_buy_signals: pd.DataFrame, 
    df_ohlcv: pd.DataFrame, 
    lookahead_window_min: int
) -> Optional[ExitTargets]:
    """
    매수 신호 이후의 최대 수익률 분포를 벡터화 연산으로 분석합니다.
    """
    print(f"\n--- [1/3] {lookahead_window_min}분 내 최대 수익률 분포 분석 (벡터화) ---")
    
    signal_times = df_buy_signals.index
    entry_prices = df_ohlcv.reindex(signal_times)['close'].dropna()
    
    if entry_prices.empty:
        print("신호 시간에 해당하는 가격 데이터가 없습니다.")
        return None

    # 모든 신호에 대한 미래 윈도우의 최고가를 한 번에 계산
    # 이 부분은 메모리를 많이 사용할 수 있으므로, 데이터가 매우 클 경우 청크 처리 필요
    rolling_window = f"{lookahead_window_min}min"
    # 'high'의 롤링 최대값을 계산합니다. closed='right'는 윈도우의 오른쪽 경계를 포함합니다.
    future_max_highs = df_ohlcv['high'].rolling(window=rolling_window, min_periods=1).max().shift(-(lookahead_window_min-1))
    
    max_prices_in_window = future_max_highs.reindex(entry_prices.index).dropna()
    
    # 최대 수익률 계산
    max_profits_pct = (max_prices_in_window / entry_prices - 1) * 100
    
    if max_profits_pct.empty:
        print("분석할 수익률 데이터가 없습니다.")
        return None

    exit_targets: ExitTargets = {
        'target1': max_profits_pct.quantile(0.25),
        'target2': max_profits_pct.quantile(0.50),
        'target3': max_profits_pct.quantile(0.75),
    }

    print("\n=== 수익률 분포 분석 결과 ===")
    print(max_profits_pct.describe(percentiles=[.1, .25, .5, .75, .9, .99]))
    
    print("\n=== 데이터 기반 분할 익절 목표가 (수익률 %) ===")
    print(f"  - 1차 익절 목표 (25% 분위): {exit_targets['target1']:.2f}%")
    print(f"  - 2차 익절 목표 (50% 분위): {exit_targets['target2']:.2f}%")
    print(f"  - 3차 익절 목표 (75% 분위): {exit_targets['target3']:.2f}%")
    
    return exit_targets
    
def backtest_single_trade(
    entry_price: float, 
    future_highs: npt.NDArray[np.float64], 
    exit_targets: List[float], 
    exit_ratios: ExitRatios
) -> float:
    """단일 거래에 대한 분할 익절 백테스트를 수행합니다."""
    position_size = 1.0
    trade_pnl = 0.0
    
    for i in range(len(exit_targets)):
        if position_size < 1e-6:
            break

        target_price = entry_price * (1 + exit_targets[i] / 100)
        
        if np.any(future_highs >= target_price):
            exit_size = exit_ratios[i]
            if position_size - exit_size < -1e-6: # 남은 포지션보다 많이 팔 수 없음
                exit_size = position_size

            trade_pnl += ((target_price / entry_price) - 1) * exit_size
            position_size -= exit_size

    # 남은 포지션이 있다면 마지막 가격으로 청산 (여기서는 단순화를 위해 생략)
    return trade_pnl

def find_optimal_exit_strategy(
    df_buy_signals: pd.DataFrame, 
    df_ohlcv: pd.DataFrame, 
    exit_targets: ExitTargets,
    config: Config
) -> None:
    """그리드 서치를 통해 최적의 분할 익절 비율을 탐색합니다."""
    print("\n--- [2/3] 최적 분할 익절 비율 탐색 ---")
    
    ratio_grid = [
        p for p in itertools.product(range(0, 101, config.RATIO_GRID_STEP), repeat=3) 
        if sum(p) == 100
    ]
    
    # 백테스트용 데이터 샘플링
    sample_signals = df_buy_signals.sample(
        n=min(len(df_buy_signals), config.BACKTEST_SAMPLE_SIZE), random_state=42
    )

    best_performance = -np.inf
    best_ratios: Optional[Tuple[int, ...]] = None
    
    # 백테스트에 필요한 데이터를 미리 준비
    signal_times = sample_signals.index
    entry_prices = df_ohlcv.loc[signal_times, 'close']
    
    memoized_futures: Dict[pd.Timestamp, npt.NDArray[np.float64]] = {}

    for ratios in tqdm(ratio_grid, desc="익절 비율 최적화 중"):
        exit_ratios_normalized = tuple(r / 100.0 for r in ratios)
        total_pnl = 0.0
        
        # itertuples()를 사용하여 타입 안정성 확보 (성능 영향 미미)
        for row in entry_prices.itertuples():
            signal_time: pd.Timestamp = row.Index
            entry_price: float = row.close
            
            if signal_time not in memoized_futures:
                end_time = signal_time + pd.Timedelta(minutes=config.LOOKAHEAD_WINDOW_MIN)
                memoized_futures[signal_time] = df_ohlcv.loc[signal_time:end_time, 'high'].to_numpy()
            
            future_highs = memoized_futures[signal_time]
            total_pnl += backtest_single_trade(
                entry_price, future_highs, sorted(exit_targets.values()), exit_ratios_normalized
            )

        performance = total_pnl / len(sample_signals)
        
        if performance > best_performance:
            best_performance = performance
            best_ratios = ratios
            
    print("\n--- [3/3] 최적 익절 전략 결과 ---")
    if best_ratios:
        print(f"최고 성과: 평균 {best_performance:.4f}% 수익")
        print(f"최적 익절 비율 (1차, 2차, 3차): {best_ratios[0]}% : {best_ratios[1]}% : {best_ratios[2]}%")
    else:
        print("최적 전략을 찾지 못했습니다.")

def main() -> None:
    """데이터 기반 익절 전략 분석 메인 함수"""
    print("=" * 60)
    print("======= 데이터 기반 익절 전략 분석 도구 시작 =======")
    print("=" * 60)
    
    config = Config()
    
    try:
        df_buy_signals, df_ohlcv = load_data(config)
        
        exit_targets = analyze_profit_distribution_vectorized(
            df_buy_signals, df_ohlcv, config.LOOKAHEAD_WINDOW_MIN
        )
        
        if exit_targets:
            find_optimal_exit_strategy(df_buy_signals, df_ohlcv, exit_targets, config)
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        print(f"분석 중 예기치 않은 오류 발생: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("=============== 익절 전략 분석 완료 ================")
    print("=" * 60)

if __name__ == '__main__':
    main() 