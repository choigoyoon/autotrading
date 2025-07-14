import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools

# --- 설정 ---
PROJECT_ROOT = Path(__file__).parent.parent
# 분석할 원본 MACD 라벨 데이터 (Lookahead Bias가 있는 데이터)
LABELS_PATH = PROJECT_ROOT / 'data' / 'labels' / 'btc_usdt_kst_merged_labels.parquet'
# 1분봉 가격 데이터
OHLCV_PATH = PROJECT_ROOT / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet'
# 분석할 기간 (분)
LOOKAHEAD_WINDOW = 240  # L 라벨 이후 4시간 동안의 가격 움직임을 분석

def load_data():
    """분석에 필요한 라벨과 가격 데이터를 로드합니다."""
    print("--- 데이터 로딩 ---")
    if not LABELS_PATH.exists() or not OHLCV_PATH.exists():
        raise FileNotFoundError("오류: 필수 데이터 파일(라벨 또는 OHLCV)을 찾을 수 없습니다.")
        
    df_labels = pd.read_parquet(LABELS_PATH)
    # L값 (매수 신호, label=1)만 필터링
    df_buy_signals = df_labels[df_labels['label'] == 1].copy()
    
    df_ohlcv = pd.read_parquet(OHLCV_PATH)
    
    print(f"매수 신호(L값) 데이터 로드 완료: {len(df_buy_signals)}개")
    print(f"가격 데이터 로드 완료: {len(df_ohlcv)}개")
    return df_buy_signals, df_ohlcv

def analyze_profit_distribution(df_buy_signals, df_ohlcv):
    """
    매수 신호 이후의 최대 수익률 분포를 분석하고, 분할 익절 목표가를 계산합니다.
    """
    print(f"\n--- [1/3] {LOOKAHEAD_WINDOW}분 내 최대 수익률 분포 분석 ---")
    
    max_profits = []
    
    for signal_time, row in tqdm(df_buy_signals.iterrows(), total=len(df_buy_signals), desc="수익률 분포 분석 중"):
        # 신호 시점의 가격 데이터 확인
        if signal_time not in df_ohlcv.index:
            continue
            
        entry_price = df_ohlcv.loc[signal_time, 'close']
        
        # 분석할 미래 데이터 윈도우 설정
        end_time = signal_time + pd.Timedelta(minutes=LOOKAHEAD_WINDOW)
        future_candles = df_ohlcv.loc[signal_time:end_time]
        
        if future_candles.empty:
            continue
            
        # 윈도우 내 최고가
        max_price_in_window = future_candles['high'].max()
        
        # 최대 가능 수익률 계산
        max_profit_pct = (max_price_in_window / entry_price - 1) * 100
        max_profits.append(max_profit_pct)

    if not max_profits:
        print("분석할 수익률 데이터가 없습니다.")
        return None

    s_profits = pd.Series(max_profits)
    
    # 수익률 분포를 기반으로 분할 익절 목표가 설정 (25%, 50%, 75% 분위수)
    exit_targets = {
        'target1': s_profits.quantile(0.25),
        'target2': s_profits.quantile(0.50),
        'target3': s_profits.quantile(0.75),
    }

    print("\n=== 수익률 분포 분석 결과 ===")
    print(s_profits.describe(percentiles=[.1, .25, .5, .75, .9, .99]))
    
    print("\n=== 데이터 기반 분할 익절 목표가 (수익률 %) ===")
    print(f"  - 1차 익절 목표 (25% 분위): {exit_targets['target1']:.2f}%")
    print(f"  - 2차 익절 목표 (50% 분위): {exit_targets['target2']:.2f}%")
    print(f"  - 3차 익절 목표 (75% 분위): {exit_targets['target3']:.2f}%")
    
    return exit_targets
    
def backtest_partial_exit_strategy(df_buy_signals, df_ohlcv, exit_targets, exit_ratios):
    """
    주어진 익절 목표가와 비율에 따라 분할 익절 전략을 백테스팅합니다.
    """
    total_pnl = 0
    
    # 신호가 많은 경우, 샘플링하여 백테스트 속도 향상
    if len(df_buy_signals) > 10000:
        df_buy_signals_sampled = df_buy_signals.sample(n=10000, random_state=42)
    else:
        df_buy_signals_sampled = df_buy_signals
        
    for signal_time, row in df_buy_signals_sampled.iterrows():
        if signal_time not in df_ohlcv.index:
            continue
            
        entry_price = df_ohlcv.loc[signal_time, 'close']
        end_time = signal_time + pd.Timedelta(minutes=LOOKAHEAD_WINDOW)
        
        position_size = 1.0  # 초기 포지션 크기
        trade_pnl = 0
        
        targets = sorted(exit_targets.values())
        ratios = list(exit_ratios) # 수정 가능한 리스트로 복사

        for i in range(3):
            if position_size < 1e-6:
                break

            target_price = entry_price * (1 + targets[i] / 100)
            
            # 남은 비율의 합을 기준으로 현재 익절 크기 계산
            remaining_ratios_sum = sum(ratios)
            if remaining_ratios_sum < 1e-6:
                break
            
            exit_ratio_of_remaining = ratios.pop(0) / remaining_ratios_sum
            exit_size = position_size * exit_ratio_of_remaining

            # 해당 target을 달성했는지 확인
            future_candles = df_ohlcv.loc[signal_time:end_time]
            hit_candles = future_candles[future_candles['high'] >= target_price]
            
            if not hit_candles.empty:
                # 목표가 도달, 해당 비율만큼 익절
                trade_pnl += ((target_price / entry_price) - 1) * exit_size
                position_size -= exit_size
            
        total_pnl += trade_pnl
        
    return total_pnl / len(df_buy_signals_sampled) if len(df_buy_signals_sampled) > 0 else 0

def find_optimal_exit_strategy(df_buy_signals, df_ohlcv, exit_targets):
    """
    그리드 서치를 통해 최적의 분할 익절 비율을 탐색합니다.
    """
    print("\n--- [2/3] 최적 분할 익절 비율 탐색 ---")
    
    # 테스트할 분할 익절 비율 조합
    ratio_grid = [p for p in itertools.product([10, 20, 30, 40, 50, 60, 70], repeat=3) if sum(p) == 100]
    
    best_performance = -np.inf
    best_ratios = None
    
    results = []

    for ratios in tqdm(ratio_grid, desc="익절 비율 최적화 중"):
        exit_ratios_normalized = [r/100 for r in ratios]
        performance = backtest_partial_exit_strategy(df_buy_signals, df_ohlcv, exit_targets, exit_ratios_normalized)
        results.append({'ratios': ratios, 'performance': performance})
        
        if performance > best_performance:
            best_performance = performance
            best_ratios = ratios
            
    print("\n--- [3/3] 최적 익절 전략 결과 ---")
    if best_ratios:
        print(f"최고 성과: 평균 {best_performance:.4f}% 수익")
        print(f"최적 익절 비율 (1차, 2차, 3차): {best_ratios[0]}% : {best_ratios[1]}% : {best_ratios[2]}%")
    else:
        print("최적 전략을 찾지 못했습니다.")

def main():
    """데이터 기반 익절 전략 분석 메인 함수"""
    print("======================================================")
    print("======= 데이터 기반 익절 전략 분석 도구 시작 =======")
    print("======================================================")
    
    try:
        df_buy_signals, df_ohlcv = load_data()
        
        # 1. 수익률 분포 분석 및 익절 목표가 설정
        exit_targets = analyze_profit_distribution(df_buy_signals, df_ohlcv)
        
        if exit_targets:
            # 2. 최적 익절 비율 탐색 및 백테스팅
            find_optimal_exit_strategy(df_buy_signals, df_ohlcv, exit_targets)
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"분석 중 예기치 않은 오류 발생: {e}")

    print("\n======================================================")
    print("=============== 익절 전략 분석 완료 ================")
    print("======================================================")

if __name__ == '__main__':
    main() 