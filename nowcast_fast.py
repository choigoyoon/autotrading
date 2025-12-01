"""
나우캐스트 준수 백테스트 - 최적화 버전
미래 데이터 참조 완전 제거
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """OHLCV 데이터 로드"""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD 계산 (벡터화)"""
    close = df['close'].values
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow
    macd_signal = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def generate_labels(df, macd_hist):
    """H/L 라벨 생성 (나우캐스트 준수)"""
    labels = np.full(len(df), '', dtype=object)
    label_prices = np.full(len(df), np.nan)
    
    current_sign = None
    segment_start = 0
    
    for i in range(len(df)):
        if np.isnan(macd_hist[i]):
            continue
        
        sign = 1 if macd_hist[i] >= 0 else -1
        
        if current_sign is not None and sign != current_sign:
            segment = df.iloc[segment_start:i]
            
            if len(segment) > 0:
                if current_sign == 1:  # H
                    max_idx = segment['high'].idxmax()
                    labels[i] = 'H'
                    label_prices[i] = segment.loc[max_idx, 'high']
                else:  # L
                    min_idx = segment['low'].idxmin()
                    labels[i] = 'L'
                    label_prices[i] = segment.loc[min_idx, 'low']
            
            segment_start = i
        
        current_sign = sign
    
    return labels, label_prices


def run_backtest(df, tp=2.0, sl=2.0, interval=10, max_age=200, long_only=True,
                 use_fvg=False, dynamic_tp=False):
    """최적화된 백테스트"""
    
    # MACD
    _, _, macd_hist = calculate_macd(df)
    
    # 라벨
    labels, label_prices = generate_labels(df, macd_hist)
    
    # 인덱스 추출
    h_indices = np.where(labels == 'H')[0]
    l_indices = np.where(labels == 'L')[0]
    h_prices = {idx: label_prices[idx] for idx in h_indices}
    l_prices = {idx: label_prices[idx] for idx in l_indices}
    
    trades = []
    last_entry = -interval - 1
    start_idx = 100
    
    close = df['close'].values
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    
    for i in range(start_idx, len(df) - 50):
        if i - last_entry < interval:
            continue
        
        # 현재까지의 H/L
        recent_h = [idx for idx in h_indices if idx < i and idx >= i - max_age]
        recent_l = [idx for idx in l_indices if idx < i and idx >= i - max_age]
        
        # 하락추세선 돌파 체크
        breakout = None
        
        if len(recent_h) >= 2:
            h1_idx, h2_idx = recent_h[-2], recent_h[-1]
            h1_p, h2_p = h_prices[h1_idx], h_prices[h2_idx]
            
            if h2_p < h1_p:  # 하락 조건
                slope = (h2_p - h1_p) / (h2_idx - h1_idx)
                tl_curr = h1_p + slope * (i - h1_idx)
                tl_prev = h1_p + slope * (i - 1 - h1_idx)
                
                if close[i-1] <= tl_prev and close[i] > tl_curr:
                    breakout = 'long'
        
        if not breakout and not long_only and len(recent_l) >= 2:
            l1_idx, l2_idx = recent_l[-2], recent_l[-1]
            l1_p, l2_p = l_prices[l1_idx], l_prices[l2_idx]
            
            if l2_p > l1_p:  # 상승 조건
                slope = (l2_p - l1_p) / (l2_idx - l1_idx)
                tl_curr = l1_p + slope * (i - l1_idx)
                tl_prev = l1_p + slope * (i - 1 - l1_idx)
                
                if close[i-1] >= tl_prev and close[i] < tl_curr:
                    breakout = 'short'
        
        if not breakout:
            continue
        
        # FVG 체크
        has_fvg = i >= 2 and low[i] > high[i-2]
        
        if use_fvg and not has_fvg:
            continue
        
        # TP 설정
        actual_tp = (2.5 if has_fvg else 1.5) if dynamic_tp else tp
        
        # 진입
        entry_idx = i + 1
        entry_price = open_p[entry_idx]
        
        if breakout == 'long':
            tp_level = entry_price * (1 + actual_tp / 100)
            sl_level = entry_price * (1 - sl / 100)
        else:
            tp_level = entry_price * (1 - actual_tp / 100)
            sl_level = entry_price * (1 + sl / 100)
        
        # 결과
        result = 'TIMEOUT'
        exit_price = close[min(entry_idx + 49, len(df) - 1)]
        exit_idx = min(entry_idx + 49, len(df) - 1)
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if breakout == 'long':
                if high[j] >= tp_level:
                    result, exit_price, exit_idx = 'TP', tp_level, j
                    break
                if low[j] <= sl_level:
                    result, exit_price, exit_idx = 'SL', sl_level, j
                    break
            else:
                if low[j] <= tp_level:
                    result, exit_price, exit_idx = 'TP', tp_level, j
                    break
                if high[j] >= sl_level:
                    result, exit_price, exit_idx = 'SL', sl_level, j
                    break
        
        # PnL
        if breakout == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
        
        trades.append({
            'entry_idx': entry_idx,
            'entry_time': df.iloc[entry_idx]['datetime'],
            'entry_price': entry_price,
            'direction': breakout,
            'has_fvg': has_fvg,
            'tp_pct': actual_tp,
            'sl_pct': sl,
            'result': result,
            'exit_price': exit_price,
            'pnl': pnl,
            'hold_bars': exit_idx - entry_idx
        })
        
        last_entry = i
    
    return trades


def calc_stats(trades):
    """통계 계산"""
    if not trades:
        return {'total_trades': 0}
    
    pnls = [t['pnl'] for t in trades]
    results = [t['result'] for t in trades]
    
    return {
        'total_trades': len(trades),
        'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
        'avg_pnl': np.mean(pnls),
        'total_pnl': sum(pnls),
        'max_win': max(pnls),
        'max_loss': min(pnls),
        'tp_count': results.count('TP'),
        'sl_count': results.count('SL'),
        'timeout_count': results.count('TIMEOUT'),
    }


def main():
    print("=" * 70)
    print("나우캐스트 준수 백테스트 (미래 참조 완전 제거)")
    print("=" * 70)
    
    # 데이터 로드
    df = load_data('btc_15m_ohlcv.csv')
    print(f"\n데이터: {len(df)}봉")
    print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    # 라벨 통계
    _, _, macd_hist = calculate_macd(df)
    labels, _ = generate_labels(df, macd_hist)
    h_count = sum(1 for l in labels if l == 'H')
    l_count = sum(1 for l in labels if l == 'L')
    print(f"H 라벨: {h_count}개, L 라벨: {l_count}개")
    
    # 파라미터 테스트
    print("\n" + "=" * 70)
    print("파라미터 최적화 (14개 조합)")
    print("=" * 70)
    
    params = [
        (1.0, 1.0, 'TP1_SL1'),
        (1.5, 1.5, 'TP1.5_SL1.5'),
        (2.0, 2.0, 'TP2_SL2'),
        (2.5, 2.5, 'TP2.5_SL2.5'),
        (3.0, 3.0, 'TP3_SL3'),
        (1.0, 2.0, 'TP1_SL2'),
        (1.5, 2.0, 'TP1.5_SL2'),
        (1.0, 3.0, 'TP1_SL3'),
        (1.5, 3.0, 'TP1.5_SL3'),
        (2.0, 3.0, 'TP2_SL3'),
        (0.5, 1.0, 'TP0.5_SL1'),
        (0.5, 2.0, 'TP0.5_SL2'),
        (0.7, 1.5, 'TP0.7_SL1.5'),
        (0.7, 2.0, 'TP0.7_SL2'),
    ]
    
    results = []
    for i, (tp, sl, name) in enumerate(params, 1):
        trades = run_backtest(df, tp=tp, sl=sl, interval=10)
        stats = calc_stats(trades)
        
        if stats['total_trades'] > 0:
            net_pnl = stats['avg_pnl'] - 0.11
            results.append({
                'name': name,
                'tp': tp,
                'sl': sl,
                'trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'net_pnl': net_pnl,
                'total_pnl': stats['total_pnl'],
                'tp_count': stats['tp_count'],
                'sl_count': stats['sl_count']
            })
        
        print(f"[{i:2d}/{len(params)}] {name:12s}: 거래={stats['total_trades']:4d}, "
              f"승률={stats['win_rate']:5.1f}%, 평균={stats['avg_pnl']:+.3f}%, "
              f"총={stats['total_pnl']:+.1f}%")
    
    # 결과 정렬
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("상위 5개 (순수익 기준)")
    print("=" * 70)
    top5 = results_df.sort_values('net_pnl', ascending=False).head(5)
    print(top5[['name', 'trades', 'win_rate', 'avg_pnl', 'net_pnl', 'total_pnl']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("상위 5개 (승률 기준)")
    print("=" * 70)
    top_wr = results_df.sort_values('win_rate', ascending=False).head(5)
    print(top_wr[['name', 'trades', 'win_rate', 'avg_pnl', 'net_pnl', 'total_pnl']].to_string(index=False))
    
    # FVG 필터 테스트
    print("\n" + "=" * 70)
    print("FVG 필터 테스트")
    print("=" * 70)
    
    for tp, sl in [(2.0, 2.0), (1.5, 2.0), (1.0, 2.0)]:
        trades = run_backtest(df, tp=tp, sl=sl, interval=10, use_fvg=True)
        stats = calc_stats(trades)
        print(f"FVG TP{tp}_SL{sl}: 거래={stats['total_trades']}, "
              f"승률={stats['win_rate']:.1f}%, 평균={stats['avg_pnl']:.3f}%")
    
    # 동적 TP 테스트
    print("\n" + "=" * 70)
    print("동적 TP 테스트 (FVG시 2.5%, 아닐시 1.5%)")
    print("=" * 70)
    
    trades = run_backtest(df, tp=2.0, sl=2.0, interval=10, dynamic_tp=True)
    stats = calc_stats(trades)
    print(f"동적 TP: 거래={stats['total_trades']}, "
          f"승률={stats['win_rate']:.1f}%, 평균={stats['avg_pnl']:.3f}%")
    
    # 최적 설정 상세
    if len(results_df) > 0:
        best = results_df.sort_values('net_pnl', ascending=False).iloc[0]
        
        print("\n" + "=" * 70)
        print(f"최적 설정: {best['name']} (TP:{best['tp']}%, SL:{best['sl']}%)")
        print("=" * 70)
        
        trades = run_backtest(df, tp=best['tp'], sl=best['sl'], interval=10)
        stats = calc_stats(trades)
        
        total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
        months = total_days / 30
        
        print(f"\n세부 결과:")
        print(f"  총 거래: {stats['total_trades']}회")
        print(f"  승률: {stats['win_rate']:.1f}%")
        print(f"  평균 수익: {stats['avg_pnl']:.3f}%")
        print(f"  수수료 후: {stats['avg_pnl'] - 0.11:.3f}%")
        print(f"  총 수익: {stats['total_pnl']:.1f}%")
        print(f"  TP: {stats['tp_count']}, SL: {stats['sl_count']}, TIMEOUT: {stats['timeout_count']}")
        
        print(f"\n월간 추정 ({months:.1f}개월 기준):")
        print(f"  월 거래: {stats['total_trades'] / months:.1f}회")
        print(f"  월 수익: {stats['total_pnl'] / months:.1f}%")
        
        # 거래 저장
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('nowcast_trades_result.csv', index=False)
        print(f"\n거래 내역 저장: nowcast_trades_result.csv")
    
    # 기존 전략 비교
    print("\n" + "=" * 70)
    print("기존 전략 vs 나우캐스트 비교")
    print("=" * 70)
    print("""
기존 전략 (미래 참조 문제):
  - 승률: 84-90%
  - 월 수익: 14%+
  - 문제: 미래 H/L로 추세선 완성 → 실전 불가

나우캐스트 전략 (수정 완료):
  - 미래 참조 완전 제거
  - 현재 시점 데이터만 사용
  - 실전 적용 가능
""")
    
    if len(results_df) > 0:
        best_wr = results_df.sort_values('win_rate', ascending=False).iloc[0]
        best_pnl = results_df.sort_values('net_pnl', ascending=False).iloc[0]
        
        print(f"나우캐스트 최고 승률: {best_wr['win_rate']:.1f}% ({best_wr['name']})")
        print(f"나우캐스트 최고 수익: {best_pnl['net_pnl']:.3f}%/거래 ({best_pnl['name']})")
        
        # 월간 추정
        best_monthly = best_pnl['total_pnl'] / months
        print(f"나우캐스트 월간 추정: {best_monthly:.2f}% (기존 14% → {best_monthly:.1f}%)")
    
    # 결과 저장
    results_df.to_csv('nowcast_optimization_results.csv', index=False)
    print(f"\n최적화 결과 저장: nowcast_optimization_results.csv")
    
    print("\n" + "=" * 70)
    print("백테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
