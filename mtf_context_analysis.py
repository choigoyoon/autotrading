import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔍 MTF (Multiple Time Frame) 상황 판단 분석")
print("=" * 80)

# Load all timeframes
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

df_1h = pd.read_csv('btc_1h_ohlcv.csv')
df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
df_1h = df_1h.sort_values('datetime').reset_index(drop=True)

df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
df_4h = df_4h.sort_values('datetime').reset_index(drop=True)

# Load backtest results
trades = pd.read_csv('backtest_confirmation_space_results.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])

print(f"\n데이터 로드 완료:")
print(f"  - 15분봉: {len(df_15m):,}개")
print(f"  - 1시간봉: {len(df_1h):,}개")
print(f"  - 4시간봉: {len(df_4h):,}개")
print(f"  - 거래: {len(trades):,}개")

def get_mtf_context(entry_time, lookback_1h=10, lookback_4h=5):
    """진입 시점의 MTF 상황 판단"""
    
    # 1H context
    entry_1h_idx = df_1h[df_1h['datetime'] <= entry_time].index
    if len(entry_1h_idx) == 0:
        return None
    
    entry_1h_idx = entry_1h_idx[-1]
    candles_1h = df_1h.iloc[max(0, entry_1h_idx - lookback_1h + 1):entry_1h_idx + 1].copy()
    
    if len(candles_1h) < 2:
        return None
    
    # 4H context
    entry_4h_idx = df_4h[df_4h['datetime'] <= entry_time].index
    if len(entry_4h_idx) == 0:
        return None
    
    entry_4h_idx = entry_4h_idx[-1]
    candles_4h = df_4h.iloc[max(0, entry_4h_idx - lookback_4h + 1):entry_4h_idx + 1].copy()
    
    if len(candles_4h) < 2:
        return None
    
    # 1H 추세 판단
    closes_1h = candles_1h['close'].values
    highs_1h = candles_1h['high'].values
    lows_1h = candles_1h['low'].values
    
    # 1H 상승 추세 판단 (Higher Highs, Higher Lows)
    hh_1h = all(highs_1h[i] >= highs_1h[i-1] for i in range(-3, 0))
    hl_1h = all(lows_1h[i] >= lows_1h[i-1] for i in range(-3, 0))
    
    # 1H 하락 추세 판단
    lh_1h = all(highs_1h[i] <= highs_1h[i-1] for i in range(-3, 0))
    ll_1h = all(lows_1h[i] <= lows_1h[i-1] for i in range(-3, 0))
    
    if hh_1h and hl_1h:
        trend_1h = "상승추세"
    elif lh_1h and ll_1h:
        trend_1h = "하락추세"
    else:
        trend_1h = "횡보/전환"
    
    # 1H 모멘텀
    momentum_1h = ((closes_1h[-1] - closes_1h[0]) / closes_1h[0]) * 100
    
    # 1H 연속 상승/하락 캔들
    consecutive_up_1h = 0
    for i in range(len(closes_1h) - 1, 0, -1):
        if closes_1h[i] > closes_1h[i-1]:
            consecutive_up_1h += 1
        else:
            break
    
    consecutive_down_1h = 0
    for i in range(len(closes_1h) - 1, 0, -1):
        if closes_1h[i] < closes_1h[i-1]:
            consecutive_down_1h += 1
        else:
            break
    
    # 4H 추세 판단
    closes_4h = candles_4h['close'].values
    highs_4h = candles_4h['high'].values
    lows_4h = candles_4h['low'].values
    
    hh_4h = all(highs_4h[i] >= highs_4h[i-1] for i in range(-2, 0)) if len(highs_4h) >= 2 else False
    hl_4h = all(lows_4h[i] >= lows_4h[i-1] for i in range(-2, 0)) if len(lows_4h) >= 2 else False
    
    lh_4h = all(highs_4h[i] <= highs_4h[i-1] for i in range(-2, 0)) if len(highs_4h) >= 2 else False
    ll_4h = all(lows_4h[i] <= lows_4h[i-1] for i in range(-2, 0)) if len(lows_4h) >= 2 else False
    
    if hh_4h and hl_4h:
        trend_4h = "상승추세"
    elif lh_4h and ll_4h:
        trend_4h = "하락추세"
    else:
        trend_4h = "횡보/전환"
    
    # 4H 모멘텀
    momentum_4h = ((closes_4h[-1] - closes_4h[0]) / closes_4h[0]) * 100
    
    # MTF 정렬 (상승 정렬 = 4H 상승 + 1H 상승)
    mtf_aligned_up = (trend_4h == "상승추세") and (trend_1h == "상승추세")
    mtf_aligned_down = (trend_4h == "하락추세") and (trend_1h == "하락추세")
    
    # 역추세 진입 (4H 하락인데 15분 롱 진입)
    counter_trend = (trend_4h == "하락추세") or (trend_1h == "하락추세")
    
    return {
        'trend_1h': trend_1h,
        'trend_4h': trend_4h,
        'momentum_1h': momentum_1h,
        'momentum_4h': momentum_4h,
        'consecutive_up_1h': consecutive_up_1h,
        'consecutive_down_1h': consecutive_down_1h,
        'mtf_aligned_up': mtf_aligned_up,
        'mtf_aligned_down': mtf_aligned_down,
        'counter_trend': counter_trend,
        'current_1h_close': closes_1h[-1],
        'current_4h_close': closes_4h[-1],
    }

# Analyze all trades with MTF context
print("\n" + "=" * 80)
print("각 거래의 MTF 상황 분석 중...")
print("=" * 80)

mtf_results = []
for idx, trade in trades.iterrows():
    if idx % 50 == 0:
        print(f"진행: {idx}/{len(trades)}")
    
    context = get_mtf_context(trade['entry_time'])
    if context:
        mtf_results.append({
            'entry_time': trade['entry_time'],
            'exit_reason': trade['exit_reason'],
            'power_score': trade['power_score'],
            'pnl_pct': trade['pnl_pct'],
            'year': trade['year'],
            **context
        })

mtf_df = pd.DataFrame(mtf_results)

print(f"\n총 {len(mtf_df)}개 거래의 MTF 상황 분석 완료")

# Analysis by exit reason and MTF context
print("\n" + "=" * 80)
print("1단계: MTF 정렬 여부에 따른 성공률")
print("=" * 80)

for exit_reason in ['SL', 'TP1_Breakeven', 'TP2_Full']:
    subset = mtf_df[mtf_df['exit_reason'] == exit_reason]
    if len(subset) == 0:
        continue
    
    print(f"\n[{exit_reason}] {len(subset)}개:")
    print(f"  - MTF 상승 정렬: {(subset['mtf_aligned_up'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 역추세 진입: {(subset['counter_trend'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 1H 상승추세: {(subset['trend_1h'] == '상승추세').sum() / len(subset) * 100:.1f}%")
    print(f"  - 1H 하락추세: {(subset['trend_1h'] == '하락추세').sum() / len(subset) * 100:.1f}%")
    print(f"  - 4H 상승추세: {(subset['trend_4h'] == '상승추세').sum() / len(subset) * 100:.1f}%")
    print(f"  - 4H 하락추세: {(subset['trend_4h'] == '하락추세').sum() / len(subset) * 100:.1f}%")
    print(f"  - 평균 1H 모멘텀: {subset['momentum_1h'].mean():.2f}%")
    print(f"  - 평균 4H 모멘텀: {subset['momentum_4h'].mean():.2f}%")

print("\n" + "=" * 80)
print("2단계: MTF 정렬 시 vs 비정렬 시 성과 비교")
print("=" * 80)

# MTF aligned UP
mtf_aligned = mtf_df[mtf_df['mtf_aligned_up'] == True]
mtf_not_aligned = mtf_df[mtf_df['mtf_aligned_up'] == False]

print(f"\n✅ MTF 상승 정렬 ({len(mtf_aligned)}개):")
print(f"  - TP2 성공률: {(mtf_aligned['exit_reason'] == 'TP2_Full').sum() / len(mtf_aligned) * 100:.1f}%")
print(f"  - SL 비율: {(mtf_aligned['exit_reason'] == 'SL').sum() / len(mtf_aligned) * 100:.1f}%")
print(f"  - 평균 PnL: {mtf_aligned['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {mtf_aligned['pnl_pct'].sum():.2f}%")

print(f"\n❌ MTF 비정렬 ({len(mtf_not_aligned)}개):")
print(f"  - TP2 성공률: {(mtf_not_aligned['exit_reason'] == 'TP2_Full').sum() / len(mtf_not_aligned) * 100:.1f}%")
print(f"  - SL 비율: {(mtf_not_aligned['exit_reason'] == 'SL').sum() / len(mtf_not_aligned) * 100:.1f}%")
print(f"  - 평균 PnL: {mtf_not_aligned['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {mtf_not_aligned['pnl_pct'].sum():.2f}%")

print("\n" + "=" * 80)
print("3단계: 1H 추세별 성과")
print("=" * 80)

for trend in ['상승추세', '하락추세', '횡보/전환']:
    subset = mtf_df[mtf_df['trend_1h'] == trend]
    if len(subset) == 0:
        continue
    
    print(f"\n1H {trend} ({len(subset)}개):")
    print(f"  - TP2 성공률: {(subset['exit_reason'] == 'TP2_Full').sum() / len(subset) * 100:.1f}%")
    print(f"  - SL 비율: {(subset['exit_reason'] == 'SL').sum() / len(subset) * 100:.1f}%")
    print(f"  - 평균 PnL: {subset['pnl_pct'].mean():.2f}%")
    print(f"  - 총 PnL: {subset['pnl_pct'].sum():.2f}%")

print("\n" + "=" * 80)
print("4단계: 4H 추세별 성과")
print("=" * 80)

for trend in ['상승추세', '하락추세', '횡보/전환']:
    subset = mtf_df[mtf_df['trend_4h'] == trend]
    if len(subset) == 0:
        continue
    
    print(f"\n4H {trend} ({len(subset)}개):")
    print(f"  - TP2 성공률: {(subset['exit_reason'] == 'TP2_Full').sum() / len(subset) * 100:.1f}%")
    print(f"  - SL 비율: {(subset['exit_reason'] == 'SL').sum() / len(subset) * 100:.1f}%")
    print(f"  - 평균 PnL: {subset['pnl_pct'].mean():.2f}%")
    print(f"  - 총 PnL: {subset['pnl_pct'].sum():.2f}%")

print("\n" + "=" * 80)
print("5단계: 역추세 진입 vs 순추세 진입")
print("=" * 80)

counter = mtf_df[mtf_df['counter_trend'] == True]
with_trend = mtf_df[mtf_df['counter_trend'] == False]

print(f"\n❌ 역추세 진입 (4H or 1H 하락 중 롱) ({len(counter)}개):")
print(f"  - TP2 성공률: {(counter['exit_reason'] == 'TP2_Full').sum() / len(counter) * 100:.1f}%")
print(f"  - SL 비율: {(counter['exit_reason'] == 'SL').sum() / len(counter) * 100:.1f}%")
print(f"  - 평균 PnL: {counter['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {counter['pnl_pct'].sum():.2f}%")

print(f"\n✅ 순추세 진입 ({len(with_trend)}개):")
print(f"  - TP2 성공률: {(with_trend['exit_reason'] == 'TP2_Full').sum() / len(with_trend) * 100:.1f}%")
print(f"  - SL 비율: {(with_trend['exit_reason'] == 'SL').sum() / len(with_trend) * 100:.1f}%")
print(f"  - 평균 PnL: {with_trend['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {with_trend['pnl_pct'].sum():.2f}%")

# Save results
mtf_df.to_csv('mtf_context_analysis.csv', index=False)

print("\n" + "=" * 80)
print("✅ MTF 분석 완료")
print("=" * 80)
print(f"결과 저장: mtf_context_analysis.csv")
