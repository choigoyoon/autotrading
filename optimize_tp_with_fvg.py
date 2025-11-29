"""
FVG 기반 TP 최적화
- 기존 매매법 (추세선 돌파) + FVG 필터
- FVG 있을 때 vs 없을 때 최적 TP/SL 비교
"""

import pandas as pd
import numpy as np
from itertools import product

print("="*60)
print("FVG 기반 TP 최적화")
print("="*60)

# 데이터 로드
print("\n데이터 로드 중...")
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

breakouts = pd.read_csv('output_phase4_breakouts.csv')
trendline_breakouts = breakouts[breakouts['type'].isin(['trendline_up', 'trendline_down'])].copy()

print(f"총 돌파: {len(trendline_breakouts):,}개")

# FVG 감지
def detect_fvg(df):
    """FVG 감지"""
    fvg_list = []

    for i in range(2, len(df)):
        candle_1 = df.iloc[i-2]
        candle_3 = df.iloc[i]

        # Bullish FVG
        if candle_1['high'] < candle_3['low']:
            gap_size = (candle_3['low'] - candle_1['high']) / candle_3['close'] * 100
            fvg_list.append({
                'idx': i-1,
                'type': 'bullish',
                'gap_bottom': candle_1['high'],
                'gap_top': candle_3['low'],
                'gap_size': gap_size,
            })

        # Bearish FVG
        elif candle_1['low'] > candle_3['high']:
            gap_size = (candle_1['low'] - candle_3['high']) / candle_3['close'] * 100
            fvg_list.append({
                'idx': i-1,
                'type': 'bearish',
                'gap_bottom': candle_3['high'],
                'gap_top': candle_1['low'],
                'gap_size': gap_size,
            })

    return pd.DataFrame(fvg_list)

print("\nFVG 감지 중...")
fvg_df = detect_fvg(df)
print(f"FVG: {len(fvg_df):,}개")

# 돌파 + FVG 매칭
def has_nearby_fvg(break_idx, direction, fvg_df, lookback=20):
    """
    돌파 근처에 FVG가 있는지 확인
    lookback: 돌파 이전 N봉 이내
    """

    # 돌파 이전 FVG 찾기
    before_fvgs = fvg_df[
        (fvg_df['idx'] >= break_idx - lookback) &
        (fvg_df['idx'] < break_idx)
    ]

    if len(before_fvgs) == 0:
        return False, None

    # 방향에 맞는 FVG
    if direction == 'long':
        matching = before_fvgs[before_fvgs['type'] == 'bullish']
    else:
        matching = before_fvgs[before_fvgs['type'] == 'bearish']

    if len(matching) == 0:
        return False, None

    # 가장 가까운 FVG
    nearest = matching.iloc[-1]

    return True, {
        'gap_size': nearest['gap_size'],
        'distance': break_idx - nearest['idx']
    }

print("\n돌파-FVG 매칭 중...")
fvg_flags = []

for idx, breakout in trendline_breakouts.iterrows():
    break_idx = breakout['break_idx']
    direction = 'long' if breakout['type'] == 'trendline_up' else 'short'

    has_fvg, fvg_info = has_nearby_fvg(break_idx, direction, fvg_df, lookback=20)

    fvg_flags.append({
        'has_fvg': has_fvg,
        'fvg_size': fvg_info['gap_size'] if has_fvg else 0,
        'fvg_distance': fvg_info['distance'] if has_fvg else 0,
    })

fvg_flags_df = pd.DataFrame(fvg_flags)
trendline_breakouts['has_fvg'] = fvg_flags_df['has_fvg'].fillna(False)
trendline_breakouts['fvg_size'] = fvg_flags_df['fvg_size'].fillna(0)

print(f"\nFVG 있는 돌파: {trendline_breakouts['has_fvg'].sum():,}개 ({trendline_breakouts['has_fvg'].sum() / len(trendline_breakouts) * 100:.1f}%)")
print(f"FVG 없는 돌파: {(trendline_breakouts['has_fvg'] == False).sum():,}개")

# 백테스트 함수
def backtest_with_params(breakouts_subset, df, tp_pct, sl_pct):
    """특정 파라미터로 백테스트"""

    results = []

    for idx, breakout in breakouts_subset.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        max_idx = min(break_idx + 200, len(df) - 1)
        if max_idx <= break_idx:
            continue

        window = df.iloc[break_idx:max_idx+1]
        break_price = window.iloc[0]['close']

        # TP/SL 레벨
        if direction == 'long':
            tp_level = break_price * (1 + tp_pct / 100)
            sl_level = break_price * (1 - sl_pct / 100)

            tp_hit = (window['high'] >= tp_level).any()
            sl_hit = (window['low'] <= sl_level).any()

            if tp_hit and sl_hit:
                tp_idx = window[window['high'] >= tp_level].index[0]
                sl_idx = window[window['low'] <= sl_level].index[0]
                pnl = tp_pct if tp_idx < sl_idx else -sl_pct
            elif tp_hit:
                pnl = tp_pct
            elif sl_hit:
                pnl = -sl_pct
            else:
                final_price = window.iloc[-1]['close']
                pnl = (final_price - break_price) / break_price * 100

        else:  # short
            tp_level = break_price * (1 - tp_pct / 100)
            sl_level = break_price * (1 + sl_pct / 100)

            tp_hit = (window['low'] <= tp_level).any()
            sl_hit = (window['high'] >= sl_level).any()

            if tp_hit and sl_hit:
                tp_idx = window[window['low'] <= tp_level].index[0]
                sl_idx = window[window['high'] >= sl_level].index[0]
                pnl = tp_pct if tp_idx < sl_idx else -sl_pct
            elif tp_hit:
                pnl = tp_pct
            elif sl_hit:
                pnl = -sl_pct
            else:
                final_price = window.iloc[-1]['close']
                pnl = (break_price - final_price) / break_price * 100

        results.append(pnl)

    if len(results) == 0:
        return None

    results_series = pd.Series(results)

    return {
        'count': len(results),
        'win_rate': (results_series > 0).sum() / len(results) * 100,
        'avg_pnl': results_series.mean(),
        'total_return': results_series.sum(),
        'sharpe': results_series.mean() / results_series.std() if results_series.std() > 0 else 0,
    }

# 파라미터 그리드
param_grid = {
    'tp_pct': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    'sl_pct': [1.0, 1.5, 2.0, 2.5],
}

print("\n" + "="*60)
print("최적 TP/SL 탐색")
print("="*60)

# 1. FVG 있는 경우
print("\n[1] FVG 있는 돌파 최적화")
print("-"*60)

with_fvg = trendline_breakouts[trendline_breakouts['has_fvg'] == True].copy()
print(f"거래 수: {len(with_fvg):,}개")

best_with_fvg = None
best_score_fvg = -999999
results_fvg = []

for tp, sl in product(param_grid['tp_pct'], param_grid['sl_pct']):
    result = backtest_with_params(with_fvg, df, tp, sl)

    if result is None:
        continue

    # 점수: 평균 수익 × 승률
    score = result['avg_pnl'] * result['win_rate']

    results_fvg.append({
        'tp': tp,
        'sl': sl,
        'score': score,
        **result
    })

    if score > best_score_fvg:
        best_score_fvg = score
        best_with_fvg = {'tp': tp, 'sl': sl, **result}

print(f"\n✅ 최적 파라미터 (FVG 있음):")
print(f"   TP: {best_with_fvg['tp']:.1f}%")
print(f"   SL: {best_with_fvg['sl']:.1f}%")
print(f"   승률: {best_with_fvg['win_rate']:.1f}%")
print(f"   평균 수익: {best_with_fvg['avg_pnl']:.3f}%")
print(f"   총 수익: {best_with_fvg['total_return']:.1f}%")

# 상위 3개
results_fvg_df = pd.DataFrame(results_fvg)
top3_fvg = results_fvg_df.nlargest(3, 'score')
print(f"\n상위 3개:")
for idx, row in top3_fvg.iterrows():
    print(f"   TP:{row['tp']:.1f}% SL:{row['sl']:.1f}% → 수익:{row['avg_pnl']:.3f}% 승률:{row['win_rate']:.1f}%")

# 2. FVG 없는 경우
print("\n[2] FVG 없는 돌파 최적화")
print("-"*60)

without_fvg = trendline_breakouts[trendline_breakouts['has_fvg'] == False].copy()
print(f"거래 수: {len(without_fvg):,}개")

best_without_fvg = None
best_score_no_fvg = -999999
results_no_fvg = []

for tp, sl in product(param_grid['tp_pct'], param_grid['sl_pct']):
    result = backtest_with_params(without_fvg, df, tp, sl)

    if result is None:
        continue

    score = result['avg_pnl'] * result['win_rate']

    results_no_fvg.append({
        'tp': tp,
        'sl': sl,
        'score': score,
        **result
    })

    if score > best_score_no_fvg:
        best_score_no_fvg = score
        best_without_fvg = {'tp': tp, 'sl': sl, **result}

print(f"\n✅ 최적 파라미터 (FVG 없음):")
print(f"   TP: {best_without_fvg['tp']:.1f}%")
print(f"   SL: {best_without_fvg['sl']:.1f}%")
print(f"   승률: {best_without_fvg['win_rate']:.1f}%")
print(f"   평균 수익: {best_without_fvg['avg_pnl']:.3f}%")
print(f"   총 수익: {best_without_fvg['total_return']:.1f}%")

# 상위 3개
results_no_fvg_df = pd.DataFrame(results_no_fvg)
top3_no_fvg = results_no_fvg_df.nlargest(3, 'score')
print(f"\n상위 3개:")
for idx, row in top3_no_fvg.iterrows():
    print(f"   TP:{row['tp']:.1f}% SL:{row['sl']:.1f}% → 수익:{row['avg_pnl']:.3f}% 승률:{row['win_rate']:.1f}%")

# 3. 비교
print("\n" + "="*60)
print("📊 FVG 유무 비교")
print("="*60)

comparison = pd.DataFrame({
    '항목': ['TP', 'SL', '승률', '평균 수익', '총 수익', 'Sharpe'],
    'FVG 있음': [
        f"{best_with_fvg['tp']:.1f}%",
        f"{best_with_fvg['sl']:.1f}%",
        f"{best_with_fvg['win_rate']:.1f}%",
        f"{best_with_fvg['avg_pnl']:.3f}%",
        f"{best_with_fvg['total_return']:.0f}%",
        f"{best_with_fvg['sharpe']:.3f}",
    ],
    'FVG 없음': [
        f"{best_without_fvg['tp']:.1f}%",
        f"{best_without_fvg['sl']:.1f}%",
        f"{best_without_fvg['win_rate']:.1f}%",
        f"{best_without_fvg['avg_pnl']:.3f}%",
        f"{best_without_fvg['total_return']:.0f}%",
        f"{best_without_fvg['sharpe']:.3f}",
    ],
})

print("\n" + comparison.to_string(index=False))

# 차이 분석
tp_diff = best_with_fvg['tp'] - best_without_fvg['tp']
win_rate_diff = best_with_fvg['win_rate'] - best_without_fvg['win_rate']
pnl_diff = best_with_fvg['avg_pnl'] - best_without_fvg['avg_pnl']

print("\n" + "="*60)
print("💡 핵심 발견")
print("="*60)

print(f"\n1. TP 차이:")
if tp_diff > 0.3:
    print(f"   ✅ FVG 있을 때 TP {tp_diff:.1f}%p 더 높게 설정 가능!")
    print(f"   → {best_with_fvg['tp']:.1f}% vs {best_without_fvg['tp']:.1f}%")
elif tp_diff < -0.3:
    print(f"   ⚠️ FVG 없을 때 TP {abs(tp_diff):.1f}%p 더 높게 설정")
else:
    print(f"   ≈ TP 차이 미미 ({tp_diff:.1f}%p)")

print(f"\n2. 승률 차이:")
if win_rate_diff > 2.0:
    print(f"   ✅ FVG 있을 때 승률 {win_rate_diff:.1f}%p 더 높음")
elif win_rate_diff < -2.0:
    print(f"   ⚠️ FVG 없을 때 승률 {abs(win_rate_diff):.1f}%p 더 높음")
else:
    print(f"   ≈ 승률 차이 미미 ({win_rate_diff:.1f}%p)")

print(f"\n3. 평균 수익 차이:")
print(f"   FVG 있음: {best_with_fvg['avg_pnl']:.3f}%")
print(f"   FVG 없음: {best_without_fvg['avg_pnl']:.3f}%")
print(f"   차이: {pnl_diff:+.3f}%")

# 4. 통합 전략 vs FVG 차별화 전략
print("\n" + "="*60)
print("📈 성과 개선 효과")
print("="*60)

# 기존 통합 전략 (TP 2.0%, SL 1.0%)
unified_result = backtest_with_params(trendline_breakouts, df, 2.0, 1.0)

# FVG 차별화 전략
fvg_enhanced_results = []

# FVG 있는 돌파
for idx, breakout in with_fvg.iterrows():
    result = backtest_with_params(
        with_fvg.iloc[[idx]],
        df,
        best_with_fvg['tp'],
        best_with_fvg['sl']
    )
    if result:
        fvg_enhanced_results.append(result['avg_pnl'])

# FVG 없는 돌파
for idx, breakout in without_fvg.iterrows():
    result = backtest_with_params(
        without_fvg.iloc[[idx]],
        df,
        best_without_fvg['tp'],
        best_without_fvg['sl']
    )
    if result:
        fvg_enhanced_results.append(result['avg_pnl'])

fvg_enhanced_avg = np.mean(fvg_enhanced_results)
fvg_enhanced_total = np.sum(fvg_enhanced_results)

print(f"\n기존 통합 전략 (TP:{unified_result['tp']}%, SL:{unified_result['sl']}%):")
print(f"   승률: {unified_result['win_rate']:.1f}%")
print(f"   평균 수익: {unified_result['avg_pnl']:.3f}%")
print(f"   총 수익: {unified_result['total_return']:.0f}%")

print(f"\nFVG 차별화 전략:")
print(f"   승률: (가중평균 계산 필요)")
print(f"   평균 수익: {fvg_enhanced_avg:.3f}%")
print(f"   총 수익: {fvg_enhanced_total:.0f}%")

improvement = fvg_enhanced_avg - unified_result['avg_pnl']
improvement_pct = improvement / unified_result['avg_pnl'] * 100

print(f"\n개선 효과:")
print(f"   평균 수익: {improvement:+.3f}%p ({improvement_pct:+.1f}%)")
print(f"   총 수익: {fvg_enhanced_total - unified_result['total_return']:+.0f}%")

# 5. 실전 가이드
print("\n" + "="*60)
print("🎯 실전 적용 가이드")
print("="*60)

print(f"""
추세선 돌파 전략 + FVG 필터:

1. FVG 확인:
   - 돌파 이전 20봉 내 FVG 존재 확인
   - 방향 일치 확인 (롱이면 Bullish FVG)

2. FVG 있는 돌파:
   ✅ TP: {best_with_fvg['tp']:.1f}%
   ✅ SL: {best_with_fvg['sl']:.1f}%
   ✅ 승률: {best_with_fvg['win_rate']:.1f}%
   → 공격적 설정

3. FVG 없는 돌파:
   ✅ TP: {best_without_fvg['tp']:.1f}%
   ✅ SL: {best_without_fvg['sl']:.1f}%
   ✅ 승률: {best_without_fvg['win_rate']:.1f}%
   → 보수적 설정

4. 예상 개선:
   평균 수익: {improvement:+.3f}%p
   총 수익: {improvement_pct:+.1f}% 향상
""")

print("\n분석 완료!")

# 저장
summary_df = pd.DataFrame({
    'Type': ['With FVG', 'Without FVG', 'Unified'],
    'TP': [best_with_fvg['tp'], best_without_fvg['tp'], 2.0],
    'SL': [best_with_fvg['sl'], best_without_fvg['sl'], 1.0],
    'Win_Rate': [best_with_fvg['win_rate'], best_without_fvg['win_rate'], unified_result['win_rate']],
    'Avg_PnL': [best_with_fvg['avg_pnl'], best_without_fvg['avg_pnl'], unified_result['avg_pnl']],
})

summary_df.to_csv('fvg_tp_optimization.csv', index=False)
print("\n저장: fvg_tp_optimization.csv")
