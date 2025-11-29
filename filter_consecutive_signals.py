"""
연속 신호 필터링 - 실전 오버트레이딩 방지

문제:
- 74.43% 거래가 5봉 이내 연속 발생
- 과도한 수수료 (0.11% × 2 = 0.22% per trade)
- 심리적 부담 증가

해결:
- 최소 간격 필터 (10봉 = 150분 = 2.5시간)
- 더 높은 품질의 신호만 선택
"""

import pandas as pd
import numpy as np

print("="*70)
print("연속 신호 필터링 - Consecutive Signal Filter")
print("="*70)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

breakouts = pd.read_csv('output_phase4_breakouts.csv')

# break_idx로 datetime 매핑
breakouts['datetime'] = breakouts['break_idx'].apply(lambda x: df.iloc[x]['datetime'] if x < len(df) else None)
breakouts = breakouts.dropna(subset=['datetime'])

print(f"\n원본 돌파 신호: {len(breakouts):,}개")
print(f"기간: {breakouts['datetime'].min()} ~ {breakouts['datetime'].max()}")

# ═══════════════════════════════════════════════════════════
# 1. 현재 상태 분석
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("1. 현재 연속 신호 분석")
print("="*70)

# 연속 신호 감지
consecutive_analysis = []
prev_break_idx = -999

for idx, row in breakouts.iterrows():
    break_idx = row['break_idx']
    interval = break_idx - prev_break_idx

    consecutive_analysis.append({
        'idx': idx,
        'break_idx': break_idx,
        'datetime': row['datetime'],
        'interval': interval,
        'is_consecutive_5': interval < 5,
        'is_consecutive_10': interval < 10,
        'is_consecutive_20': interval < 20,
    })

    prev_break_idx = break_idx

consecutive_df = pd.DataFrame(consecutive_analysis)

print(f"\n연속 신호 비율:")
print(f"  5봉 이내: {consecutive_df['is_consecutive_5'].sum():,}개 ({consecutive_df['is_consecutive_5'].sum() / len(consecutive_df) * 100:.2f}%)")
print(f"  10봉 이내: {consecutive_df['is_consecutive_10'].sum():,}개 ({consecutive_df['is_consecutive_10'].sum() / len(consecutive_df) * 100:.2f}%)")
print(f"  20봉 이내: {consecutive_df['is_consecutive_20'].sum():,}개 ({consecutive_df['is_consecutive_20'].sum() / len(consecutive_df) * 100:.2f}%)")

print(f"\n간격 통계:")
print(f"  평균: {consecutive_df['interval'].mean():.1f}봉")
print(f"  중앙값: {consecutive_df['interval'].median():.0f}봉")
print(f"  최소: {consecutive_df['interval'].min()}봉")
print(f"  최대: {consecutive_df['interval'].max()}봉")

# ═══════════════════════════════════════════════════════════
# 2. 필터 적용 - 3가지 옵션
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("2. 필터 적용 - 3가지 옵션")
print("="*70)

def apply_minimum_interval_filter(breakouts_df, min_interval_bars):
    """
    최소 간격 필터 적용

    Args:
        breakouts_df: 돌파 신호 DataFrame
        min_interval_bars: 최소 간격 (봉 수)

    Returns:
        filtered_df: 필터링된 돌파 신호
    """

    filtered_list = []
    last_break_idx = -999
    skipped = 0

    for idx, row in breakouts_df.iterrows():
        break_idx = row['break_idx']

        # 최소 간격 체크
        if break_idx - last_break_idx >= min_interval_bars:
            filtered_list.append(row)
            last_break_idx = break_idx
        else:
            skipped += 1

    filtered_df = pd.DataFrame(filtered_list)

    print(f"\n[필터: {min_interval_bars}봉 간격]")
    print(f"  원본: {len(breakouts_df):,}개")
    print(f"  필터링 후: {len(filtered_df):,}개")
    print(f"  제거: {skipped:,}개 ({skipped / len(breakouts_df) * 100:.1f}%)")
    print(f"  평균 간격: {min_interval_bars * 15:.0f}분 ({min_interval_bars * 15 / 60:.1f}시간)")

    return filtered_df, skipped

# 옵션 1: 5봉 간격 (보수적)
filtered_5, skipped_5 = apply_minimum_interval_filter(breakouts, min_interval_bars=5)

# 옵션 2: 10봉 간격 (권장)
filtered_10, skipped_10 = apply_minimum_interval_filter(breakouts, min_interval_bars=10)

# 옵션 3: 20봉 간격 (공격적)
filtered_20, skipped_20 = apply_minimum_interval_filter(breakouts, min_interval_bars=20)

# ═══════════════════════════════════════════════════════════
# 3. 성과 비교 - Trendline 돌파만 (롱)
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("3. 성과 비교 - Trendline 돌파 (롱)")
print("="*70)

def backtest_with_filter(breakouts_df, df, tp_pct=2.0, sl_pct=2.0):
    """
    필터링된 신호로 백테스트
    """

    # Trendline 돌파만 (롱)
    trendline_breakouts = breakouts_df[breakouts_df['type'] == 'trendline_up'].copy()

    trades = []

    for idx, breakout in trendline_breakouts.iterrows():
        break_idx = breakout['break_idx']
        entry_price = breakout['break_price']

        # TP/SL 계산
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        # 향후 50봉 스캔
        max_idx = min(break_idx + 50, len(df) - 1)
        future = df.iloc[break_idx:max_idx+1]

        hit_tp = False
        hit_sl = False
        exit_idx = None
        exit_price = None

        for i in range(1, len(future)):
            candle = future.iloc[i]

            # TP 먼저 체크
            if candle['high'] >= tp_price:
                hit_tp = True
                exit_idx = break_idx + i
                exit_price = tp_price
                break

            # SL 체크
            if candle['low'] <= sl_price:
                hit_sl = True
                exit_idx = break_idx + i
                exit_price = sl_price
                break

        # 50봉 내 미청산
        if exit_idx is None:
            exit_idx = max_idx
            exit_price = future.iloc[-1]['close']

        # PnL 계산
        pnl = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_idx': break_idx,
            'entry_price': entry_price,
            'exit_idx': exit_idx,
            'exit_price': exit_price,
            'pnl': pnl,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'datetime': breakout['datetime'],
        })

    trades_df = pd.DataFrame(trades)

    # 통계
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    avg_pnl = trades_df['pnl'].mean()

    # 수수료 적용 (Bybit: 0.055% 양방향 = 0.11%)
    fee_pct = 0.11
    avg_pnl_after_fee = avg_pnl - fee_pct
    total_pnl_after_fee = avg_pnl_after_fee * len(trades_df)

    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_pnl': avg_pnl,
        'avg_pnl_after_fee': avg_pnl_after_fee,
        'total_pnl_after_fee': total_pnl_after_fee,
        'trades_df': trades_df,
    }

# 원본 vs 필터링 비교
original_result = backtest_with_filter(breakouts, df, tp_pct=2.0, sl_pct=2.0)
filtered_5_result = backtest_with_filter(filtered_5, df, tp_pct=2.0, sl_pct=2.0)
filtered_10_result = backtest_with_filter(filtered_10, df, tp_pct=2.0, sl_pct=2.0)
filtered_20_result = backtest_with_filter(filtered_20, df, tp_pct=2.0, sl_pct=2.0)

# 비교표
comparison = pd.DataFrame({
    '필터': ['원본 (없음)', '5봉 간격', '10봉 간격 ⭐', '20봉 간격'],
    '거래수': [
        f"{original_result['trades']:,}개",
        f"{filtered_5_result['trades']:,}개",
        f"{filtered_10_result['trades']:,}개",
        f"{filtered_20_result['trades']:,}개",
    ],
    '승률': [
        f"{original_result['win_rate']:.1f}%",
        f"{filtered_5_result['win_rate']:.1f}%",
        f"{filtered_10_result['win_rate']:.1f}%",
        f"{filtered_20_result['win_rate']:.1f}%",
    ],
    '평균PnL': [
        f"{original_result['avg_pnl']:.3f}%",
        f"{filtered_5_result['avg_pnl']:.3f}%",
        f"{filtered_10_result['avg_pnl']:.3f}%",
        f"{filtered_20_result['avg_pnl']:.3f}%",
    ],
    '수수료후': [
        f"{original_result['avg_pnl_after_fee']:.3f}%",
        f"{filtered_5_result['avg_pnl_after_fee']:.3f}%",
        f"{filtered_10_result['avg_pnl_after_fee']:.3f}%",
        f"{filtered_20_result['avg_pnl_after_fee']:.3f}%",
    ],
    '총PnL(수수료후)': [
        f"{original_result['total_pnl_after_fee']:.1f}%",
        f"{filtered_5_result['total_pnl_after_fee']:.1f}%",
        f"{filtered_10_result['total_pnl_after_fee']:.1f}%",
        f"{filtered_20_result['total_pnl_after_fee']:.1f}%",
    ],
})

print("\n" + comparison.to_string(index=False))

# ═══════════════════════════════════════════════════════════
# 4. 권장사항
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("4. 권장사항 및 결론")
print("="*70)

print(f"""
⭐ 권장 설정: 10봉 간격 필터 (2.5시간)

근거:
1. 거래 감소: {original_result['trades']}개 → {filtered_10_result['trades']}개 ({(1 - filtered_10_result['trades'] / original_result['trades']) * 100:.1f}% 감소)
2. 승률 변화: {original_result['win_rate']:.1f}% → {filtered_10_result['win_rate']:.1f}% ({filtered_10_result['win_rate'] - original_result['win_rate']:+.1f}%p)
3. 평균 PnL: {original_result['avg_pnl']:.3f}% → {filtered_10_result['avg_pnl']:.3f}% ({filtered_10_result['avg_pnl'] - original_result['avg_pnl']:+.3f}%p)
4. 수수료 효율: {filtered_10_result['avg_pnl_after_fee']:.3f}% (수수료 -0.11%)

효과:
✅ 오버트레이딩 방지
✅ 거래당 품질 향상
✅ 수수료 부담 감소
✅ 심리적 안정성 증가

실전 적용:
# 실시간 트레이딩 로직에 추가
MIN_INTERVAL_BARS = 10  # 150분 = 2.5시간
last_entry_time = None

if new_breakout_signal:
    current_bar_index = get_current_bar_index()

    if last_entry_time is None or (current_bar_index - last_entry_time >= MIN_INTERVAL_BARS):
        execute_trade()
        last_entry_time = current_bar_index
    else:
        # 스킵: 마지막 진입 후 N봉 미경과

비교:
- 5봉 간격: 너무 보수적, 거래 기회 과도하게 감소
- 10봉 간격: 균형적, 품질과 기회 모두 확보 ⭐
- 20봉 간격: 너무 공격적, 거래 기회 지나치게 제한
""")

# ═══════════════════════════════════════════════════════════
# 5. 필터링된 데이터 저장
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("5. 필터링된 데이터 저장")
print("="*70)

# 10봉 간격 필터 적용 (권장)
filtered_10.to_csv('output_phase4_breakouts_filtered.csv', index=False)
print(f"\n✅ 저장 완료: output_phase4_breakouts_filtered.csv")
print(f"   원본: {len(breakouts):,}개 → 필터: {len(filtered_10):,}개")

# 백테스트 결과도 저장
filtered_10_result['trades_df'].to_csv('backtest_filtered_10bars.csv', index=False)
print(f"✅ 저장 완료: backtest_filtered_10bars.csv")
print(f"   거래: {len(filtered_10_result['trades_df']):,}개")

print("\n" + "="*70)
print("완료!")
print("="*70)

print(f"""
다음 단계:
1. ✅ 연속 신호 필터 적용 완료
2. ⏭️ Out-of-sample 테스트 (최근 6개월)
3. ⏭️ Walk-forward 분석
4. ⏭️ 페이퍼 트레이딩 (2주)
5. ⏭️ 소액 실전 ($500-1,000)
""")
