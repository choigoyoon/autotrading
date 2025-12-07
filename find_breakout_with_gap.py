"""
추세선 돌파 + Gap 필터링으로 진입 시그널 찾기
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 60)
print("추세선 돌파 시그널 (Gap 필터링)")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

down = pd.read_csv('downtrend_3touch.csv', parse_dates=['h1_time', 'h2_time', 'h2_confirmed'])
up = pd.read_csv('uptrend_3touch.csv', parse_dates=['l1_time', 'l2_time', 'l2_confirmed'])

print(f"15분봉: {len(df):,}개")
print(f"하락추세선 (3점+): {len(down)}개")
print(f"상승추세선 (3점+): {len(up)}개")

def get_trendline_price(p1_time, p1_price, p2_time, p2_price, target_time):
    """추세선 가격"""
    if p2_time == p1_time:
        return p1_price
    time_ratio = (target_time - p1_time).total_seconds() / (p2_time - p1_time).total_seconds()
    return p1_price + time_ratio * (p2_price - p1_price)

# =============================================================================
# 하락추세선 돌파 시그널 (Long)
# =============================================================================
print("\n" + "=" * 60)
print("하락추세선 돌파 시그널 찾기")
print("=" * 60)

long_signals = []

for idx, tl in down.iterrows():
    if idx % 2000 == 0:
        print(f"진행: {idx}/{len(down)}")
    
    h2_confirmed = tl['h2_confirmed']
    if pd.isna(h2_confirmed):
        continue
    
    # H2 확정 후 48시간 내 데이터
    mask = (df['datetime'] > h2_confirmed) & (df['datetime'] <= h2_confirmed + timedelta(hours=48))
    period_df = df[mask]
    
    if len(period_df) < 2:
        continue
    
    # 돌파 찾기
    prev_close = None
    prev_tl = None
    
    for _, row in period_df.iterrows():
        tl_price = get_trendline_price(
            tl['h1_time'], tl['h1_price'],
            tl['h2_time'], tl['h2_price'],
            row['datetime']
        )
        
        if prev_close is not None and prev_tl is not None:
            # 돌파: 이전 종가 < 추세선, 현재 종가 > 추세선
            if prev_close < prev_tl and row['close'] > tl_price:
                gap_pct = (row['close'] - tl_price) / tl_price * 100
                
                long_signals.append({
                    'time': row['datetime'],
                    'price': row['close'],
                    'trendline_price': tl_price,
                    'gap_pct': gap_pct,
                    'touch_count': tl['touch_count'],
                    'h1_time': tl['h1_time'],
                    'h2_time': tl['h2_time']
                })
                break  # 첫 돌파만
        
        prev_close = row['close']
        prev_tl = tl_price

print(f"\n하락추세선 돌파: {len(long_signals)}개")

# =============================================================================
# 상승추세선 이탈 시그널 (Short)
# =============================================================================
print("\n" + "=" * 60)
print("상승추세선 이탈 시그널 찾기")
print("=" * 60)

short_signals = []

for idx, tl in up.iterrows():
    if idx % 2000 == 0:
        print(f"진행: {idx}/{len(up)}")
    
    l2_confirmed = tl['l2_confirmed']
    if pd.isna(l2_confirmed):
        continue
    
    mask = (df['datetime'] > l2_confirmed) & (df['datetime'] <= l2_confirmed + timedelta(hours=48))
    period_df = df[mask]
    
    if len(period_df) < 2:
        continue
    
    prev_close = None
    prev_tl = None
    
    for _, row in period_df.iterrows():
        tl_price = get_trendline_price(
            tl['l1_time'], tl['l1_price'],
            tl['l2_time'], tl['l2_price'],
            row['datetime']
        )
        
        if prev_close is not None and prev_tl is not None:
            # 이탈: 이전 종가 > 추세선, 현재 종가 < 추세선
            if prev_close > prev_tl and row['close'] < tl_price:
                gap_pct = (tl_price - row['close']) / tl_price * 100
                
                short_signals.append({
                    'time': row['datetime'],
                    'price': row['close'],
                    'trendline_price': tl_price,
                    'gap_pct': gap_pct,
                    'touch_count': tl['touch_count'],
                    'l1_time': tl['l1_time'],
                    'l2_time': tl['l2_time']
                })
                break
        
        prev_close = row['close']
        prev_tl = tl_price

print(f"\n상승추세선 이탈: {len(short_signals)}개")

# =============================================================================
# Gap 필터링 결과
# =============================================================================
print("\n" + "=" * 60)
print("Gap 필터링 결과")
print("=" * 60)

years = (df['datetime'].max() - df['datetime'].min()).days / 365

# Long 시그널 Gap별
print("\n[하락추세선 돌파 (Long) - Gap별]")
if long_signals:
    long_df = pd.DataFrame(long_signals)
    
    for gap_max in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 100]:
        cnt = (long_df['gap_pct'] < gap_max).sum()
        print(f"  Gap < {gap_max}%: {cnt}개 (연간 {cnt/years:.1f}개, 월간 {cnt/years/12:.1f}개)")

# Short 시그널 Gap별
print("\n[상승추세선 이탈 (Short) - Gap별]")
if short_signals:
    short_df = pd.DataFrame(short_signals)
    
    for gap_max in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 100]:
        cnt = (short_df['gap_pct'] < gap_max).sum()
        print(f"  Gap < {gap_max}%: {cnt}개 (연간 {cnt/years:.1f}개, 월간 {cnt/years/12:.1f}개)")

# 전체 합계
print("\n[전체 (Long + Short) - Gap별]")
if long_signals and short_signals:
    for gap_max in [0.1, 0.2, 0.3, 0.5, 1.0]:
        long_cnt = (long_df['gap_pct'] < gap_max).sum()
        short_cnt = (short_df['gap_pct'] < gap_max).sum()
        total = long_cnt + short_cnt
        print(f"  Gap < {gap_max}%: {total}개 (연간 {total/years:.1f}개, 월간 {total/years/12:.1f}개)")

# 저장
if long_signals:
    long_df.to_csv('breakout_long_signals.csv', index=False)
    print(f"\nLong 시그널 저장: breakout_long_signals.csv")

if short_signals:
    short_df.to_csv('breakout_short_signals.csv', index=False)
    print(f"Short 시그널 저장: breakout_short_signals.csv")
