"""
추세선 재정의: 3점 이상 터치하는 추세선 찾기

사용자 정의:
- 꼬리(고가/저가)도 터치로 인정
- 3번 이상 닿으면 유효한 추세선
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 60)
print("추세선 재정의: 3점 이상 터치")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

h_values = pd.read_csv('h_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])
l_values = pd.read_csv('l_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])

print(f"15분봉: {len(df):,}개")
print(f"H값: {len(h_values):,}개")
print(f"L값: {len(l_values):,}개")

def get_trendline_price(p1_time, p1_price, p2_time, p2_price, target_time):
    """추세선의 특정 시점 가격"""
    if p2_time == p1_time:
        return p1_price
    time_ratio = (target_time - p1_time).total_seconds() / (p2_time - p1_time).total_seconds()
    return p1_price + time_ratio * (p2_price - p1_price)

def count_touches(p1_time, p1_price, p2_time, p2_price, df, tolerance_pct=0.3, trend_type='down'):
    """
    추세선에 닿는 횟수 계산 (꼬리 포함)
    
    tolerance_pct: 추세선에서 얼마나 가까우면 터치로 인정 (%)
    trend_type: 'down' = 하락추세선(고가 터치), 'up' = 상승추세선(저가 터치)
    """
    # p1 ~ p2 구간의 데이터
    mask = (df['datetime'] >= p1_time) & (df['datetime'] <= p2_time + timedelta(hours=48))
    period_df = df[mask].copy()
    
    if len(period_df) < 3:
        return 0, []
    
    touches = []
    
    for _, row in period_df.iterrows():
        tl_price = get_trendline_price(p1_time, p1_price, p2_time, p2_price, row['datetime'])
        tolerance = tl_price * tolerance_pct / 100
        
        if trend_type == 'down':
            # 하락추세선: 고가가 추세선에 닿으면 터치
            if abs(row['high'] - tl_price) <= tolerance or row['high'] >= tl_price:
                touches.append({
                    'time': row['datetime'],
                    'price': row['high'],
                    'trendline': tl_price,
                    'diff_pct': (row['high'] - tl_price) / tl_price * 100
                })
        else:
            # 상승추세선: 저가가 추세선에 닿으면 터치
            if abs(row['low'] - tl_price) <= tolerance or row['low'] <= tl_price:
                touches.append({
                    'time': row['datetime'],
                    'price': row['low'],
                    'trendline': tl_price,
                    'diff_pct': (row['low'] - tl_price) / tl_price * 100
                })
    
    # 연속 터치는 1번으로 카운트 (4시간 이내)
    unique_touches = []
    last_touch_time = None
    
    for t in touches:
        if last_touch_time is None or (t['time'] - last_touch_time).total_seconds() > 4 * 3600:
            unique_touches.append(t)
            last_touch_time = t['time']
    
    return len(unique_touches), unique_touches

# =============================================================================
# 하락추세선 찾기 (3점 이상 터치)
# =============================================================================
print("\n" + "=" * 60)
print("하락추세선 찾기 (3점 이상 터치)")
print("=" * 60)

valid_downtrend_lines = []

# 모든 H값 쌍에서 추세선 후보 생성
h_list = h_values.sort_values('datetime').reset_index(drop=True)

for i in range(len(h_list) - 1):
    h1 = h_list.iloc[i]
    
    for j in range(i + 1, min(i + 20, len(h_list))):  # 최대 20개 H값 내에서
        h2 = h_list.iloc[j]
        
        # LH 조건: h2 < h1 (하락추세)
        if h2['price'] >= h1['price']:
            continue
        
        # 터치 횟수 계산
        touch_count, touches = count_touches(
            h1['datetime'], h1['price'],
            h2['datetime'], h2['price'],
            df, tolerance_pct=0.3, trend_type='down'
        )
        
        if touch_count >= 3:
            valid_downtrend_lines.append({
                'h1_time': h1['datetime'],
                'h1_price': h1['price'],
                'h2_time': h2['datetime'],
                'h2_price': h2['price'],
                'h2_confirmed': h2['confirmed_at'],
                'touch_count': touch_count,
                'slope_pct': (h2['price'] - h1['price']) / h1['price'] * 100
            })

print(f"3점 이상 터치 하락추세선: {len(valid_downtrend_lines)}개")

# 중복 제거 (비슷한 추세선)
if valid_downtrend_lines:
    down_df = pd.DataFrame(valid_downtrend_lines)
    down_df = down_df.sort_values(['h1_time', 'touch_count'], ascending=[True, False])
    down_df = down_df.drop_duplicates(subset=['h1_time'], keep='first')
    print(f"중복 제거 후: {len(down_df)}개")
    
    # 터치 횟수별 분포
    print("\n터치 횟수별 분포:")
    for tc in sorted(down_df['touch_count'].unique()):
        cnt = (down_df['touch_count'] == tc).sum()
        print(f"  {tc}번 터치: {cnt}개")

# =============================================================================
# 상승추세선 찾기 (3점 이상 터치)
# =============================================================================
print("\n" + "=" * 60)
print("상승추세선 찾기 (3점 이상 터치)")
print("=" * 60)

valid_uptrend_lines = []

l_list = l_values.sort_values('datetime').reset_index(drop=True)

for i in range(len(l_list) - 1):
    l1 = l_list.iloc[i]
    
    for j in range(i + 1, min(i + 20, len(l_list))):
        l2 = l_list.iloc[j]
        
        # HL 조건: l2 > l1 (상승추세)
        if l2['price'] <= l1['price']:
            continue
        
        touch_count, touches = count_touches(
            l1['datetime'], l1['price'],
            l2['datetime'], l2['price'],
            df, tolerance_pct=0.3, trend_type='up'
        )
        
        if touch_count >= 3:
            valid_uptrend_lines.append({
                'l1_time': l1['datetime'],
                'l1_price': l1['price'],
                'l2_time': l2['datetime'],
                'l2_price': l2['price'],
                'l2_confirmed': l2['confirmed_at'],
                'touch_count': touch_count,
                'slope_pct': (l2['price'] - l1['price']) / l1['price'] * 100
            })

print(f"3점 이상 터치 상승추세선: {len(valid_uptrend_lines)}개")

if valid_uptrend_lines:
    up_df = pd.DataFrame(valid_uptrend_lines)
    up_df = up_df.sort_values(['l1_time', 'touch_count'], ascending=[True, False])
    up_df = up_df.drop_duplicates(subset=['l1_time'], keep='first')
    print(f"중복 제거 후: {len(up_df)}개")
    
    print("\n터치 횟수별 분포:")
    for tc in sorted(up_df['touch_count'].unique()):
        cnt = (up_df['touch_count'] == tc).sum()
        print(f"  {tc}번 터치: {cnt}개")

# =============================================================================
# 결과 저장
# =============================================================================
print("\n" + "=" * 60)
print("결과 저장")
print("=" * 60)

if valid_downtrend_lines:
    down_df.to_csv('downtrend_lines_3touch.csv', index=False)
    print(f"하락추세선 저장: downtrend_lines_3touch.csv ({len(down_df)}개)")

if valid_uptrend_lines:
    up_df.to_csv('uptrend_lines_3touch.csv', index=False)
    print(f"상승추세선 저장: uptrend_lines_3touch.csv ({len(up_df)}개)")

# 요약
print("\n" + "=" * 60)
print("요약")
print("=" * 60)
years = (df['datetime'].max() - df['datetime'].min()).days / 365
print(f"기간: {years:.1f}년")
print(f"하락추세선 (3점+): {len(down_df) if valid_downtrend_lines else 0}개 (연간 {len(down_df)/years:.1f}개)" if valid_downtrend_lines else "")
print(f"상승추세선 (3점+): {len(up_df) if valid_uptrend_lines else 0}개 (연간 {len(up_df)/years:.1f}개)" if valid_uptrend_lines else "")
