"""
추세선 재정의: 3점 이상 터치 (최적화 버전)
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 60)
print("추세선 재정의: 3점 이상 터치 (최적화)")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

h_values = pd.read_csv('h_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])
l_values = pd.read_csv('l_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])

print(f"15분봉: {len(df):,}개")
print(f"H값: {len(h_values):,}개")

# numpy 배열로 변환 (속도 향상)
df_times = df['datetime'].values
df_highs = df['high'].values
df_lows = df['low'].values

def count_touches_fast(p1_idx, p1_price, p2_idx, p2_price, trend_type='down', tolerance_pct=0.3):
    """빠른 터치 횟수 계산"""
    if p2_idx <= p1_idx:
        return 0
    
    # 해당 구간만
    end_idx = min(p2_idx + 192, len(df_times))  # +48시간 (192캔들)
    
    touches = 0
    last_touch_idx = -100
    
    for i in range(p1_idx, end_idx):
        # 추세선 가격 계산
        ratio = (i - p1_idx) / (p2_idx - p1_idx) if p2_idx != p1_idx else 0
        tl_price = p1_price + ratio * (p2_price - p1_price)
        tolerance = tl_price * tolerance_pct / 100
        
        if trend_type == 'down':
            price = df_highs[i]
            # 고가가 추세선 근처이거나 위
            if price >= tl_price - tolerance:
                if i - last_touch_idx > 16:  # 4시간 간격
                    touches += 1
                    last_touch_idx = i
        else:
            price = df_lows[i]
            # 저가가 추세선 근처이거나 아래
            if price <= tl_price + tolerance:
                if i - last_touch_idx > 16:
                    touches += 1
                    last_touch_idx = i
    
    return touches

# H값에 df index 매핑
h_values['df_idx'] = h_values['datetime'].apply(
    lambda x: df[df['datetime'] == x].index[0] if len(df[df['datetime'] == x]) > 0 else -1
)
h_values = h_values[h_values['df_idx'] >= 0]

l_values['df_idx'] = l_values['datetime'].apply(
    lambda x: df[df['datetime'] == x].index[0] if len(df[df['datetime'] == x]) > 0 else -1
)
l_values = l_values[l_values['df_idx'] >= 0]

print(f"\nH값 (매핑됨): {len(h_values)}개")
print(f"L값 (매핑됨): {len(l_values)}개")

# =============================================================================
# 하락추세선 찾기 (3점 이상)
# =============================================================================
print("\n" + "=" * 60)
print("하락추세선 찾기 (3점 이상)")
print("=" * 60)

valid_down = []
h_list = h_values.sort_values('datetime').reset_index(drop=True)

total = len(h_list)
for i in range(total - 1):
    if i % 500 == 0:
        print(f"진행: {i}/{total}")
    
    h1 = h_list.iloc[i]
    
    for j in range(i + 1, min(i + 10, total)):  # 최대 10개 내에서
        h2 = h_list.iloc[j]
        
        # LH 조건
        if h2['price'] >= h1['price']:
            continue
        
        touch_count = count_touches_fast(
            int(h1['df_idx']), h1['price'],
            int(h2['df_idx']), h2['price'],
            trend_type='down'
        )
        
        if touch_count >= 3:
            valid_down.append({
                'h1_time': h1['datetime'],
                'h1_price': h1['price'],
                'h2_time': h2['datetime'],
                'h2_price': h2['price'],
                'h2_confirmed': h2['confirmed_at'],
                'touch_count': touch_count
            })

print(f"\n3점+ 하락추세선: {len(valid_down)}개")

# =============================================================================
# 상승추세선 찾기 (3점 이상)
# =============================================================================
print("\n" + "=" * 60)
print("상승추세선 찾기 (3점 이상)")
print("=" * 60)

valid_up = []
l_list = l_values.sort_values('datetime').reset_index(drop=True)

total = len(l_list)
for i in range(total - 1):
    if i % 500 == 0:
        print(f"진행: {i}/{total}")
    
    l1 = l_list.iloc[i]
    
    for j in range(i + 1, min(i + 10, total)):
        l2 = l_list.iloc[j]
        
        # HL 조건
        if l2['price'] <= l1['price']:
            continue
        
        touch_count = count_touches_fast(
            int(l1['df_idx']), l1['price'],
            int(l2['df_idx']), l2['price'],
            trend_type='up'
        )
        
        if touch_count >= 3:
            valid_up.append({
                'l1_time': l1['datetime'],
                'l1_price': l1['price'],
                'l2_time': l2['datetime'],
                'l2_price': l2['price'],
                'l2_confirmed': l2['confirmed_at'],
                'touch_count': touch_count
            })

print(f"\n3점+ 상승추세선: {len(valid_up)}개")

# =============================================================================
# 결과 저장 및 요약
# =============================================================================
print("\n" + "=" * 60)
print("결과")
print("=" * 60)

years = (df['datetime'].max() - df['datetime'].min()).days / 365

if valid_down:
    down_df = pd.DataFrame(valid_down)
    down_df.to_csv('downtrend_3touch.csv', index=False)
    print(f"하락추세선 (3점+): {len(down_df)}개")
    print(f"  → 연간 {len(down_df)/years:.1f}개")
    print(f"  → 월간 {len(down_df)/years/12:.1f}개")

if valid_up:
    up_df = pd.DataFrame(valid_up)
    up_df.to_csv('uptrend_3touch.csv', index=False)
    print(f"상승추세선 (3점+): {len(up_df)}개")
    print(f"  → 연간 {len(up_df)/years:.1f}개")
    print(f"  → 월간 {len(up_df)/years/12:.1f}개")

total_lines = len(valid_down) + len(valid_up)
print(f"\n전체 추세선: {total_lines}개")
print(f"  → 연간 {total_lines/years:.1f}개")
print(f"  → 월간 {total_lines/years/12:.1f}개")
